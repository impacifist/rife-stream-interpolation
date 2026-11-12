"""RIFE frame interpolation that streams frames between FFmpeg processes.

Install this directory under ComfyUI/custom_nodes. The implementation uses
ComfyUI's IFNet and model patcher, while keeping decoded and generated video
frames out of a full-length IMAGE batch in host RAM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths
import numpy as np
import torch
from comfy import model_management
from comfy_extras.frame_interpolation_models.ifnet import IFNet, detect_rife_config

logger = logging.getLogger("rife_stream_interpolation")


@contextmanager
def _fast_inference_backends(device: torch.device, allow_tf32: bool):
    """Enable fixed-shape cuDNN tuning and TF32 for FP32 CUDA inference."""
    if device.type != "cuda":
        yield
        return

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        with torch.backends.cudnn.flags(benchmark=True, allow_tf32=allow_tf32):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    """Read one raw frame, distinguishing clean EOF from a truncated frame."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            if not chunks:
                return None
            raise RuntimeError("FFmpeg ended in the middle of a decoded video frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _probe_video(video_path: Path, ffprobe: str) -> tuple[int, int, Fraction]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffprobe executable not found: {ffprobe}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"Could not inspect input video: {detail}") from exc

    try:
        streams = json.loads(completed.stdout)["streams"]
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
        fps = None
        for rate_text in (stream.get("avg_frame_rate"), stream.get("r_frame_rate")):
            try:
                candidate = Fraction(rate_text)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if candidate > 0:
                fps = candidate
                break
        if fps is None:
            raise ValueError("Video frame rate is unavailable")
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError("Could not determine video dimensions and frame rate") from exc

    if width < 1 or height < 1 or fps <= 0:
        raise RuntimeError("Input video has invalid dimensions or frame rate")
    return width, height, fps


def _prepare_frame(
    frame: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    align: int,
    channels_last: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype).div_(255.0)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    if align > 1:
        from comfy.ldm.common_dit import pad_to_patch_size

        tensor = pad_to_patch_size(tensor, (align, align), padding_mode="reflect")
        if channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def _tensor_to_rgb8(tensor: torch.Tensor, height: int, width: int) -> np.ndarray:
    return (
        tensor[0, :, :height, :width]
        .permute(1, 2, 0)
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(device="cpu", dtype=torch.uint8)
        .numpy()
    )


def _interpolate_pair_group(
    frame_bytes: list[bytes],
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
    align: int,
    channels_last: bool,
    timesteps: list[float],
    model,
) -> list[BinaryIO]:
    """Run a bounded group of adjacent pairs as a GPU batch."""
    frames = [np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3) for raw in frame_bytes]
    tensors = [_prepare_frame(frame, device, dtype, align, channels_last) for frame in frames]
    all_frames = torch.cat(tensors, dim=0)
    del tensors

    features = model.extract_features(all_frames)
    pair_count = len(frame_bytes) - 1
    feature_cache = {"img0": features[:-1], "img1": features[1:]}
    del features
    outputs = [tempfile.TemporaryFile(mode="w+b") for _ in range(pair_count)]  # noqa: SIM115
    try:
        for timestep in timesteps:
            timestep_tensor = torch.full(
                (pair_count, 1, all_frames.shape[2], all_frames.shape[3]),
                timestep,
                device=device,
                dtype=dtype,
            )
            if channels_last:
                timestep_tensor = timestep_tensor.contiguous(memory_format=torch.channels_last)
            middle = model(
                all_frames[:-1],
                all_frames[1:],
                timestep=timestep_tensor,
                cache=feature_cache,
            )
            for pair_index in range(pair_count):
                outputs[pair_index].write(
                    _tensor_to_rgb8(middle[pair_index : pair_index + 1], height, width).tobytes()
                )
            del middle, timestep_tensor

        for output in outputs:
            output.seek(0)
        return outputs
    except Exception:
        for output in outputs:
            output.close()
        raise


def _make_output_paths(video_path: Path, output_name: str, multiplier: int) -> tuple[Path, Path]:
    output_dir = Path(folder_paths.get_output_directory())
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = output_name.strip()
    if requested:
        if Path(requested).name != requested or requested in {".", ".."}:
            raise ValueError("Output name must be a filename, not a path")
        if Path(requested).suffix.lower() != ".mp4":
            raise ValueError("Output filename must end in .mp4")
        stem = Path(requested).stem
    else:
        stem = f"{video_path.stem}_rife_{multiplier}x"

    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._") or "interpolated"
    unique = uuid.uuid4().hex[:8]
    final_path = output_dir / f"{stem}_{unique}.mp4"
    temporary_path = output_dir / f".{stem}_{unique}.partial.mp4"
    return final_path, temporary_path


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None and not pipe.closed:
            pipe.close()


class RIFEStreamModelLoader:
    """Load a ComfyUI-compatible RIFE IFNet checkpoint in the selected dtype."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("frame_interpolation"),),
                "precision": (["bf16", "fp32_tf32", "fp32_strict"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("INTERP_MODEL",)
    RETURN_NAMES = ("interp_model",)
    FUNCTION = "load_model"
    CATEGORY = "loaders/video"

    @classmethod
    def load_model(cls, model_name: str, precision: str):
        model_path = folder_paths.get_full_path_or_raise("frame_interpolation", model_name)
        state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)

        # Match ComfyUI's built-in RIFE checkpoint normalization. This accepts
        # checkpoints whose keys are prefixed with module./flownet. or blockN.
        state_dict = comfy.utils.state_dict_prefix_replace(
            state_dict, {"module.": "", "flownet.": ""}
        )
        key_map: dict[str, str] = {}
        for key in state_dict:
            for index in range(5):
                prefix = f"block{index}."
                if key.startswith(prefix):
                    key_map[key] = f"blocks.{index}.{key[len(prefix) :]}"
        if key_map:
            state_dict = {key_map.get(key, key): value for key, value in state_dict.items()}
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(("teacher.", "caltime."))
        }

        try:
            head_channels, channels = detect_rife_config(state_dict)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Unsupported RIFE checkpoint. Use a ComfyUI-compatible RIFE v4.26+ "
                "checkpoint with the five-block IFNet architecture."
            ) from exc

        model = IFNet(head_ch=head_channels, channels=channels)
        model.load_state_dict(state_dict)

        device = model_management.get_torch_device()
        if precision not in {"bf16", "fp32_tf32", "fp32_strict"}:
            raise ValueError(f"Unsupported RIFE precision: {precision}")
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if precision == "bf16" and not model_management.should_use_bf16(device):
            raise RuntimeError(f"BF16 is not supported on the selected ComfyUI device: {device}")

        model.eval().to(dtype=dtype)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        model._rife_allow_tf32 = precision == "fp32_tf32"
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=device,
            offload_device=model_management.unet_offload_device(),
            fast_disk=comfy.storage.state_dict_fast_disk(state_dict),
        )
        return (patcher,)


class RIFEStreamInterpolate:
    """Interpolate a video from disk and write frames incrementally to MP4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL",),
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
                "pair_batch_size": ("INT", {"default": 4, "min": 1, "max": 16}),
            },
            "optional": {
                "output_name": ("STRING", {"default": "", "multiline": False}),
                "ffmpeg": ("STRING", {"default": "ffmpeg", "multiline": False}),
                "ffprobe": ("STRING", {"default": "ffprobe", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video",)
    FUNCTION = "interpolate"
    CATEGORY = "video/interpolation"
    OUTPUT_NODE = True

    @classmethod
    def interpolate(
        cls,
        interp_model,
        video_path: str,
        multiplier: int,
        pair_batch_size: int = 4,
        output_name: str = "",
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
    ):
        source = Path(video_path.strip()).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source}")
        if multiplier < 2 or multiplier > 16:
            raise ValueError("Multiplier must be between 2 and 16")
        if pair_batch_size < 1 or pair_batch_size > 16:
            raise ValueError("Pair batch size must be between 1 and 16")
        if not ffmpeg or not ffprobe:
            raise ValueError("Set both FFmpeg and ffprobe executable names or paths")

        width, height, fps = _probe_video(source, ffprobe)
        output_fps = fps * multiplier
        final_path, temporary_path = _make_output_paths(source, output_name, multiplier)

        device = interp_model.load_device
        dtype = interp_model.model_dtype()
        inference_model = interp_model.model
        align = max(1, int(getattr(inference_model, "pad_align", 1)))
        padded_height = ((height + align - 1) // align) * align
        padded_width = ((width + align - 1) // align) * align
        activation_memory = (
            inference_model.memory_used_forward(
                (pair_batch_size, padded_height, padded_width, 3), dtype
            )
            * pair_batch_size
        )
        model_management.load_models_gpu([interp_model], memory_required=activation_memory)

        decode_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-noautorotate",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        encode_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{output_fps.numerator}/{output_fps.denominator}",
            "-i",
            "pipe:0",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "apad",
            "-shortest",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]

        decoder: subprocess.Popen[bytes] | None = None
        encoder: subprocess.Popen[bytes] | None = None
        output_frames = 0
        frame_size = width * height * 3
        final_frame_bytes: bytes | None = None

        try:
            try:
                encoder = subprocess.Popen(  # noqa: S603
                    encode_command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )  # noqa: S603
                decoder = subprocess.Popen(  # noqa: S603
                    decode_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )  # noqa: S603
            except FileNotFoundError as exc:
                raise RuntimeError(f"FFmpeg executable not found: {exc.filename}") from exc

            assert decoder.stdout is not None
            assert encoder.stdin is not None
            first_bytes = _read_exact(decoder.stdout, frame_size)
            if first_bytes is None:
                raise RuntimeError("Input video contains no decodable frames")
            encoder.stdin.write(first_bytes)
            output_frames = 1

            use_channels_last = device.type == "cuda"
            frame_buffer = [first_bytes]
            timestep_values = [step / multiplier for step in range(1, multiplier)]
            batch_size = pair_batch_size
            decoder_finished = False
            with (
                torch.inference_mode(),
                _fast_inference_backends(
                    device, bool(getattr(inference_model, "_rife_allow_tf32", False))
                ),
            ):
                while True:
                    while len(frame_buffer) < batch_size + 1 and not decoder_finished:
                        frame_bytes = _read_exact(decoder.stdout, frame_size)
                        if frame_bytes is None:
                            decoder_finished = True
                            break
                        frame_buffer.append(frame_bytes)

                    pair_count = min(batch_size, len(frame_buffer) - 1)
                    if pair_count == 0:
                        break
                    group = frame_buffer[: pair_count + 1]
                    try:
                        interpolated = _interpolate_pair_group(
                            group,
                            width,
                            height,
                            device,
                            dtype,
                            align,
                            use_channels_last,
                            timestep_values,
                            inference_model,
                        )
                    except Exception as exc:
                        if not model_management.is_oom(exc) or pair_count <= 1:
                            raise
                        del exc
                        batch_size = max(1, pair_count // 2)
                        model_management.soft_empty_cache()
                        logger.warning(
                            "RIFE ran out of device memory; reducing adjacent-pair batch to %d",
                            batch_size,
                        )
                        continue

                    for pair_index, middle_frames in enumerate(interpolated):
                        for _ in timestep_values:
                            middle_frame = _read_exact(middle_frames, frame_size)
                            if middle_frame is None:
                                raise RuntimeError("Incomplete interpolation batch output")
                            encoder.stdin.write(middle_frame)
                            output_frames += 1
                        encoder.stdin.write(group[pair_index + 1])
                        output_frames += 1
                        middle_frames.close()

                    final_frame_bytes = group[-1]
                    del frame_buffer[:pair_count]

            # Hold the terminal frame for the remaining input-frame interval so
            # the CFR output duration stays aligned with the original audio.
            if final_frame_bytes is not None:
                for _ in range(multiplier - 1):
                    encoder.stdin.write(final_frame_bytes)
                    output_frames += 1

            decoder.stdout.close()
            decoder_code = decoder.wait()
            encoder.stdin.close()
            encoder_code = encoder.wait()
            if decoder_code != 0:
                raise RuntimeError(f"FFmpeg video decoding failed with exit code {decoder_code}")
            if encoder_code != 0:
                raise RuntimeError(f"FFmpeg video encoding failed with exit code {encoder_code}")
            if output_frames < 2:
                raise RuntimeError("Input video must contain at least two frames")

            os.replace(temporary_path, final_path)
        except Exception:
            _stop_process(decoder)
            _stop_process(encoder)
            temporary_path.unlink(missing_ok=True)
            raise

        logger.info(
            "RIFE stream interpolation complete: %s frames, %s fps, effective adjacent-pair batch=%s, output=%s",
            output_frames,
            output_fps,
            batch_size,
            final_path,
        )
        result = str(final_path)
        return {"ui": {"text": [result]}, "result": (result,)}


NODE_CLASS_MAPPINGS = {
    "RIFEStreamModelLoader": RIFEStreamModelLoader,
    "RIFEStreamInterpolate": RIFEStreamInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFEStreamModelLoader": "Load RIFE v4.26+ (BF16 / TF32 / FP32)",
    "RIFEStreamInterpolate": "RIFE Stream Frame Interpolation",
}
