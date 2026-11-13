"""IMAGE-compatible RIFE node for ComfyUI FrameInterpolate workflows."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import weakref

import comfy.utils
import torch
from comfy import model_management

from .rife_stream_interpolation import _fast_inference_backends

logger = logging.getLogger("rife_stream_interpolation")


def _remove_tensor_file(path: str) -> None:
    """Remove the backing file once all tensor views have released the map."""

    def remove_when_unmapped() -> None:
        for _ in range(240):
            try:
                os.unlink(path)
                return
            except FileNotFoundError:
                return
            except OSError:
                # Windows refuses unlink while an IMAGE view still maps the file.
                time.sleep(0.25)

    threading.Thread(target=remove_when_unmapped, daemon=True).start()


def _allocate_images(
    shape: tuple[int, int, int, int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Use ComfyUI's intermediate device, backing CPU IMAGE data on disk."""
    if device.type != "cpu":
        return torch.empty(shape, dtype=dtype, device=device)

    descriptor, path = tempfile.mkstemp(prefix="comfy-rife-images-", suffix=".tensor")
    os.close(descriptor)
    try:
        elements = 1
        for dimension in shape:
            elements *= dimension
        element_size = torch.empty((), dtype=dtype).element_size()
        os.truncate(path, elements * element_size)
        mapped = torch.from_file(path, shared=True, size=elements, dtype=dtype)
        tensor = mapped.view(shape)
    except Exception:
        _remove_tensor_file(path)
        raise
    # Retain cleanup until every returned slice/view releases this shared
    # storage, then retry unlink for Windows' mmap close timing.
    weakref.finalize(mapped, _remove_tensor_file, path)
    return tensor


class RIFEFrameInterpolate:
    """Drop-in IMAGE interface for ComfyUI's FrameInterpolate node."""

    _INITIAL_PAIR_BATCH = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL",),
                "images": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
            },
            "optional": {
                "multiplier_input": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": "Connected INT overrides the multiplier widget.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video"

    @classmethod
    def interpolate(
        cls,
        interp_model,
        images: torch.Tensor,
        multiplier: int,
        multiplier_input: int | None = None,
    ):
        effective_multiplier = multiplier if multiplier_input is None else multiplier_input
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images must have ComfyUI IMAGE shape [frames, height, width, 3]")
        frame_count, height, width, _ = images.shape
        if frame_count < 2:
            return (images,)
        if not 2 <= effective_multiplier <= 16:
            raise ValueError("Multiplier must be between 2 and 16")
        multiplier = effective_multiplier

        total_pairs = frame_count - 1
        interpolants_per_pair = multiplier - 1
        total_output_frames = total_pairs * multiplier + 1
        output_dtype = model_management.intermediate_dtype()
        # The required IMAGE output is returned as a CPU tensor backed by a
        # temporary file, avoiding an equally large resident output allocation.
        output = _allocate_images(
            (total_output_frames, height, width, 3),
            output_dtype,
            model_management.intermediate_device(),
        )
        output[0].copy_(images[0].to(device=output.device, dtype=output_dtype).clamp(0.0, 1.0))

        device = interp_model.load_device
        dtype = interp_model.model_dtype()
        model = interp_model.model
        align = max(1, int(getattr(model, "pad_align", 1)))
        padded_height = ((height + align - 1) // align) * align
        padded_width = ((width + align - 1) // align) * align
        pair_batch = min(cls._INITIAL_PAIR_BATCH, total_pairs)
        required_memory = (
            model.memory_used_forward((pair_batch, padded_height, padded_width, 3), dtype)
            * pair_batch
        )
        model_management.load_models_gpu([interp_model], memory_required=required_memory)

        timestep_values = [step / multiplier for step in range(1, multiplier)]
        progress = comfy.utils.ProgressBar(total_pairs * interpolants_per_pair)
        channels_last = device.type == "cuda"

        try:
            with (
                torch.inference_mode(),
                _fast_inference_backends(device, bool(getattr(model, "_rife_allow_tf32", False))),
            ):
                pair_start = 0
                while pair_start < total_pairs:
                    pair_count = min(pair_batch, total_pairs - pair_start)
                    source_group = None
                    feature_batch = None
                    cache = None
                    timestep_batch = None
                    interpolated = None
                    try:
                        # Copy the small working set to avoid modifying the
                        # caller's input if input tensors happen to be views.
                        source_group = (
                            images[pair_start : pair_start + pair_count + 1]
                            .movedim(-1, 1)
                            .to(device=device, dtype=dtype)
                            .clone()
                        )
                        source_group.clamp_(0.0, 1.0)

                        if align > 1:
                            from comfy.ldm.common_dit import pad_to_patch_size

                            source_group = pad_to_patch_size(
                                source_group, (align, align), padding_mode="reflect"
                            )
                        if channels_last:
                            source_group = source_group.contiguous(
                                memory_format=torch.channels_last
                            )

                        feature_batch = model.extract_features(source_group)
                        cache = {
                            "img0": feature_batch[:-1],
                            "img1": feature_batch[1:],
                        }
                        del feature_batch

                        for step_index, timestep in enumerate(timestep_values):
                            timestep_batch = torch.full(
                                (pair_count, 1, source_group.shape[2], source_group.shape[3]),
                                timestep,
                                device=device,
                                dtype=dtype,
                            )
                            if channels_last:
                                timestep_batch = timestep_batch.contiguous(
                                    memory_format=torch.channels_last
                                )
                            interpolated = model(
                                source_group[:-1],
                                source_group[1:],
                                timestep=timestep_batch,
                                cache=cache,
                            )
                            for local_pair in range(pair_count):
                                output_index = (
                                    (pair_start + local_pair) * multiplier + step_index + 1
                                )
                                frame = (
                                    interpolated[local_pair, :, :height, :width]
                                    .movedim(0, -1)
                                    .to(device=output.device, dtype=output_dtype)
                                    .clamp_(0.0, 1.0)
                                )
                                output[output_index].copy_(frame)
                            del interpolated, timestep_batch

                        for local_frame in range(1, pair_count + 1):
                            output_index = (pair_start + local_frame) * multiplier
                            output[output_index].copy_(
                                images[pair_start + local_frame]
                                .to(device=output.device, dtype=output_dtype)
                                .clamp(0.0, 1.0)
                            )

                        progress.update(pair_count * interpolants_per_pair)
                        del source_group, cache
                        pair_start += pair_count
                    except Exception as exc:
                        if not model_management.is_oom(exc) or pair_count <= 1:
                            raise
                        del exc
                        source_group = None
                        feature_batch = None
                        cache = None
                        timestep_batch = None
                        interpolated = None
                        pair_batch = max(1, pair_count // 2)
                        model_management.soft_empty_cache()
                        logger.warning(
                            "RIFE IMAGE inference OOM; reducing pair batch size to %d",
                            pair_batch,
                        )
        finally:
            del progress

        return (output,)


NODE_CLASS_MAPPINGS = {"RIFEFrameInterpolate": RIFEFrameInterpolate}
NODE_DISPLAY_NAME_MAPPINGS = {"RIFEFrameInterpolate": "RIFE Frame Interpolation (IMAGE In/Out)"}
