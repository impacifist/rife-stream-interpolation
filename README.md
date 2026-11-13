# RIFE Stream Frame Interpolation for ComfyUI

A ComfyUI custom node set for RIFE v4.26+ that includes both an `IMAGE`-compatible replacement for ComfyUI's built-in frame interpolation node and a bounded-memory video-file streaming node.

## Features

- BF16, FP32 with TF32, and strict FP32 inference modes.
- Adjacent-pair GPU batching, CUDA channels-last layout, and cuDNN autotuning.
- Automatic reduction of the pair batch size after a VRAM OOM.
- FFmpeg-based frame streaming with H.264 output and optional AAC audio passthrough/transcoding.
- ComfyUI model patcher and memory manager integration.
- An IMAGE-in/IMAGE-out node with the same `interp_model`, `images`, and `multiplier` sockets as ComfyUI's built-in `FrameInterpolate`.
- Optional `multiplier_input` INT socket for a value calculated by a separate math node.

## Install

Copy this repository directory into `ComfyUI/custom_nodes/rife_stream_interpolation`, place a ComfyUI-compatible RIFE v4.26+ checkpoint in `ComfyUI/models/frame_interpolation`, ensure `ffmpeg` and `ffprobe` are on PATH, and restart ComfyUI. See [README.ko.md](README.ko.md) for setup and usage in Korean.

The custom nodes depend on ComfyUI's `comfy_extras.frame_interpolation_models.ifnet` implementation and are intended to be installed into a ComfyUI checkout. The checkpoint must match ComfyUI's five-block IFNet architecture.

## Precision

- `bf16`: BF16 weights and activations; requires device support.
- `fp32_tf32`: FP32 tensors with CUDA TF32 allowed for cuDNN/cuBLAS operations.
- `fp32_strict`: FP32 with TF32 disabled.

The `pair_batch_size` widget defaults to 4. Increase it to raise GPU utilization if VRAM permits, or lower it for high-resolution inputs. Actual throughput and TF32 behavior depend on the GPU, drivers, PyTorch build, and ComfyUI configuration.

For existing IMAGE-batch workflows, search for **RIFE Frame Interpolation (IMAGE In/Out)** and replace the built-in node while reconnecting the same model, frame-batch, and multiplier links. It returns an IMAGE batch; CPU intermediate output is disk-backed although the input batch still occupies memory. Use **RIFE Stream Frame Interpolation** for the lowest RAM footprint with long videos.

To supply a multiplier from a separate math node, connect its `INT` output to the optional `multiplier_input` socket. When connected, that value overrides the multiplier widget; otherwise the widget is used. The valid multiplier range is 2–16.
