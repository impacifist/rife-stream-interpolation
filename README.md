# RIFE Stream Frame Interpolation for ComfyUI

A ComfyUI custom node for RIFE v4.26+ that interpolates videos as a bounded frame stream and writes MP4 output directly, avoiding a full-length `IMAGE` batch in system RAM.

## Features

- BF16, FP32 with TF32, and strict FP32 inference modes.
- Adjacent-pair GPU batching, CUDA channels-last layout, and cuDNN autotuning.
- Automatic reduction of the pair batch size after a VRAM OOM.
- FFmpeg-based frame streaming with H.264 output and optional AAC audio passthrough/transcoding.
- ComfyUI model patcher and memory manager integration.

## Install

Copy this repository directory into `ComfyUI/custom_nodes/rife_stream_interpolation`, place a ComfyUI-compatible RIFE v4.26+ checkpoint in `ComfyUI/models/frame_interpolation`, ensure `ffmpeg` and `ffprobe` are on PATH, and restart ComfyUI. See [README.ko.md](README.ko.md) for setup and usage in Korean.

The custom node depends on ComfyUI's `comfy_extras.frame_interpolation_models.ifnet` implementation and is intended to be installed into a ComfyUI checkout. The checkpoint must match ComfyUI's five-block IFNet architecture.

## Precision

- `bf16`: BF16 weights and activations; requires device support.
- `fp32_tf32`: FP32 tensors with CUDA TF32 allowed for cuDNN/cuBLAS operations.
- `fp32_strict`: FP32 with TF32 disabled.

The `pair_batch_size` widget defaults to 4. Increase it to raise GPU utilization if VRAM permits, or lower it for high-resolution inputs. Actual throughput and TF32 behavior depend on the GPU, drivers, PyTorch build, and ComfyUI configuration.
