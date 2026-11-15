# RIFE Frame Interpolation for ComfyUI

A two-node RIFE v4.26+ ComfyUI frame interpolation package: a model loader and an IMAGE-batch interpolation node with a multiplier widget, following the built-in `FrameInterpolate` workflow.

## Nodes

1. **Load RIFE v4.26+ (BF16 / TF32 / FP32)** loads a ComfyUI-compatible IFNet checkpoint from `ComfyUI/models/frame_interpolation` and returns `interp_model`.
2. **RIFE Frame Interpolation** accepts that model, a video frame `IMAGE` batch, and a multiplier widget (2–16), then returns the interpolated `IMAGE` batch.

The interpolation node follows the built-in `FrameInterpolate` input/output contract. Calculate source/target FPS and the resulting multiplier in an external math node if needed, then set the multiplier widget. This package exposes only the model loader and frame interpolation nodes.

## Install

Copy this repository directory into `ComfyUI/custom_nodes/rife_stream_interpolation`, place a compatible RIFE v4.26+ checkpoint in `ComfyUI/models/frame_interpolation`, and restart ComfyUI. See [README.ko.md](README.ko.md) for Korean setup instructions.

## Precision

- `bf16`: BF16 weights and activations; requires device support.
- `fp32_tf32`: FP32 tensors with CUDA TF32 allowed for cuDNN/cuBLAS operations.
- `fp32_strict`: FP32 with TF32 disabled.

The inference node uses adjacent frame-pair batches, CUDA channels-last layout, cuDNN autotuning, and reduces the pair batch after VRAM OOM. Actual throughput depends on the GPU, drivers, PyTorch, and ComfyUI configuration. Inputs and outputs follow ComfyUI's IMAGE-batch memory behavior.
