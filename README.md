# RIFE Frame Interpolation for ComfyUI

A two-node RIFE v4.26+ ComfyUI frame interpolation package: a model loader and an IMAGE-batch interpolation node with a multiplier widget, following the built-in `FrameInterpolate` workflow.

## Nodes

1. **Load RIFE v4.26+ (FP16 / BF16 / FP32)** loads a compatible IFNet checkpoint and lets you choose FP16, BF16, TF32-enabled FP32, or strict FP32.
2. **RIFE Frame Interpolation** accepts the model, a video frame `IMAGE` batch, a multiplier (2–16), and `chunk_size` (1–16 adjacent frame pairs), then returns the interpolated `IMAGE` batch.

Connect the model loader's `interp_model` output to the interpolation node's `interp_model` input. Connect a video loader's IMAGE frame batch to `images`, set the `multiplier` widget, and pass the resulting IMAGE batch downstream.

## Install

Copy this repository directory into `ComfyUI/custom_nodes/rife_stream_interpolation`, place a compatible RIFE v4.26+ checkpoint in `ComfyUI/models/frame_interpolation`, and restart ComfyUI. See [README.ko.md](README.ko.md) for Korean setup instructions.

## Precision

- `fp16`: FP16 weights and activations; requires device support.
- `bf16`: BF16 weights and activations; requires device support.
- `fp32_tf32`: FP32 tensors with CUDA TF32 allowed for cuDNN/cuBLAS operations.
- `fp32_strict`: FP32 with TF32 disabled.

The `chunk_size` widget controls how many adjacent frame pairs are processed together. Smaller chunks lower peak VRAM use; larger chunks may improve throughput. CUDA inference uses channels-last and cuDNN autotuning; VRAM OOM reduces the pair chunk and retries. Actual performance depends on GPU, drivers, PyTorch, and ComfyUI configuration. Inputs and outputs retain IMAGE-batch memory behavior.
