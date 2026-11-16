# RIFE Frame Interpolation for ComfyUI

Two ComfyUI nodes for RIFE v4.26+: a model loader and an IMAGE-batch frame interpolation node with a multiplier and inference-chunk control.

## Nodes and workflow

1. **Load RIFE v4.26+ (FP16 / BF16 / FP32)** loads a compatible IFNet checkpoint from `ComfyUI/models/frame_interpolation` and outputs `interp_model`.
2. Connect the loader's `interp_model` output to **RIFE Frame Interpolation** → `interp_model`.
3. Connect your video loader's IMAGE frame batch to `images`.
4. Set `multiplier` (2–16) and `chunk_size` (1–16 adjacent frame pairs). Connect the returned IMAGE batch downstream.

## Precision

- `fp16`: Default. Uses FP16 weights and the main inference path when supported by the selected device.
- `bf16`: Uses BF16 weights and the main inference path; requires device support.
- `fp32_tf32`: FP32 model tensors, with CUDA TF32 enabled for supported cuDNN/cuBLAS operations.
- `fp32_strict`: FP32 with TF32 disabled.

The selected precision applies to model weights and the main inference path. Some auxiliary flow-warp/grid-sampling operations use FP32 internally for sampling before converting the result back to the model dtype.

## GPU memory and chunking

The loader places the model on ComfyUI's selected compute device and wraps it with `CoreModelPatcher`. ComfyUI's model manager applies its active VRAM/offload policy; the full model is not necessarily pinned in VRAM in every configuration. CUDA inference uses channels-last tensors and cuDNN autotuning.

`chunk_size` controls how many neighboring frame pairs are inferred together (default 4, range 1–16). Smaller chunks reduce peak activation memory; larger chunks can improve throughput if VRAM permits. On VRAM OOM, the node halves the chunk and retries. CPU intermediate output uses a temporary disk-backed tensor, while the input IMAGE batch and downstream nodes still use memory.

## Install

Copy this repository directory into `ComfyUI/custom_nodes/rife_stream_interpolation`, put a ComfyUI-compatible RIFE v4.26+ IFNet checkpoint in `ComfyUI/models/frame_interpolation`, and restart ComfyUI. See [README.ko.md](README.ko.md) for Korean instructions.
