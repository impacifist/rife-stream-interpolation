# RIFE Frame Interpolation for ComfyUI

[English](README.md) | [한국어](README.ko.md)

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

## 한국어

RIFE v4.26+용 ComfyUI 노드 두 개를 제공합니다: **모델 로더**와 **IMAGE 배치 프레임 보간** 노드입니다.

### 연결 방법

1. `Load RIFE v4.26+ (FP16 / BF16 / FP32)`에서 체크포인트와 연산 정밀도를 선택합니다. 모델은 `ComfyUI/models/frame_interpolation`에 둡니다.
2. 모델 로더의 `interp_model` 출력을 `RIFE Frame Interpolation`의 `interp_model` 입력에 연결합니다.
3. 비디오 로더의 IMAGE 프레임 배치를 `images`에 연결하고, `multiplier`(2–16)를 설정합니다.
4. `chunk_size`(1–16)는 한 번에 처리할 인접 프레임 쌍 수입니다. 작은 값은 peak VRAM 사용량을 낮추고, 큰 값은 메모리가 허용할 때 처리량을 높일 수 있습니다. OOM이면 chunk를 절반으로 줄여 재시도합니다.

### 정밀도와 GPU 메모리

- **FP16**: 기본값이며 지원 장치에서 모델 가중치와 주요 추론 경로에 사용합니다.
- **BF16**: 지원 GPU에서 모델 가중치와 주요 추론 경로에 사용합니다.
- **FP32 + TF32** 또는 **엄격한 FP32**도 선택할 수 있습니다.
- 모델 로드와 VRAM/offload는 ComfyUI 모델 관리자의 정책을 따릅니다. 모델 전체를 항상 VRAM에 고정하는 방식은 아닙니다.

보간 결과는 ComfyUI IMAGE 배치입니다. 자세한 설명은 [한국어 README](README.ko.md)를 참고하세요.
