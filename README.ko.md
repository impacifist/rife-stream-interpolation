# ComfyUI RIFE 프레임 보간

FP16/BF16/FP32 정밀도 선택과 프레임 쌍 chunk 처리를 제공하는 RIFE v4.26+ 보간 패키지입니다. 모델 로더와 IMAGE 배치 보간 노드 두 가지로 구성됩니다.

## 노드 연결

1. **Load RIFE v4.26+ (FP16 / BF16 / FP32)**에서 `models/frame_interpolation`의 체크포인트와 정밀도를 선택합니다. FP32는 TF32 사용 여부도 고릅니다.
2. 이 노드의 **`interp_model` 출력**을 **RIFE Frame Interpolation의 `interp_model` 입력**에 연결합니다.
3. 비디오 로더의 IMAGE 프레임 배치를 **RIFE Frame Interpolation의 `images` 입력**에 연결합니다.
4. `multiplier`(2–16)와 `chunk_size`(1–16 인접 프레임 쌍)를 설정하고 결과 IMAGE 배치를 다음 노드에 연결합니다.

```text
Load RIFE Model: interp_model ──→ RIFE Frame Interpolation: interp_model
Video Loader: IMAGE frames ─────→ RIFE Frame Interpolation: images
                                     multiplier: 배수 설정
                                     chunk_size: 프레임 쌍 chunk 크기
RIFE Frame Interpolation: images → 다음 IMAGE 노드
```

## 정밀도 선택

- **FP16**: 기본 선택입니다. 지원 장치에서 16-bit 모델 가중치와 주요 추론 경로를 사용합니다.
- **BF16**: 16-bit 모델 가중치와 주요 추론 경로를 사용하며 GPU 지원 여부를 확인합니다.
- **FP32 + TF32**: 모델 텐서는 FP32로 두고, CUDA가 지원하는 cuDNN/cuBLAS 연산에서 TF32 사용을 허용합니다.
- **엄격한 FP32**: TF32를 끈 FP32 연산입니다.

모델은 ComfyUI가 선택한 장치에 배치되어 `CoreModelPatcher`로 전달됩니다. 추론 시 ComfyUI 모델 관리자가 현재 VRAM/offload 정책을 적용하므로 모델이 항상 VRAM에 전부 고정되는 것은 아닙니다. CUDA에서는 channels-last 텐서와 cuDNN autotune을 사용합니다.

선택한 정밀도는 모델 가중치와 주요 추론 경로에 적용됩니다. RIFE flow-warp/grid-sampling 보조 연산 일부는 sampling 동안 내부적으로 FP32를 사용한 뒤 모델 dtype으로 되돌립니다.

## GPU 메모리와 chunk

`chunk_size`는 동시에 추론하는 인접 프레임 쌍 수입니다(기본값 4, 범위 1–16). chunk가 작으면 peak 활성화 메모리가 줄어듭니다. 크게 하면 여유 VRAM에서 처리량을 높일 수 있습니다. VRAM OOM 발생 시 chunk를 절반으로 줄여 재시도합니다.

노드는 ComfyUI IMAGE 배치를 입력받아 IMAGE 배치를 출력합니다. CPU intermediate 출력은 결과 전체를 일반 RAM에 상주시켜 할당하지 않도록 임시 disk-backed 텐서를 사용하지만, 입력 배치와 후속 노드의 메모리 사용은 별도입니다.

## 설치

이 폴더를 `ComfyUI/custom_nodes/rife_stream_interpolation`에 복사하고 ComfyUI를 재시작한 뒤 `RIFE`를 검색하세요. 체크포인트는 `models/frame_interpolation`에 둡니다.
