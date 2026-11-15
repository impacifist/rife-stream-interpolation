# RIFE Frame Interpolation for ComfyUI

ComfyUI 내장 프레임 보간과 같은 두 노드로 구성한 RIFE v4.26+ 커스텀 노드입니다.

## 노드와 연결

1. **Load RIFE v4.26+ (BF16 / TF32 / FP32)**에서 모델과 정밀도를 선택합니다. RIFE v4.26+ IFNet 체크포인트는 `models/frame_interpolation`에 둡니다.
2. **RIFE Frame Interpolation**에 로더의 `interp_model` 출력과 비디오 로더의 `images` 프레임 배치를 연결합니다.
3. `multiplier` 위젯에서 보간 배수(2–16)를 설정하고, `images` 출력을 후속 IMAGE/비디오 노드에 연결합니다.

```text
RIFE Model Loader: interp_model ──→ RIFE Frame Interpolation: interp_model
Video Loader: IMAGE frames ───────→ RIFE Frame Interpolation: images
                                     multiplier: 배수 설정
RIFE Frame Interpolation: images ─→ 다음 IMAGE/비디오 노드
```

소스 FPS와 타겟 FPS 계산은 별도 수학 노드에서 수행하고, 계산한 배수를 보간 노드의 `multiplier` 위젯에 설정합니다. 보간 노드는 내장 `FrameInterpolate`와 같은 모델, IMAGE 배치, 배수 입력으로 동작합니다.

## 설치

이 폴더를 `ComfyUI/custom_nodes/rife_stream_interpolation`에 복사하고 ComfyUI를 재시작합니다. `RIFE`를 검색하면 모델 로더와 프레임 보간 노드가 표시됩니다. 로더는 BF16, TF32 허용 FP32, 엄격한 FP32를 지원합니다.
