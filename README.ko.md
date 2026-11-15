# ComfyUI RIFE 프레임 보간

ComfyUI에서 사용하는 노드는 두 가지입니다.

## 노드 연결

1. **Load RIFE v4.26+ (BF16 / TF32 / FP32)**에서 `models/frame_interpolation`의 모델을 선택합니다.
2. 이 노드의 **`interp_model` 출력**을 **RIFE Frame Interpolation의 `interp_model` 입력**에 연결합니다.
3. 비디오 로더의 IMAGE 프레임 배치를 **RIFE Frame Interpolation의 `images` 입력**에 연결합니다.
4. `multiplier` 위젯에서 배수를 설정하고, 보간 노드의 `images` 출력을 다음 노드에 연결합니다.

```text
Load RIFE Model: interp_model ──→ RIFE Frame Interpolation: interp_model
Video Loader: IMAGE frames ─────→ RIFE Frame Interpolation: images
                                    multiplier: 배수 설정
RIFE Frame Interpolation: images → 다음 IMAGE 노드
```

## 설치

이 폴더를 `ComfyUI/custom_nodes/rife_stream_interpolation`에 복사하고 ComfyUI를 재시작한 뒤 `RIFE`를 검색하세요.
