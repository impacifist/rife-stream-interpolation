# ComfyUI RIFE 프레임 보간

ComfyUI에서 사용하는 노드는 두 가지입니다.

## 노드 연결

1. **Load RIFE v4.26+ (FP16 / BF16 / FP32)**에서 `models/frame_interpolation`의 모델과 연산 정밀도(FP16/BF16/FP32)를 선택합니다. FP32는 TF32 사용 여부를 함께 선택합니다.
2. 이 노드의 **`interp_model` 출력**을 **RIFE Frame Interpolation의 `interp_model` 입력**에 연결합니다.
3. 비디오 로더의 IMAGE 프레임 배치를 **RIFE Frame Interpolation의 `images` 입력**에 연결합니다.
4. `multiplier` 위젯에서 배수(2–16)를 설정합니다. `chunk_size` 위젯은 한 번에 추론할 인접 프레임 쌍 수입니다. 작은 값은 VRAM을 덜 쓰고, 큰 값은 여유 VRAM이 있을 때 처리량을 높일 수 있습니다. 보간 결과 IMAGE를 다음 노드에 연결합니다.

```text
Load RIFE Model: interp_model ──→ RIFE Frame Interpolation: interp_model
Video Loader: IMAGE frames ─────→ RIFE Frame Interpolation: images
                                     multiplier: 배수 설정
                                     chunk_size: 프레임 쌍 chunk 크기
RIFE Frame Interpolation: images → 다음 IMAGE 노드
```

## 설치

이 폴더를 `ComfyUI/custom_nodes/rife_stream_interpolation`에 복사하고 ComfyUI를 재시작한 뒤 `RIFE`를 검색하세요. 로더 정밀도는 FP16, BF16, TF32 허용 FP32, 엄격한 FP32입니다.
