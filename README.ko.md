# RIFE Stream Frame Interpolation

RIFE v4.26+ IFNet 계열 모델로 비디오를 스트리밍 보간하는 ComfyUI 커스텀 노드입니다. 전체 프레임을 ComfyUI `IMAGE` 배치로 읽거나 결과로 쌓지 않고, FFmpeg로 프레임을 한 장씩 디코드하고 보간 결과를 MP4로 바로 인코딩합니다.

## 설치

1. `rife_stream_interpolation` 폴더를 ComfyUI의 `custom_nodes` 아래에 복사합니다.
2. ComfyUI의 `models/frame_interpolation` 폴더에 ComfyUI 호환 RIFE v4.26 이상 체크포인트를 둡니다.
3. 시스템 PATH에서 `ffmpeg`와 `ffprobe`를 실행할 수 있게 하고 ComfyUI를 재시작합니다.

체크포인트가 프레임 보간 모델 목록에 나타나지 않으면, ComfyUI의 frame interpolation 모델 경로 설정을 확인하세요.

## 노드 사용

1. **Load RIFE v4.26+ (BF16 / TF32 / FP32)** 노드에서 체크포인트와 정밀도를 선택합니다.
   - `bf16`: BF16 weights/activations. 선택한 장치가 ComfyUI에서 BF16을 지원해야 합니다.
   - `fp32_tf32`: FP32 모델이며 CUDA cuDNN/cuBLAS의 TF32 연산 허용. TF32 Tensor Core 경로를 사용해 FP32보다 빠를 수 있으며, 엄격한 IEEE FP32와 수치 특성이 다릅니다.
   - `fp32_strict`: TF32 비활성 FP32 연산입니다.
2. **RIFE Frame Interpolation (IMAGE In/Out)** 노드는 모델과 프레임 배치를 입력받고, multiplier 위젯으로 배수를 정해 IMAGE 배치를 출력합니다.
3. 입력 비디오 경로를 직접 스트리밍하려면 **RIFE Stream Frame Interpolation** 노드에 모델을 연결하고 `video_path`에 전체 경로를 넣습니다.
4. 스트리밍 노드의 `pair_batch_size` 기본값은 4입니다. 인접한 프레임 쌍을 GPU 배치 처리하며 VRAM OOM 시 자동으로 배치를 줄여 재시도합니다. 결과 MP4는 ComfyUI의 `output` 폴더에 저장됩니다.

IMAGE 호환 노드는 기존 IMAGE 배치 인터페이스를 유지합니다. CPU intermediate 출력은 disk-backed 텐서를 사용하고, 추론 시 작은 프레임 그룹만 장치로 옮깁니다. 입력 프레임은 이미 메모리에 있어야 하고 후속 노드가 출력을 메모리로 올릴 수 있으므로, 긴 고해상도 영상은 경로 기반 스트리밍 노드를 권장합니다.

배수를 별도 수학 노드에서 계산하려면 수학 노드의 **INT 출력**을 `multiplier_input` 소켓에 연결하세요. 연결되어 있으면 이 입력이 multiplier 위젯보다 우선하고, 연결하지 않으면 위젯 값을 사용합니다. 허용 배수 범위는 2–16입니다. 소스 FPS·타겟 FPS 계산은 보간 노드와 분리된 수학 노드에서 구성할 수 있습니다.

CUDA 경로에서는 channels-last 가중치/프레임 레이아웃과 cuDNN 고정 해상도 autotune을 사용합니다. `fp32_tf32`에서만 matmul/cuDNN TF32 허용을 켜며 노드 종료 후 matmul 설정을 복원합니다. BF16은 ComfyUI가 선택한 장치에서 지원 여부를 확인하며, 지원되지 않으면 실행을 중단합니다. FP32는 더 많은 VRAM을 사용할 수 있습니다. 모델 로드와 활성화 메모리는 ComfyUI의 모델 관리자를 사용하므로, ComfyUI의 GPU 선택·VRAM 정책을 따릅니다.

비디오 디코드 및 인코드는 FFmpeg subprocess를 사용합니다. 출력은 H.264 MP4, 18 CRF이며 입력 오디오는 AAC로 함께 기록합니다. 노드는 입력을 평균 프레임레이트 기반으로 처리하므로 가변 프레임레이트(VFR) 영상은 먼저 CFR로 변환하는 것을 권장합니다.

이 노드는 ComfyUI 코어의 `comfy_extras.frame_interpolation_models.ifnet`을 사용합니다. 체크포인트의 키/텐서 형식이 코어 IFNet 구현과 호환되어야 합니다. 모델 체크포인트 자체에 신뢰할 수 있는 버전 메타데이터가 없는 경우 버전 문자열이 아닌 구조 호환 여부로 판정합니다.

## 메모리 동작

- 경로 기반 스트리밍 노드는 입력 비디오 전체를 RAM에 적재하지 않고 제한된 프레임 그룹과 feature만 유지합니다.
- 스트리밍 노드는 IMAGE 출력 배치를 만들지 않고 프레임별로 MP4 인코더에 전달합니다.
- IMAGE 호환 노드는 기존 소켓에 맞춰 IMAGE 배치를 반환하며 CPU intermediate 출력에는 disk-backed 임시 텐서를 사용합니다.
- FFmpeg 디코더 1개와 인코더 1개가 파이프로 동시에 연결됩니다. 시스템 RAM의 실제 사용량은 FFmpeg, PyTorch 및 ComfyUI의 다른 노드 사용량에 따라 달라질 수 있습니다.
- 이 노드는 실행하는 ComfyUI 프로세스의 로컬 입력 경로만 읽습니다. 생성된 MP4는 해당 ComfyUI 인스턴스의 `output` 디렉터리에 기록됩니다.
