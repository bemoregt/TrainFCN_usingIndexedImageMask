# TrainFCN_usingIndexedImageMask

FCN ResNet50 모델을 이용한 시맨틱 세그멘테이션(Semantic Segmentation) 학습 프로젝트입니다. 이 프로젝트는 PASCAL VOC 2012 데이터셋을 사용하여 Fully Convolutional Network (FCN)를 학습시키고 객체 세그멘테이션을 수행합니다.

## 프로젝트 개요

- **목적**: 이미지에서 픽셀 단위 시맨틱 세그멘테이션 수행
- **모델**: FCN ResNet50 (torchvision에서 제공)
- **데이터셋**: PASCAL VOC 2012
- **프레임워크**: PyTorch

## 주요 기능

1. **커스텀 VOC 데이터셋 클래스**: 인덱스드 마스크를 처리하기 위한 VOCSegmentationIndexed 클래스 구현
2. **모델 학습**: FCN ResNet50 모델 학습 및 학습 과정 시각화
3. **결과 시각화**: 원본 이미지, 실제 마스크, 예측 마스크를 비교하여 시각화
4. **다양한 디바이스 지원**: MPS(Mac), CUDA, CPU 디바이스 자동 감지 및 사용

## 환경 설정

필요한 패키지:
```
numpy
torch
torchvision
pillow
matplotlib
tqdm
```

## 사용법

1. 저장소 클론:
```bash
git clone https://github.com/bemoregt/TrainFCN_usingIndexedImageMask.git
cd TrainFCN_usingIndexedImageMask
```

2. 필요한 패키지 설치:
```bash
pip install numpy torch torchvision pillow matplotlib tqdm
```

3. 학습 실행:
```bash
python train_fcn.py
```

## 코드 구조 설명

### 1. VOCSegmentationIndexed 클래스
- VOC 데이터셋을 로드하고 인덱스드 마스크를 처리합니다.
- 이미지와 마스크를 동일한 크기(480x480)로 리사이즈합니다.
- 이미지는 정규화하고 마스크는 인덱스드 이미지 형태를 유지합니다.

### 2. 학습 함수
- 모델을 훈련시키고 진행 상황을 tqdm으로 표시합니다.
- CrossEntropyLoss를 사용하여 세그멘테이션 마스크를 학습합니다.

### 3. 시각화 함수
- 학습된 모델의 예측 결과를 시각화합니다.
- 원본 이미지, 실제 마스크, 예측 마스크를 나란히 표시합니다.

### 4. 메인 함수
- 데이터셋 로드 및 데이터 변환 설정
- 적절한 디바이스(MPS, CUDA, CPU) 선택
- 모델 초기화, 학습 및 저장
- 학습된 모델을 사용한 예측 시각화

## 결과

학습 후에는 다음과 같은 결과물이 생성됩니다:
1. 학습된 모델 가중치 파일: `fcn_resnet50_segmentation.pth`
2. 예측 시각화 이미지: `prediction_X_Y.png`

## 참고사항

- VOC 데이터셋은 처음 실행 시 자동으로 다운로드됩니다.
- MPS 디바이스(Apple Silicon)를 우선 사용하도록 설정되어 있으며, 없는 경우 CUDA 또는 CPU를 사용합니다.
- 배치 크기는 기본값이 4이며, 메모리 상황에 따라 조정 가능합니다.
- 학습 에폭 수는 기본값이 10이며, 필요에 따라 변경 가능합니다.
