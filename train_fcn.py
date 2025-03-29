import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF

# VOC 데이터셋을 직접 사용
class VOCSegmentationIndexed(datasets.VOCSegmentation):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        
        # 동일한 크기로 리사이즈 (480x480)
        img = TF.resize(img, (480, 480))
        mask = TF.resize(mask, (480, 480), interpolation=TF.InterpolationMode.NEAREST)
        
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = TF.to_tensor(img)
            img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # 마스크를 텐서로 변환하여 인덱스드 이미지 형태 유지
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask

# 학습 함수
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        
        # 진행 상황 표시를 위한 tqdm 사용
        loop = tqdm(dataloader)
        for inputs, masks in loop:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # 그라디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)['out']
            
            # 손실 계산
            loss = criterion(outputs, masks)
            
            # 역전파 및 최적화
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # tqdm 상태 업데이트
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}')
        
    return model

# 예측 및 시각화 함수
def visualize_prediction(model, dataloader, device, num_images=1):
    model.eval()
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(dataloader):
            if i >= num_images:
                break
            
            inputs = inputs.to(device)
            outputs = model(inputs)['out']
            
            # 클래스별 확률을 예측 클래스로 변환
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 이미지, 마스크, 예측 시각화
            for j in range(inputs.size(0)):
                plt.figure(figsize=(15, 5))
                
                # 원본 이미지
                img = inputs[j].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title('Original Image')
                
                # 실제 마스크
                plt.subplot(1, 3, 2)
                plt.imshow(masks[j].cpu().numpy())
                plt.title('Ground Truth')
                
                # 예측 마스크
                plt.subplot(1, 3, 3)
                plt.imshow(preds[j])
                plt.title('Prediction')
                
                plt.tight_layout()
                plt.savefig(f'prediction_{i}_{j}.png')
                plt.close()

if __name__ == '__main__':
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # VOC 데이터셋 로드 (torchvision의 내장 VOCSegmentation 사용)
    print("Loading VOC dataset...")
    dataset = VOCSegmentationIndexed(
        root='./data', 
        year='2012',
        image_set='train',
        download=True,
        transforms=transform
    )
    
    print(f"Dataset size: {len(dataset)}")

    # MPS 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device is available. Using MPS for training.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MPS device is not available. Using {device} for training.")

    # 데이터 로더 생성 (배치 크기 조정 - MPS에서 메모리 관리를 위해)
    batch_size = 4  # 필요에 따라 조정 가능
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )

    # FCN ResNet50 모델 로드
    num_classes = 21  # Pascal VOC의 경우 배경 포함 21 클래스
    model = fcn_resnet50(weights=None, num_classes=num_classes)
    model = model.to(device)
    
    # MPS 디바이스 최적화를 위한 설정
    model = model.float()
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255는 일반적으로 무시되는 값
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 모델 학습
    print("Starting training...")
    print(f"Training with batch size: {batch_size}")
    trained_model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)

    # 모델 저장
    torch.save(trained_model.state_dict(), 'fcn_resnet50_segmentation.pth')
    print("Model saved to fcn_resnet50_segmentation.pth")

    # 예측 시각화
    print("Visualizing predictions...")
    visualize_prediction(trained_model, dataloader, device, num_images=2)
    print("Predictions visualized and saved")
