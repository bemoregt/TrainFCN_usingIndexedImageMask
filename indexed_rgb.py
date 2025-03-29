import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# VOC 클래스별 컬러맵 정의 (간소화된 버전)
voc_colormap = [
    [0, 0, 0],        # 배경 - 검정
    [128, 0, 0],      # aeroplane - 어두운 빨강
    [0, 128, 0],      # bicycle - 어두운 녹색
    [128, 128, 0],    # bird - 어두운 노랑
    [0, 0, 128],      # boat - 어두운 파랑
    [128, 0, 128],    # bottle - 어두운 마젠타
    [0, 128, 128],    # bus - 어두운 시안
    [128, 128, 128],  # car - 회색
    [64, 0, 0],       # cat - 매우 어두운 빨강
    [192, 0, 0],      # chair - 밝은 빨강
    [64, 128, 0],     # cow - 어두운 옐로우-그린
    [192, 128, 0],    # diningtable - 어두운 오렌지
    [64, 0, 128],     # dog - 어두운 퍼플
    [192, 0, 128],    # horse - 밝은 마젠타
    [64, 128, 128],   # motorbike - 어두운 청록
    [192, 128, 128],  # person - 분홍
    [0, 64, 0],       # pottedplant - 매우 어두운 녹색
    [128, 64, 0],     # sheep - 갈색
    [0, 192, 0],      # sofa - 밝은 녹색
    [128, 192, 0],    # train - 라임 그린
    [0, 64, 128]      # tvmonitor - 짙은 하늘색
]
voc_colormap = np.array(voc_colormap)

# 가로x세로 480x320 크기의 가상 인덱스드 이미지 생성
w, h = 480, 320
indexed_img = np.zeros((h, w), dtype=np.uint8)

# 몇 가지 세그먼트 예시 생성 (배경: 0)
# 인물(person): 15
indexed_img[100:250, 200:300] = 15
# 자동차(car): 7
indexed_img[180:280, 50:200] = 7
# 나무(pottedplant): 16
indexed_img[50:150, 350:450] = 16
# 소파(sofa): 18
indexed_img[230:300, 320:450] = 18

# 인덱스드 이미지를 RGB로 변환하는 함수
def convert_indexed_to_rgb(indexed_img, colormap):
    h, w = indexed_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(len(colormap)):
        rgb_img[indexed_img == i] = colormap[i]
    
    return rgb_img

# 인덱스드 이미지를 RGB로 변환
rgb_img = convert_indexed_to_rgb(indexed_img, voc_colormap)

# 원본 인덱스드 이미지와 컬러맵 이미지 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(indexed_img, cmap='tab20')
plt.title('Indexed Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb_img)
plt.title('RGB Visualization')
plt.axis('off')

plt.tight_layout()
plt.savefig('indexed_rgb.png', dpi=150, bbox_inches='tight')

# 범례 추가한 이미지 생성
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.imshow(rgb_img)
plt.title('Semantic Segmentation Visualization')
plt.axis('off')

# 범례를 위한 작은 컬러 박스와 레이블 추가
ax = plt.subplot(2, 1, 2)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

labels = ['Background', 'Person', 'Car', 'PottedPlant', 'Sofa']
colors = [voc_colormap[0], voc_colormap[15], voc_colormap[7], voc_colormap[16], voc_colormap[18]]

for i, (label, color) in enumerate(zip(labels, colors)):
    x = 0.1 + (i % 3) * 0.3
    y = 0.7 - (i // 3) * 0.4
    ax.add_patch(plt.Rectangle((x, y), 0.05, 0.05, color=color/255))
    ax.text(x + 0.07, y + 0.025, label, va='center')

plt.savefig('indexed_rgb_with_legend.png', dpi=150, bbox_inches='tight')
print("이미지 생성 완료: indexed_rgb.png, indexed_rgb_with_legend.png")
