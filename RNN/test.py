# 필요한 라이브러리들 임포트 (numpy, matplotlib, pytorch, torchvision 등)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image

# GPU를 사용할 수 있으면 CUDA 장치를 사용, 그렇지 않으면 CPU를 사용
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

# 하이퍼파라미터 설정: 배치 사이즈와 에포크 수
BATCH_SIZE = 8
EPOCHS = 15

# 이미지에서 관심 영역(ROI)을 잘라내는 클래스를 정의
class ROICrop:
    def __init__(self, roi_box):
        self.roi_box = roi_box  # 관심 영역의 박스 좌표 (좌, 상, 우, 하) 설정

    def __call__(self, img):
        return img.crop(self.roi_box)  # 주어진 박스 좌표로 이미지를 잘라냄

# 데이터 변환 설정: 훈련용과 검증용 데이터에 대해 각각 변환을 정의
data_transforms = {
    'train': transforms.Compose([          # 훈련용 이미지 변환 파이프라인
        ROICrop((320, 0, 960, 360)),       # 관심 영역(ROI)만 잘라냄 (320, 0)에서 (960, 360)까지  # original image (1280 * 720)
        transforms.Resize((224, 224)),     # 이미지 크기를 224x224로 변경
        transforms.RandomHorizontalFlip(), # 50% 확률로 이미지를 좌우 반전
        transforms.ToTensor(),             # 이미지를 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
    ]),
    'val': transforms.Compose([            # 검증용 이미지 변환 파이프라인
        ROICrop((320, 0, 960, 360)),       # 훈련과 동일하게 ROI 영역을 자름
        transforms.Resize((224, 224)),     # 크기 변경
        transforms.ToTensor(),             # 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
    ])
}

# 데이터셋 정의: './data_final/train'과 './data_final/val' 경로에서 데이터 로드 및 변환 적용
image_datasets = {x: datasets.ImageFolder("./data_final/"+x,
                                          data_transforms[x]) for x in ['train', 'val']}
# DataLoader 정의: 각 데이터셋에 대해 배치 사이즈와 기타 설정 적용
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size = BATCH_SIZE,
                                              num_workers = 0,  # 워커 수 (0이면 메인 프로세스에서 처리)
                                              shuffle = True) for x in ['train','val']}

# 훈련 함수 정의
def train(model, train_loader, optimizer, log_interval):
    model.train()  # 모델을 훈련 모드로 설정
    model.to(DEVICE)  # 모델을 GPU로 이동
    for batch_idx, (image, label) in enumerate(train_loader):  # 배치 단위로 이미지와 레이블을 가져옴
        image = image.to(DEVICE)  # 이미지를 GPU로 이동
        label = label.to(DEVICE)  # 레이블을 GPU로 이동
        optimizer.zero_grad()  # 옵티마이저의 이전 그래디언트를 초기화
        output = model(image).cuda()  # 모델이 이미지를 입력으로 받아 출력을 계산
        loss = criterion(output, label)  # 손실 함수 계산 (CrossEntropy)
        loss.backward()  # 역전파로 그래디언트 계산
        optimizer.step()  # 옵티마이저를 사용해 가중치 업데이트

        # 로그 간격마다 훈련 상태 출력
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {Epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tTrain Loss: {loss.item():.6f}")

# 평가 함수 정의
def evaluate(model, test_loader):
    model.eval()  # 평가 모드로 설정 (배치 정규화 및 드롭아웃 비활성화)
    test_loss = 0  # 테스트 손실 초기화
    correct = 0  # 정확도 계산을 위한 초기화

    # 평가할 때는 그래디언트를 계산하지 않음
    with torch.no_grad():
        for image, label in test_loader:  # 검증 데이터셋을 배치 단위로 처리
            image = image.to(DEVICE)  # 이미지를 GPU로 이동
            label = label.to(DEVICE)  # 레이블을 GPU로 이동
            output = model(image)  # 모델 예측값 계산
            test_loss += criterion(output, label).item()  # 배치 손실 합산
            prediction = output.max(1, keepdim=True)[1]  # 출력값에서 가장 높은 값(클래스) 추출
            correct += prediction.eq(label.view_as(prediction)).sum().item()  # 정답과 비교하여 맞춘 개수 계산

    # 평균 테스트 손실 계산
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    # 정확도 계산
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy  # 테스트 손실과 정확도 반환

# 사전 학습된 알렉스넷 모델 불러오기
model = models.alexnet(pretrained = True).cuda()
# 출력 계층을 3개의 클래스에 맞게 수정
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 3)  # 3 클래스 출력 (교통 표지판 등)

# Adam 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
# 교차 엔트로피 손실 함수 설정
criterion = nn.CrossEntropyLoss()

# 훈련 및 평가 반복
for Epoch in range(1, EPOCHS + 1):
    train(model, dataloaders['train'], optimizer, log_interval=5)  # 훈련 실행
    test_loss, test_accuracy = evaluate(model, dataloaders['val'])  # 검증 실행
    print(f"[Epoch: {Epoch}] \tTest Loss: {test_loss:.4f} \tTest Accuracy: {test_accuracy:.2f}%")  # 결과 출력

# 학습된 모델 저장
torch.save(model.state_dict(), 'traffic_alexnet_adam_15.pt')
print("Saved model!")  # 모델 저장 완료 메시지 출력
