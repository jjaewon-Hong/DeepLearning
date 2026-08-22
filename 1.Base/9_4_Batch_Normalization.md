## 역전파 시 기울기 크기 문제 (학습 최적화 문제)



- Gradient Vanishing

    - 소멸 문제, 기울기가 0으로 감

- Gradient Exploding

    - 기울기가 무한대로 감



⇒ 둘 중, 하나라도 발생하면 딥러닝 학습이 어려워짐



## 순전파 및 학습 중 데이터 분포의 변화 문제 (과적합 문제)



- Internal Covariate Shift

- 입력 데이터의 분포(평균, 분산)가 널뛰듯 계속 변함

- 뒤쪽 레이어가 계속 바뀌는 분포에 적응하는 과정에서 학습이 극도로 느려짐



## 해결법



- 간접적 해결법 (기울기 문제 해결)

    - Change activation function (ReLU)

    - Careful Initialization (초기화를 잘하자)

    - Small learning rate (gradient exploding만의 해결법)



- 직접적 해결법 (기울기 및 분포 문제 모두 해결)

    

    ⇒ **Batch Normalization**

    

    - Dropout(랜덤 노이즈)과 같은 Regularization 효과 (통째 암기 방지)

    - 학습 속도 가속 효과 (경사하강법에서 훨씬 더 큰 학습률 설정을 가능하게 함)

    - ★ Internal Covariate Shift를 해결하기 위해서, 각 layer(mini-batch)마다 Normalization을 두어 불안정하거나 쏠린 분포가 나오지 않도록 함



## Train & eval model



- model.train()

    - Dropout = True 상태

    - 즉, train 모드에서는 내부적으로 dropout의 뉴런을 껐다 켰다 하는 동작을 자동으로 함

        

        (과적합 방지, 앙상블 효과)

        

        *앙상블 : 수 많은 서브 모델 조합들이 도출해낸 결과들 중 마지막에 다 합쳐서 최선을 결정을 내림

        

- model.eval()

    - Dropout = False 상태

    



## Code : mnist_batchnorm



- BatchNorm 적용 여부에 따른 성능 차이 비교

- linear와 nn_linear는 완전히 독립적인 레이어 객체라 필요에 따라 맞춰 쓰는 형태임

    - linear 1~3 , bn1~2, relu : 배치 정규화 모델용 부품

    - nn_linear 1~3 : 비교 모델용 부품



```python

import torch

import torch.nn as nn



# 첫 번째 완전 연결층: 784차원 입력을 받아서 32차원 출력으로 변환

# MNIST 같은 이미지 28x28=784 입력을 가정할 때 많이 쓰는 구조

linear1 = torch.nn.Linear(784, 32, bias=False)



# hidden_layer: 32차원 입력을 받아서 32차원 출력으로 변환

linear2 = torch.nn.Linear(32, 32, bias=True)



# output_layer : 32차원 입력을 받아서 10차원 출력으로 변환

# 최종 클래스 수가 10개일 때 마지막 출력층으로 사용함

linear3 = torch.nn.Linear(32, 10, bias=True)



# ReLU 활성화 함수: 선형 변환 결과에 비선형성을 추가함 (음수는 0, 양수는 그대로)

relu = torch.nn.ReLU()



# 배치 정규화 레이어 1: 32차원 특성을 정규화함

# 각 미니배치에서 평균 0, 분산 1로 맞춰서 학습 안정성을 높임

bn1 = torch.nn.BatchNorm1d(32)

bn2 = torch.nn.BatchNorm1d(32)



# 아래는 BatchNorm 없이 바로 입력을 출력으로 연결하는 단순 모델

# nn_linear은 784(32)차원 입력을 10차원 출력으로 바로 매핑하는 레이어

nn_linear1 = torch.nn.Linear(784, 10, bias=True)

nn_linear2 = torch.nn.Linear(32, 10, bias=True)

nn_linear3 = torch.nn.Linear(32, 10, bias=True)

```



## Full Code



```python

import torch

import torch.nn as nn

import torchvision.datasets as dsets

import torchvision.transforms as transforms

from torch.utils.data import DataLoader



# [1] 디바이스 및 하이퍼파라미터 정의

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_epochs = 5

batch_size = 32

learning_rate = 0.01



# [2] 데이터셋 및 데이터로더 로드 (학습용 / 테스트용)

mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)



train_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)



# [3] 모델 구조 정의 (동일한 3개 층 조건으로 공정 비교)



# (1) BatchNorm 모델: BN 직전 Linear는 bias=False 적용

bn_model = nn.Sequential(

    nn.Linear(784, 32, bias=False), # BN이 편향을 상쇄하므로 bias=False

    nn.BatchNorm1d(32),

    nn.ReLU(),

    nn.Linear(32, 32, bias=False),  # BN이 편향을 상쇄하므로 bias=False

    nn.BatchNorm1d(32),

    nn.ReLU(),

    nn.Linear(32, 10, bias=True)    # 마지막 출력층은 BN이 없으므로 bias=True

).to(device)



# (2) 일반 모델: 동일한 3개 층 깊이 (BN만 제거)

nn_model = nn.Sequential(

    nn.Linear(784, 32, bias=True),

    nn.ReLU(),

    nn.Linear(32, 32, bias=True),

    nn.ReLU(),

    nn.Linear(32, 10, bias=True)

).to(device)



# [4] 손실함수 및 옵티마이저 정의

criterion = nn.CrossEntropyLoss().to(device)

bn_optimizer = torch.optim.Adam(bn_model.parameters(), lr=learning_rate)

nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)



# [5] 학습 루프 

print("=== 학습 시작 ===")

for epoch in range(training_epochs):

    bn_model.train()

    nn_model.train()



    bn_total_loss = 0.0

    nn_total_loss = 0.0



    for X, Y in train_loader:

        X = X.view(-1, 784).to(device)

        Y = Y.to(device)



        # 1) BatchNorm 모델 학습

        bn_optimizer.zero_grad()

        bn_prediction = bn_model(X)

        bn_loss = criterion(bn_prediction, Y)

        bn_loss.backward()

        bn_optimizer.step()

        bn_total_loss += bn_loss.item()



        # 2) 일반 모델 학습

        nn_optimizer.zero_grad()

        nn_prediction = nn_model(X)

        nn_loss = criterion(nn_prediction, Y)

        nn_loss.backward()

        nn_optimizer.step()

        nn_total_loss += nn_loss.item()



    avg_bn_loss = bn_total_loss / len(train_loader)

    avg_nn_loss = nn_total_loss / len(train_loader)

    

    print(f"Epoch {epoch+1:02d}/{training_epochs:02d} | BN Avg Loss: {avg_bn_loss:.4f} | Vanilla Avg Loss: {avg_nn_loss:.4f}")



# [6] 최종 테스트 검증 루프 

bn_model.eval()

nn_model.eval()



bn_correct = 0

nn_correct = 0

total = 0



with torch.no_grad():

    for X, Y in test_loader:

        X = X.view(-1, 784).to(device)

        Y = Y.to(device)



        # BN 모델 예측

        bn_pred = bn_model(X).argmax(dim=1)

        bn_correct += (bn_pred == Y).sum().item()



        # 일반 모델 예측

        nn_pred = nn_model(X).argmax(dim=1)

        nn_correct += (nn_pred == Y).sum().item()



        total += Y.size(0)



print("\n=== 최종 테스트 검증 결과 ===")

print(f"BatchNorm 모델 정확도 : {bn_correct / total * 100:.2f}%")

print(f"일반 모델 정확도: {nn_correct / total * 100:.2f}%")

```



> 

> 

> 

> === 학습 시작 ===

> Epoch 01/05 | BN Avg Loss: 0.2848 | Vanilla Avg Loss: 0.3007

> Epoch 02/05 | BN Avg Loss: 0.1774 | Vanilla Avg Loss: 0.2098

> Epoch 03/05 | BN Avg Loss: 0.1482 | Vanilla Avg Loss: 0.1894

> Epoch 04/05 | BN Avg Loss: 0.1331 | Vanilla Avg Loss: 0.1754

> Epoch 05/05 | BN Avg Loss: 0.1227 | Vanilla Avg Loss: 0.1670

> 

> === 최종 테스트 검증 결과 ===

> BatchNorm 모델 정확도 : 96.86%

> 일반 모델 정확도: 95.02%

> 



- 파라미터나 다른 요소는 모두 동일한 조건 기준

    - bn 사용한 모델이 bn 사용하지 않은 모델보다 loss 적고 정확도 높음



- BatchNorm 배치 순서

    - Linear → BatchNorm → ReLU 순서 기억하기

