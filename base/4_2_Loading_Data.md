## Minibatch Gradient Descent



- 등장 배경 : 엄청난 양의 데이터를 한번에 학습 시킬 수 없다는 현실적인 문제

    

    ( 너무 느림 & HW적으로 불가능)

    

- 제시 방안 : 전체 데이터를 균일하게 나눠 일부분만 학습을 하자

    

    ⇒ 업데이트를 좀 더 빠르게 할 수 있다

    

    ⇒ 전체 데이터를 쓰지 않아서 잘못된 방향으로 업데이트를 할 수도 있다

    



Minibatch라는건 개념적인 이름이고, 실제로 이러한 문법이 있는 것은 X



⇒ 단지, batch_size를 낮추면 그것이 Minibatch



## PyTorch Dataset



- __len__() : 이 데이터셋의 총 데이터 수

- __getitem__() : 어떠한 인덱스를 받았을 때, 그에 상응하는 입출력 데이터 반환

- batch_size=2 : 각 minibatch 크기, 통상적으로 2의 제곱수로 설정함

- shuffle=True : Epoch마다 데이터셋을 섞어서, 데이터가 학습되는 순서를 바꿈



```python

import torch 

from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):

    def __init__(self) : 

        self.x_data = torch.FloatTensor([[73,80,75],

                                        [93,88,93],

                                        [89,91,90],

                                        [96,98,100],

                                        [73,66,70]])

        self.y_data = torch.FloatTensor([[152],[185],[180],[196],[142]])



    def __len__(self):

        return len(self.x_data)



    def __getitem__(self, idx):

        return self.x_data[idx], self.y_data[idx]

        # __getitem__ 측에서 FloatTensor로 변환하는것보다 

        # __init__ 부분에 최초에 한번 FloatTensor 먹이는게 메모리 할당 최적화



dataset = CustomDataset()



dataloader = DataLoader(

	dataset,

	batch_size = 2,

	shuffle = True, # 권장 (학습순서 자체를 외우는 것을 방지)

)

```



## Full Code with Dataset & DataLoader



- enumerate(dataloader) : minibatch 인덱스와 데이터를 받음

- len(dataloader) : 한 epoch당 minitbatch 개수



```python

import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import Dataset

from torch.utils.data import DataLoader



class MultivariateLinearRegressionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(3, 1)



    def forward(self, x):

        return self.linear(x)



class CustomDataset(Dataset):

    # 생략 (위와 동일)



# 모델 초기화

model =  MultivariateLinearRegressionModel()



# optimizer 설정

optimizer = torch.optim.SGD(model.parameters(), lr=2e-5)



nb_epoch = 10

for epoch in range(nb_epoch + 1):

    for batch_idx, samples in enumerate(dataloader):

        x_train, y_train = samples



        # H(x)

        prediction = model(x_train)



        # cost 계산

        cost = F.mse_loss(prediction, y_train)



        # cost로 H(x) 개선

        optimizer.zero_grad()

        cost.backward()

        optimizer.step()



        print(f'Epoch {epoch:4d}/{nb_epoch}  Batch {batch_idx+1}/{len(dataloader)}  Cost: {cost.item():.6f}')

```



- 출력 결과

    

    Epoch    0/10  Batch 1/3  Cost: 34382.988281

    Epoch    0/10  Batch 2/3  Cost: 1206.082764

    Epoch    0/10  Batch 3/3  Cost: 41.161339

    Epoch    1/10  Batch 1/3  Cost: 6.372858

    Epoch    1/10  Batch 2/3  Cost: 7.454859

    Epoch    1/10  Batch 3/3  Cost: 32.257984

    Epoch    2/10  Batch 1/3  Cost: 2.800480

    Epoch    2/10  Batch 2/3  Cost: 20.818623

    Epoch    2/10  Batch 3/3  Cost: 1.702886

    Epoch    3/10  Batch 1/3  Cost: 17.257273

    Epoch    3/10  Batch 2/3  Cost: 3.413954

    Epoch    3/10  Batch 3/3  Cost: 6.484862

    Epoch    4/10  Batch 1/3  Cost: 7.712011

    Epoch    4/10  Batch 2/3  Cost: 8.991093

    Epoch    4/10  Batch 3/3  Cost: 8.288277

    Epoch    5/10  Batch 1/3  Cost: 0.316175

    Epoch    5/10  Batch 2/3  Cost: 8.423635

    Epoch    5/10  Batch 3/3  Cost: 16.434305

    Epoch    6/10  Batch 1/3  Cost: 11.203875

    Epoch    6/10  Batch 2/3  Cost: 7.998269

    Epoch    6/10  Batch 3/3  Cost: 12.343960

    Epoch    7/10  Batch 1/3  Cost: 12.255815

    Epoch    7/10  Batch 2/3  Cost: 2.992579

    Epoch    7/10  Batch 3/3  Cost: 29.773695

    Epoch    8/10  Batch 1/3  Cost: 19.572779

    Epoch    8/10  Batch 2/3  Cost: 9.068856

    Epoch    8/10  Batch 3/3  Cost: 13.333374

    Epoch    9/10  Batch 1/3  Cost: 10.531460

    Epoch    9/10  Batch 2/3  Cost: 7.545208

    Epoch    9/10  Batch 3/3  Cost: 31.408964

    Epoch   10/10  Batch 1/3  Cost: 16.860435

    Epoch   10/10  Batch 2/3  Cost: 15.436975

    Epoch   10/10  Batch 3/3  Cost: 5.294297

    



전체적인 흐름



- 모델 정의 및 데이터 준비

    - nn.Module 상속받은 model 생성

    - nn.Linear(3,1) 통해 입력 3개, 출력 1개 가지는 선형 회귀 모델 구조 정의 및 객체 생성

- 20 Epoch 동안 미니 배치(batch_size=2), 즉 2행의 텐서 단위로 순회 (3개의 열 포함)

    - 각 미니 배치마다 예측값 H(x) 계산

    - MSE Loss(cost)계산

    - 경사하강법(Optimizer)

    

    ⇒ 위의 3가지 과정 반복

