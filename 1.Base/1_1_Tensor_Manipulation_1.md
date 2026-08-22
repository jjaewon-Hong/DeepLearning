## 데이터의 차원 별 이름



---



- 0차원 : 스칼라 (Scalar)

- 1차원 : 벡터 (Vector)

- 2차원 : 행렬 (Matrix)

- 3차원 이상 : 텐서 (Tensor)



⇒ 구글의 TensorFlow나 메타의 PyTorch 같은 딥러닝 프로그램 틀 안에서는 이 모든것을 Tensor라는 하나의 자료로 통칭함.



## Tensor



---



- 1D Tensor

    - 원래 데이터 한 개의 모양이 1D인것 같아도 딥러닝 모델에 들어가는 순간,

    - ‘배치’ 차원이 맨 앞에 무조건 추가되기 때문에 최소 2D Tensor 부터 시작함.



- 2D Tensor

    - 배치 크기와 데이터 특징 수(batch size, dim)의 형태를 가지므로

    - 일반적인 표 데이터를 다룰 때 무조건 2D 데이터 사용함.



- 3D Tensor

    - 표준 형태 (NLP / 시계열): (batch size, length, dim)

        - 자연어 처리(NLP) : 배치 크기, 문장 길아/토큰 수, 단어 임베딩 차원

            - 예: (1, 3, 256) → 배치 1개, 단어 3개, 단어당 256차원 Vector

        - 시계열 데이터(Time Series) : 배치 크기, 시간/타임스텝, 측정 변수 개수

            - 예: (batch_size, 11, 3) → 11초간, 3개 좌표(X, Y, Z) 측정값을 GRU/LSTM 입력으로 사용

    - 참고 (CV 단일 이미지):

        - 컬러 이미지 1장 단독 : (Channel, Height, Width)

        - 흑백 이미지 여러 장 : (Batch size, Height, Width)



- 4D Tensor

    - 표준 형태 : (Batch size, Channel, Height, Width) 컬러 이미지 여러 장

        - 컴퓨터 비전(CV) : 2D 이미지 데이터의 표준 CNN 모델 입력 형태

    - 비디오 데이터는 프레임/시간 차원이 추가되어 5D 텐서 (B, C, T, H, W)로 다루는 것이 일반적임



## Import



---



```jsx

import numpy as np

import torch

```



## NumPy & PyTorch



- GPU 가속(CUDA) : NumPy는 CPU에서만 작동하지만, PyTorch는 GPU(CUDA) 연산을 지원하여 대규모 행렬 연산을 훨씬 빠르게 처리

- 자동 미분 : PyTorch는 복잡한 신경망의 역전파 미분값을 cost.backward() 한줄로 계산하므로, 일일이 수학 공식 적는 NumPy보다 훨씬 간단함



## NumPy



---



1차원 배열 



---



```python

t = np.array([0., 1., 2., 3., 4., 5., 6.])

print(t)

```



> `[ 0.  1.  2.  3.  4.  5.  6.]`

> 



```jsx

print ('Rank of t : ', t.ndim) # 차원 

print(Shape of t : ', t.shape) # 요소 개수

```



> 

> 

> 

> Rank  of t:  1

> Shape of t:  (7,)

> 



```python

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) # Element

print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) # Slicing

print('t[:2] t[3:] = ', t[:2], t[3:]) # Slicing

```



> 

> 

> 

> t[0] t[1] t[-1] =  0.0 1.0 6.0

> t[2:5] t[4:-1]  =  [ 2.  3.  4.] [ 4.  5.]

> t[:2] t[3:]     =  [ 0.  1.] [ 3.  4.  5.  6.]

> 



---



2차원 배열 



---



```python

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])

print(t)

```



---



> 

> 

> 

> [[  1.   2.   3.]

>  [  4.   5.   6.]

>  [  7.   8.   9.]

>  [ 10.  11.  12.]]

> 



```python

print('Rank  of t: ', t.ndim)

print('Shape of t: ', t.shape)

```



> 

> 

> 

> Rank  of t:  2

> Shape of t:  (4, 3)

> 



## PyTorch Tensor



---



1차원 배열 



---



```python

t = torch.tensor([0.,1.,2.,3.,4.,5.,6.])

print(t)

```



> `tensor([ 0.  1.  2.  3.  4.  5.  6.])`

> 



```jsx

print(t.dim())

print(t.shape)

print(t.size())

print(t[0], t[1], t[-1])

print(t[2:5], t[4:-1])

print(t[:2], t[3:])

```



> 

> 

> 

> torch.Size([7])

> torch.Size([7])

> tensor(0.) tensor(1.) tensor(6.)

> tensor([2., 3., 4.]) tensor([4., 5.])

> tensor([0., 1.]) tensor([3., 4., 5., 6.])

> 



---



2차원 배열 



---



```python

t = torch.tensor([[1.,2.,3.],

                      [4.,5.,6.],

                      [7.,8.,9.],

                      [10.,11.,12.]

                      ])

print(t)

```



---



> 

> 

> 

> tensor([[ 1.,  2.,  3.],

> [ 4.,  5.,  6.],

> [ 7.,  8.,  9.],

> [10., 11., 12.]])

> 



```python

print(t.dim())

print(t.size())

print(t[:,1])

print(t[:, 1].size())

print(t[:, :-1])

```



> 

> 

> 

> 2

> torch.Size([4, 3])

> tensor([ 2.,  5.,  8., 11.])

> torch.Size([4])

> tensor([[ 1.,  2.],

> [ 4.,  5.],

> [ 7.,  8.],

> [10., 11.]])

> 



## Broadcasting



서로 다른 텐서끼리 연산할 때,  **PyTorch/NumPy 내부의 ‘연산엔진’**이



모자란 부분을 자동으로 복사해서 크기를 맞춰주는 NumPy 및 PyTorch의 자동 확장 기능



⇒ 편리함이 목적이지만 사용자의 의도 벗어날 수 있음



```python

m1 = torch.tensor([[3,3]])

m2 = torch.tensor([[2,2]])

print(m1 + m2)

```



> tensor([[5., 5.]])

> 



```python

m1 = torch.FloatTensor([[1,2]])

m2 = torch.FloatTensor([3])

print(m1 + m2)

```



> tensor([[4., 5.]])

> 



```python

m1 = torch.FloatTensor([[1,2]])

m2 = torch.FloatTensor([[3],[4]])

print(m1 + m2)

```



> 

> 

> 

> tensor([[4., 5.], [5., 6.]])

> 



## mul vs. matmul



multiplication는 원소 별 곱, matrix multiplication는 수학적 행렬 곱



```python

#mul

m1 = torch.FloatTensor([[[1,2],[3,4]]])

m2 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 1 : ', m1.shape)

print('Shape of Matrix 2 : ', m2.shape)

print(m1 * m2)

print(m1.mul(m2))



#matmul

m1 = torch.FloatTensor([[[1,2],[3,4]]])

m2 = torch.FloatTensor([[1], [2]])

print('Shape of Matrix 1 : ', m1.shape)

print('Shape of Matrix 2 : ', m2.shape)

print(m1.matmul(m2))

```



> 

> 

> 

> Shape of Matrix 1 :  torch.Size([1, 2, 2])

> Shape of Matrix 2 :  torch.Size([2, 1])

> tensor([[[1., 2.], [6., 8.]]])

> tensor([[[1., 2.], [6., 8.]]])

> 

> Shape of Matrix 1 : torch.Size([1, 2, 2])

> Shape of Matrix 2 : torch.Size([2, 1])

> tensor([[[ 5.], [11.]]])

> 



## mean



수치적으로는 ‘평균 값’을 의미하지만 구조적으로는 ‘사라짐(축소)’을 의미함



```python

t = torch.FloatTensor([1,2])

print(t.mean())

```



> tensor(1.5000)

> 



```python

t = torch.FloatTensor([1,2])

try :

    print(t.mean())

except Exception as exc :

    print(exc)

```



> mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long

> 



⇒  mean은 무조건 float 타입에서만 사용 가능



```python

t = torch.FloatTensor([[1,2],[3,4]])



print(t.mean())

print(t.mean(dim=0))

print(t.mean(dim=1))

print(t.mean(dim=-1))

```



> 

> 

> 

> tensor(2.5000)

> tensor([2., 3.])

> tensor([1.5000, 3.5000])

> tensor([1.5000, 3.5000])

> 



## Sum



```python

t = torch.FloatTensor([[1,2],[3,4]])



print(t.sum())

print(t.sum(dim=0))

print(t.sum(dim=1))

print(t.sum(dim=-1))

```



> 

> 

> 

> tensor(10.)

> tensor([4., 6.])

> tensor([3., 7.])

> tensor([3., 7.])

> 



## Max and Argmax



```python

t = torch.FloatTensor([[1,2],[3,4]])



print(t.max())

```



> tensor(4.)

> 



위에는 dim을 활용하지 않아 그 방향에서 최댓값이 무엇 인지만 도출했다면



아래에는 dim을 활용하여 최댓값이 무엇인지 그게 몇 번째 위치에 있는지 세트로 묶어서 둘 다 알려줌



```python

print(t.max(dim = 0))

print('Max : ', t.max(dim = 0)[0])

print('Argmax : ', t.max(dim = 0)[1])

```



> 

> 

> 

> torch.return_types.max(

> values=tensor([3., 4.]),

> indices=tensor([1, 1]))

> 

> Max :  tensor([3., 4.])

> Argmax :  tensor([1, 1])

> 



```python

print(t.max(dim=1))

print(t.max(dim=-1))

```



> 

> 

> 

> torch.return_types.max(

> values=tensor([2., 4.]),

> indices=tensor([1, 1]))

> 

> torch.return_types.max(

> values=tensor([2., 4.]),

> indices=tensor([1, 1]))

>

