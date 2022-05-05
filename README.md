# Copytorch

 

  

Copytorch는 자동 미분을 기반으로 한 Pytorch의 기능과 문법이 유사한 딥러닝 프레임워크입니다.

  

### 왜 만들었는가?

  

Pytorch와 여러 기타 딥러닝 라이브러리는 고수준 API를 지원하여 쉽게 모델을 제작하고 실행할 수 있는 환경을 제공합니다. 하지만 기본적인 기계학습에 대한 이해와 모델이 어떻게 동작하는지에 대한 지식이 있다고 하더라도, 실제로 프레임워크 상에서 어떤 방법으로 작동이 되는지 알기란 굉장히 어렵습니다. Copytorch는 C 로 작동하는 부분이 없어 이해가 쉬우며, 또한 Pytorch의 가장 핵심적인 기능만 구현하여 기능과 수학적 이해의 간극을 메꾸고자 만들어졌습니다.

  

### 어떻게 작동하나요?

  

Copytorch는 Variable 과 Function 클래스를 기반으로, 함수나 신경망을 Pytorch 문법과 유사하게 작성할 수 있습니다. 기본적인 2차원 선형 신경망에 대한 학습은 아래와 같이 작성할 수 있습니다.

  

```python
import numpy as np
import copytorch.functions as F

from copytorch import nn, optim

class TwoLayerNet(nn.Module):
  def __init__(self,hidden_size, num_classes):
    super(TwoLayerNet, self).__init__()
    self.linear1 = nn.Linear(hidden_size)
    self.linear2 = nn.Linear(num_classes)

  def forward(self,x):
    x = F.sigmoid(self.linear(x))
    x = self.linear2(x)
    return x
    
model = TwoLayerNet(hidden_size, 1)
optimizer = optim.SGD(model.parameters(), lr=0.2)

for i in range(10000):
  y_pred = model(x)
  
  loss = F.mean_squared_error(y,y_pred)
  loss.backward()
  optimizer.step()
  
  if i%1000 == 0:
    print(loss)

```

  

## Features

  



  

### Autograd(Reverse-mode auto-differentiation)

  

Copytorch는 Pytorch, Chainer과 같은 프레임워크가 작동하는 원리인 ‘Define by Run’을 기반으로 작성되었습니다. 그에 따라 데이터와 함수를 파이썬 문법으로 작성할 수 있으며, 디버깅도 파이썬의 디버거를 사용하여 분석이 가능합니다.

  

### Pythonic data handling

  

Python의 유명 패키지인 Numpy를 지원하며, Numpy의 broadcasting, python의 연산자와 기본 함수도 대응합니다.

  

```python
import numpy as np
from copytorch import Variable

def  matyas(x,y):
  z = 0.26 * (x**2 + y**2) - 0.48 * x * y
  return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)

z.backward()

# x,y의 미분 값
print(x.grad, y.grad)
```

  

## Documentation(Work in process)

  

