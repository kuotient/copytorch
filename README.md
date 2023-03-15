# CoPyTorch: A PyTorch Reimplementation in Pure Python

 

  
CoPyTorch is a deep learning framework based on automatic differentiation with similar functionality and syntax to PyTorch. It is designed to be easy to understand with no parts that work in C, bridging the gap between functions and mathematical understanding by implementing only the core features of PyTorch in pure Python.

## 🎓 Why Reimplement PyTorch in Pure Python?
Reimplementing the core components of a deep learning framework like PyTorch can be extremely helpful in understanding the underlying principles of deep learning. CoPyTorch was created to provide a deeper understanding of how PyTorch and other similar frameworks work. A few reasons why this exercise can be beneficial are:

1. **Reinforce theoretical knowledge**: Building the framework from scratch allows you to revisit and reinforce your understanding of deep learning concepts such as backpropagation, gradient descent, and the computational graph.

2. **Deepen understanding of PyTorch**: By implementing the core functionalities, you gain insights into how PyTorch operates behind the scenes, which can help you make better use of the framework and optimize your models.

3. **Improve coding skills**: Developing a deep learning framework requires a solid understanding of Python, data structures, and algorithms. This project serves as an excellent opportunity to hone your programming skills.

4. **Customize and extend**: By building your own version of PyTorch, you can easily modify, extend, or optimize the framework for your specific needs.

## 🚀 Features
- 🔥 **Autograd**: Reverse-mode auto-differentiation based on the "Define by Run" principle.
- 📐 **Broadcasting**: Supports NumPy's broadcasting, Python's operators, and basic functions.
- 🧩 **Same API as PyTorch**: Use CoPyTorch functions and modules in the same way as in PyTorch.
- 🐍 **Pythonic data handling**: Easy integration with NumPy, a popular Python package.  


## 📚 Getting Started
To start using CoPyTorch, clone the repository and install the required dependencies.
```bash
git clone https://github.com/kuotient/copytorch.git
cd copytorch
pip install -r requirements.txt
```
## 💡 Example
Here's an example of how to use CoPyTorch to implement a basic 2D linear neural network. 
- Module
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
- Data Handling
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
