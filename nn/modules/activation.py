from .module import Module
from copytorch import Variable

import copytorch.functions as F

class Sigmoid(Module):
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def forward(self, input: Variable) -> Variable:
        return F.sigmoid(input)