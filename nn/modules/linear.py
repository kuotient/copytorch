import copytorch
import numpy as np

from ..parameter import Parameter
import copytorch.functions as F
from .module import Module
from copytorch import Variable
from typing import Optional

class Linear(Module):
    def __init__(self, out_features:int, bias=True, dtype=np.float32, in_features:Optional[int] = None) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.weight = Parameter(None, name='weight')
        if self.in_features is not None:
            self._init_weight()

        if bias:
            self.bias = Parameter(np.zeros(out_features).astype(dtype), name='bias')
        else:
            self.bias = None

    def _init_weight(self) -> None:
        weight_data = np.random.randn(self.in_features, self.out_features).astype(self.dtype) * np.sqrt(1.0 / self.in_features)
        self.weight.data = weight_data
        
    def forward(self, input: Variable) -> Variable:
        if self.weight.data is None:
            self.in_features = input.shape[1]
            self._init_weight()
        
        return F.linear(input, self.weight, self.bias)