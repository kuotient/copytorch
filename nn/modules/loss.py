from .module import Module
from copytorch import Variable
from typing import Union, Iterable

import copytorch.functions as F

class MSELoss(Module):
    def __init__(self) -> None:
        super(MSELoss, self).__init__()
    
    def forward(self, input: Variable, target: Variable) -> Union[Variable, Iterable[Variable]]:
        return F.mean_squared_error(input, target)