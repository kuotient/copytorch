from ..parameter import Parameter
from typing import Union, Iterable
from copytorch import Variable
from copytorch import utils

import copytorch
import weakref

class Module:
    def __init__(self):
        self._parameters = set()

    def __call__(self, *inputs: Variable) -> Variable:
        outputs = self.forward(*inputs)
        # Might change tuple to Iterable
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs: Variable) -> Union[Variable, Iterable[Variable]]:
        raise NotImplementedError()

    def parameters(self):
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Module):
                yield from obj.parameters()
            else:
                yield obj

    def zero_grad(self,set_to_none=False):
        for p in self.parameters():
            p.zero_grad(set_to_none=set_to_none)

    def __setattr__(self, name: str, value) -> None:
        # params = self.__dict__.get('_parameters')
        if isinstance(value, (Parameter, Module)):
            self._parameters.add(name)
        super(Module, self).__setattr__(name, value)

    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file= to_file)

    