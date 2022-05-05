import numpy as np
import numpy.typing as npt
import heapq
import weakref
import contextlib
import copytorch


from dataclasses import dataclass, field
# typing is not supporting 3rd party structures, should be replaced with numpy.typing
from typing import Tuple, Union, Iterable, Optional


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


@dataclass
class Variable:
    __array_priority__ = 200

    data: np.ndarray
    name: Optional[str] = None
    # Pytorch may not use grad as variable, but use as method. but How?
    grad: np.ndarray = field(init=False, default=None)
    grad_fn : 'Function' = field(init=False, default=None)
    # TODO: Do not compute gradients when this option is false.
    requires_grad : bool = False
    generation: int = 0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        
        if self.grad_fn is not None:
            f = self.grad_fn.__class__.__name__
            return f'variable({p}, grad_fn={f})'
        else:
            return f'variable({p})'

    def set_grad_fn(self, func):
        self.grad_fn = func
        self.generation = func.generation + 1
    
    def zero_grad(self, set_to_none = False):
        self.grad = None if set_to_none else np.zeros_like(self.data)

    def backward(self, *, retain_grad = False, create_graph = False) -> None:
        # TODO: If no_grad() is True but this method is called, raise error.
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        
        funcs = []
        visited = set()

        def add_func(f):
            if f not in visited:
                visited.add(f)
                heapq.heappush(funcs, (-f.generation, len(visited), f))

        add_func(self.grad_fn)

        while funcs:
            func = heapq.heappop(funcs)[2]
            gys = [output().grad for output in func.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = func.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(func.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.grad_fn is not None:
                        add_func(x.grad_fn)
            
            if not retain_grad:
                for y in func.outputs:
                    y().grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        return copytorch.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], Iterable) or axes[0] is None:
                axes = axes[0]
        return copytorch.functions.transpose(self, axes)

    @property
    def T(self):
        return copytorch.functions.transpose(self)

    def sum(self, *, dim = None, keepdim = False):
        return copytorch.functions.sum(self, dim, keepdim)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x  


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj) 


class Function:
    def __call__(self, *inputs: Union[Variable, Tuple[Variable, ...]]):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_grad_fn(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = copytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = copytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = copytorch.functions.sum_to(gx0, x0.shape)
            gx1 = copytorch.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = copytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = copytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = copytorch.functions.sum_to(gx0, x0.shape)
            gx1 = copytorch.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = copytorch.functions.get_item

    # Variable.matmaul = copytorch.functions.matmul
    # Variable.dot = copytorch.functions.matmul
    # Variable.max = copytorch.functions.max
    # Variable.min = copytorch.functions.min

