import numpy as np

class Optimizer:
    def __init__(self, parameters) -> None:
        self.target = None
        self.hooks = []
        self.param_groups = [p for p in parameters]

    def step(self):
        params = [p for p in self.param_groups if p.grad is not None]

        for hook in self.hooks:
            hook(params)
        
        for parameter in params:
            self.update_parameter(parameter)
    
    def update_parameter(self, parameter):
        raise NotImplementedError()
    
    def add_hook(self, hook):
        self.hooks.append(hook)

    def zero_grad(self, set_to_none=False):
        for p in self.parameters:
            p.grad = None if set_to_none else np.zeros_like(p.data)