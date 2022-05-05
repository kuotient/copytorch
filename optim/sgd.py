from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super(SGD, self).__init__(parameters)
        self.lr = lr
        
    def update_parameter(self, parameter):
        parameter.data -= self.lr * parameter.grad.data