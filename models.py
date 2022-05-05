from pkg_resources import add_activation_listener
import copytorch.functions as F
import copytorch.nn as nn

class MLP(nn.Module):
    def __init__(self, output_sizes, activation=F.sigmoid) -> None:
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(output_sizes):
            layer = nn.Linear(out_size)
            setattr(self, f'layer_{i}', layer)
            self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers[:-1]:
            input = self.activation(layer(input))
        return self.layers[-1](input)
