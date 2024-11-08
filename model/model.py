from torch import nn


def sequential_layers(io_units, hidden_units, is_linear = True):
    last_units = io_units if len(hidden_units) == 0 else hidden_units[0]
    layers = [nn.Linear(io_units, last_units)]
    if not is_linear:
        layers.append(nn.ReLU())
    for idx in range(len(hidden_units) - 1):
        layers.append(nn.Linear(last_units, hidden_units[idx+1]))
        last_units = hidden_units[idx+1]
        if not is_linear:
            layers.append(nn.ReLU())
    layers.append(nn.Linear(last_units, io_units))
    return layers

class MLP(nn.Module):

    def __init__(self, io_units, hidden_units, is_linear):
        super().__init__()
        self.layers = nn.Sequential(*sequential_layers(io_units, hidden_units, is_linear))

    def forward(self, x):
        x = self.layers(x)
        return x