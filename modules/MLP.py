import torch.nn as nn
import math


def Xavier(m):
    """Function for xavier initialization"""
    fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
    std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    m.weight.data.uniform_(-a, a)
    m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 200, bias=True)
        self.fc3 = nn.Linear(200, num_classes, bias=True)
        self.num_classes = num_classes

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                Xavier(m)