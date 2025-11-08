import torch
from torch import nn

class LinearClassifierModel(nn.Module):
    def __init__(self, input_feat, output_feat):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(input_feat, output_feat)

    def forward(self, x):
        x_flattened = self.flatten(x)
        return self.linear_layer(x_flattened)

    