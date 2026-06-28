import torch.nn as nn
import torch 

# ---------------------------------------------------------------------
# 2. Small MLP: (x,t) -> phi
# ---------------------------------------------------------------------

class SimpleAC1D(nn.Module):
    """
    Minimal MLP for the blind 1D benchmark.

    Input:
        x, t with shape [N, 1]

    Output:
        phi_theta(x,t) with shape [N, 1]

    The inputs are internally mapped to [-1,1] for easier optimization.
    """

    def __init__(self, hidden_layers: int, hidden_dim: int):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):

        inp = torch.cat([x, t], dim=1)
        return self.net(inp)