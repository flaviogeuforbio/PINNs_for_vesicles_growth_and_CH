import torch
from torch import nn

#very simple FFNN that maps (x, t) -> (c, mu) solving the 1D Cahn-Hilliard equation
class CahnHilliardPINN(nn.Module):
    def __init__(self, hidden_layers: int = 2, hidden_dim: int = 64):
        super().__init__()

        layers = []

        #input layer: (N, 2) -> (N, hidden_dim)
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        #output layer: (N, hidden_dim) -> (N, 2). It outputs concentration of conserved quantity c and chemical potential mu
        layers.append(nn.Linear(hidden_dim, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        #packing the input
        inputs = torch.cat([x, t], dim = 1)
        
        #forward propagating the inputs and extracting c, mu output predictions
        outputs = self.net(inputs)
        c = outputs[:, :1] #[N, 1]
        mu = outputs[:, 1:] #[N, 1]

        return c, mu
    


