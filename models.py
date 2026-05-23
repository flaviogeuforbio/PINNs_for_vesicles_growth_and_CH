import torch
from torch import nn

class CoupledACCHPINN2d(nn.Module):
    def __init__(self, hidden_layers: int = 4, hidden_dim: int = 128):
        super().__init__()

        layers = []

        #input layer: (N, 3) -> (N, hidden_dim)
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        #output layer: (N, hidden_dim) -> (N, 3). It outputs non-conserved phield phi, concentration of conserved quantity c, chemical potential mu
        layers.append(nn.Linear(hidden_dim, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        #packing the input
        inputs = torch.cat([x, y, t], dim = 1)
        
        #forward propagating the inputs and extracting c, mu output predictions
        outputs = self.net(inputs)
        phi = outputs[:, 0:1] #[N, 1]
        c = outputs[:, 1:2] #[N, 1]
        mu = outputs[:, 2:3] #[N, 1]

        return phi, c, mu
    

class BioACCHPINN2d(nn.Module):
    def __init__(self, hidden_layers: int = 4, hidden_dim: int = 128):
        super().__init__()

        layers = []

        #input layer: (N, 3) -> (N, hidden_dim)
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        #output layer: (N, hidden_dim) -> (N, 4). It outputs non-conserved phield phi, mu_phi (chem. pot.), concentration of conserved quantity c, mu_c
        layers.append(nn.Linear(hidden_dim, 4))

        self.net = nn.Sequential(*layers)

    def forward(self, x, y, t):
        #packing the input
        inputs = torch.cat([x, y, t], dim = 1)
        
        #forward propagating the inputs and extracting phi, mu, psi, nu output predictions
        outputs = self.net(inputs)
        phi = outputs[:, 0:1] #[N, 1]
        mu = outputs[:, 1:2] #[N, 1]

        raw_psi = outputs[:, 2:3] #[N, 1]
        psi = torch.sigmoid(raw_psi) #to make sure the concentration psi has values in [0, 1] (for simplicity)

        nu = outputs[:, 3:4] #[N, 1]

        return phi, mu, psi, nu