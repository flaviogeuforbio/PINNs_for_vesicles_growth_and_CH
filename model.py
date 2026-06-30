"""
Neural network model for the source-free AC1D inverse benchmark.

The blind_ac1d_inverse branch uses a deliberately reduced PINN architecture:

    (x, t) -> phi_theta(x, t)

The chemical potential is not predicted as an independent network output.
Instead, it is computed from phi_theta by automatic differentiation inside the
loss:

    mu_theta = k * [ (1/eps) * (phi_theta^3 - phi_theta)
                    - eps * phi_theta_xx ]

This avoids introducing a freely learned auxiliary field mu_theta in the blind
inverse benchmark, where such a field could create a scale ambiguity with the
trainable mobility m_phi.
"""

from typing import Tuple

import torch
from torch import Tensor, nn


class SimpleAC1D(nn.Module):
    """
    Fully-connected tanh MLP for the 1D Allen-Cahn inverse problem.

    The network maps space-time coordinates to the physical field:

        (x, t) -> phi_theta(x, t)

    Parameters
    ----------
    hidden_layers:
        Total number of hidden layers.
    hidden_dim:
        Number of neurons in each hidden layer.

    Notes
    -----
    The input coordinates are used as provided. No internal normalization is
    applied in this model.
    """

    def __init__(self, hidden_layers: int, hidden_dim: int):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        layers = [
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate phi_theta at space-time coordinates.

        Parameters
        ----------
        x:
            Spatial coordinates, shape (N, 1).
        t:
            Time coordinates, shape (N, 1).

        Returns
        -------
        Tensor
            Predicted phase field phi_theta(x,t), shape (N, 1).
        """
        inputs = torch.cat([x, t], dim=1)
        phi = self.net(inputs)

        return phi