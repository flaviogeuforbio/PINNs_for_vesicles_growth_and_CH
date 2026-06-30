"""
Neural network models for the 2D phase-field PINN experiments.

This module defines the fully-connected PINN architectures used in the
bio_phasefield_2d branch.

Two models are provided:

1. CoupledACCHPINN2d
   Legacy three-output model for coupled Allen-Cahn / Cahn-Hilliard prototypes:
       (x, y, t) -> (phi, c, mu)

2. BioACCHPINN2d
   Four-output mixed formulation used for the minimal bio-inspired phase-field
   model:
       (x, y, t) -> (phi, mu, psi, nu)

The mixed formulation predicts chemical potentials as auxiliary fields in order
to reduce the order of spatial derivatives required by automatic differentiation.
"""

import torch
from torch import Tensor, nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: int = 4,
    hidden_dim: int = 128,
) -> nn.Sequential:
    """
    Build a fully-connected tanh MLP.

    Notes
    -----
    The implementation preserves the original architecture of this branch:
    one input projection layer is followed by `hidden_layers` additional hidden
    tanh blocks, and then by the final output layer.

    Parameters
    ----------
    input_dim:
        Number of input coordinates.
    output_dim:
        Number of predicted fields.
    hidden_layers:
        Number of additional hidden layers after the input projection.
    hidden_dim:
        Number of neurons in each hidden layer.

    Returns
    -------
    nn.Sequential
        Fully-connected neural network.
    """
    if hidden_layers < 0:
        raise ValueError("hidden_layers must be non-negative.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")

    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
    ]

    for _ in range(hidden_layers):
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            ]
        )

    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


class CoupledACCHPINN2d(nn.Module):
    """
    Legacy 2D PINN for coupled Allen-Cahn / Cahn-Hilliard prototypes.

    The network maps space-time coordinates to three fields:
        (x, y, t) -> (phi, c, mu)

    where:
    - phi is the non-conserved phase/order parameter;
    - c is the conserved concentration-like field;
    - mu is the chemical potential associated with c.

    This class is kept for compatibility with earlier coupled AC/CH experiments.
    The final bio-inspired model uses `BioACCHPINN2d`.
    """

    def __init__(self, hidden_layers: int = 4, hidden_dim: int = 128):
        super().__init__()

        self.net = build_mlp(
            input_dim=3,
            output_dim=3,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate the network at space-time coordinates.

        Parameters
        ----------
        x, y, t:
            Column tensors of shape (N, 1).

        Returns
        -------
        phi, c, mu:
            Predicted fields, each with shape (N, 1).
        """
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.net(inputs)

        phi = outputs[:, 0:1]
        c = outputs[:, 1:2]
        mu = outputs[:, 2:3]

        return phi, c, mu


class BioACCHPINN2d(nn.Module):
    """
    Four-output mixed PINN for the minimal bio-inspired phase-field model.

    The network maps space-time coordinates to:
        (x, y, t) -> (phi, mu, psi, nu)

    where:
    - phi is the non-conserved vesicle/interface phase field;
    - mu is the chemical potential associated with phi;
    - psi is the conserved concentration field;
    - nu is the chemical potential associated with psi.

    Predicting mu and nu as auxiliary outputs keeps the PDE residuals at lower
    differential order, which is important for stable automatic differentiation
    in the coupled 2D phase-field model.
    """

    def __init__(self, hidden_layers: int = 4, hidden_dim: int = 128):
        super().__init__()

        self.net = build_mlp(
            input_dim=3,
            output_dim=4,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Evaluate the network at space-time coordinates.

        Parameters
        ----------
        x, y, t:
            Column tensors of shape (N, 1).

        Returns
        -------
        phi, mu, psi, nu:
            Predicted fields, each with shape (N, 1).
        """
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.net(inputs)

        phi = outputs[:, 0:1]
        mu = outputs[:, 1:2]
        psi = outputs[:, 2:3]
        nu = outputs[:, 3:4]

        return phi, mu, psi, nu