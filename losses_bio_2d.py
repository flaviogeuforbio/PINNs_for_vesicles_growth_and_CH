"""
Loss functions for the 2D bio-inspired phase-field PINN.

This module contains the loss terms used in the bio_phasefield_2d branch:

1. PDE residual loss for the mixed four-field formulation:
       (phi, mu, psi, nu)

2. Initial-condition loss.

3. Homogeneous Neumann boundary-condition loss.

4. Manufactured-data loss for inverse manufactured experiments.

The manufactured setting is handled by subtracting the prescribed source terms
from the dynamic residuals R_phi and R_psi. This is used for verification and
diagnostic experiments, not for leakage-free blind inference.
"""

from argparse import Namespace
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from manufactured_bio_2d import exact_phi_psi, manufactured_sources
from utils_bio_2d import grad, pde_residuals


def pde_loss(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    model: nn.Module,
    args: Namespace,
    m_phi_eff: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the PDE residual loss for the 2D mixed bio phase-field PINN.

    The network is assumed to output:
        (phi, mu, psi, nu)

    The residuals are computed by `utils_bio_2d.pde_residuals`.
    If `args.manufactured == True`, manufactured source terms are subtracted
    from the dynamic residuals R_phi and R_psi.

    Parameters
    ----------
    x, y, t:
        Collocation coordinates, each with shape (N, 1).
    model:
        PINN model mapping (x, y, t) to (phi, mu, psi, nu).
    args:
        Runtime configuration containing PDE weights and physical parameters.
    m_phi_eff:
        Optional effective mobility. This is used in inverse runs when m_phi is
        a trainable parameter. If None, the value stored in args is used inside
        `pde_residuals`.

    Returns
    -------
    total_pde_loss:
        Weighted sum of the four residual MSEs.
    loss_pde_phi, loss_pde_mu, loss_pde_psi, loss_pde_nu:
        Individual unweighted residual losses.
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    phi, mu, psi, nu = model(x, y, t)

    res_phi, res_mu, res_psi, res_nu = pde_residuals(
        x=x,
        y=y,
        t=t,
        phi=phi,
        psi=psi,
        mu=mu,
        nu=nu,
        args=args,
        m_phi_eff=m_phi_eff,
    )

    if getattr(args, "manufactured", False):
        source_phi, source_psi = manufactured_sources(x, y, t, args)

        res_phi = res_phi - source_phi
        res_psi = res_psi - source_psi

    loss_pde_phi = torch.mean(res_phi**2)
    loss_pde_mu = torch.mean(res_mu**2)
    loss_pde_psi = torch.mean(res_psi**2)
    loss_pde_nu = torch.mean(res_nu**2)

    total_pde_loss = (
        args.pde_phi_w * loss_pde_phi
        + args.pde_mu_w * loss_pde_mu
        + args.pde_psi_w * loss_pde_psi
        + args.pde_nu_w * loss_pde_nu
    )

    return total_pde_loss, loss_pde_phi, loss_pde_mu, loss_pde_psi, loss_pde_nu


def ic_loss(
    model: nn.Module,
    x_ic: Tensor,
    y_ic: Tensor,
    t_ic: Tensor,
    phi_ic_true: Tensor,
    mu_ic_true: Tensor,
    psi_ic_true: Tensor,
    nu_ic_true: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the initial-condition loss for all four fields.

    Parameters
    ----------
    model:
        PINN model mapping (x, y, t) to (phi, mu, psi, nu).
    x_ic, y_ic, t_ic:
        Initial-condition coordinates.
    phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true:
        Target initial profiles.

    Returns
    -------
    total_ic_loss:
        Sum of the four initial-condition MSEs.
    loss_ic_phi, loss_ic_mu, loss_ic_psi, loss_ic_nu:
        Individual initial-condition losses.
    """
    phi_ic_pred, mu_ic_pred, psi_ic_pred, nu_ic_pred = model(x_ic, y_ic, t_ic)

    loss_ic_phi = torch.mean((phi_ic_pred - phi_ic_true) ** 2)
    loss_ic_mu = torch.mean((mu_ic_pred - mu_ic_true) ** 2)
    loss_ic_psi = torch.mean((psi_ic_pred - psi_ic_true) ** 2)
    loss_ic_nu = torch.mean((nu_ic_pred - nu_ic_true) ** 2)

    total_ic_loss = loss_ic_phi + loss_ic_mu + loss_ic_psi + loss_ic_nu

    return total_ic_loss, loss_ic_phi, loss_ic_mu, loss_ic_psi, loss_ic_nu


def bc_loss(
    model: nn.Module,
    x_bc: Tensor,
    y_bc: Tensor,
    t_bc: Tensor,
    normal: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the homogeneous Neumann boundary-condition loss.

    The zero-flux condition is imposed on all four network outputs:
        d_n phi = d_n mu = d_n psi = d_n nu = 0.

    Parameters
    ----------
    model:
        PINN model mapping (x, y, t) to (phi, mu, psi, nu).
    x_bc, y_bc, t_bc:
        Boundary coordinates.
    normal:
        Outward unit normal vectors at the boundary points, shape (N, 2).

    Returns
    -------
    total_bc_loss:
        Sum of the four Neumann boundary MSEs.
    loss_bc_phi, loss_bc_mu, loss_bc_psi, loss_bc_nu:
        Individual boundary losses.
    """
    x_bc.requires_grad_(True)
    y_bc.requires_grad_(True)

    phi_b, mu_b, psi_b, nu_b = model(x_bc, y_bc, t_bc)

    phi_b_x = grad(phi_b, x_bc)
    phi_b_y = grad(phi_b, y_bc)

    mu_b_x = grad(mu_b, x_bc)
    mu_b_y = grad(mu_b, y_bc)

    psi_b_x = grad(psi_b, x_bc)
    psi_b_y = grad(psi_b, y_bc)

    nu_b_x = grad(nu_b, x_bc)
    nu_b_y = grad(nu_b, y_bc)

    nx = normal[:, 0:1]
    ny = normal[:, 1:2]

    phi_b_n = phi_b_x * nx + phi_b_y * ny
    mu_b_n = mu_b_x * nx + mu_b_y * ny
    psi_b_n = psi_b_x * nx + psi_b_y * ny
    nu_b_n = nu_b_x * nx + nu_b_y * ny

    loss_bc_phi = torch.mean(phi_b_n**2)
    loss_bc_mu = torch.mean(mu_b_n**2)
    loss_bc_psi = torch.mean(psi_b_n**2)
    loss_bc_nu = torch.mean(nu_b_n**2)

    total_bc_loss = loss_bc_phi + loss_bc_mu + loss_bc_psi + loss_bc_nu

    return total_bc_loss, loss_bc_phi, loss_bc_mu, loss_bc_psi, loss_bc_nu


def add_deterministic_relative_noise(
    u: Tensor,
    noise_level: float,
    seed: Optional[int],
    offset: int = 0,
) -> Tensor:
    """
    Add deterministic relative Gaussian noise to a tensor.

    The noise amplitude is scaled by the standard deviation of the detached
    target tensor. If the tensor is nearly constant, the mean absolute value is
    used as a fallback scale.

    This function is used to perturb manufactured data in a reproducible way for
    inverse robustness tests.

    Parameters
    ----------
    u:
        Target tensor.
    noise_level:
        Relative noise amplitude. If <= 0, the tensor is returned unchanged.
    seed:
        Base random seed. Required when noise_level > 0.
    offset:
        Integer offset added to the seed. This allows different fields to receive
        independent deterministic noise.

    Returns
    -------
    Tensor
        Noisy tensor with the same shape as u.
    """
    if noise_level is None or noise_level <= 0.0:
        return u

    if seed is None:
        raise ValueError("noise_level > 0 but seed is None.")

    u_detached = u.detach()
    scale = torch.std(u_detached)

    if scale.item() < 1e-12:
        scale = torch.mean(torch.abs(u_detached)) + 1e-12

    generator = torch.Generator(device=u.device)
    generator.manual_seed(int(seed) + int(offset))

    noise = torch.randn(
        u.shape,
        generator=generator,
        device=u.device,
        dtype=u.dtype,
    )

    return u + noise_level * scale * noise


def data_loss_manufactured(
    model: nn.Module,
    x_data: Tensor,
    y_data: Tensor,
    t_data: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute the data loss for manufactured inverse experiments.

    The data term compares the predicted physical fields phi and psi against the
    exact manufactured fields. The auxiliary potentials mu and nu are not used as
    observed data.

    If `args.data_noise > 0`, deterministic relative Gaussian noise is added to
    the manufactured phi and psi targets.

    Parameters
    ----------
    model:
        PINN model mapping (x, y, t) to (phi, mu, psi, nu).
    x_data, y_data, t_data:
        Data coordinates.
    args:
        Runtime configuration. Expected optional attributes:
        - data_noise
        - data_noise_segment_seed

    Returns
    -------
    total_data_loss:
        Sum of phi and psi data MSEs.
    loss_phi:
        Data MSE for phi.
    loss_psi:
        Data MSE for psi.
    """
    phi_pred, _, psi_pred, _ = model(x_data, y_data, t_data)

    phi_exact, psi_exact = exact_phi_psi(x_data, y_data, t_data, args)

    if getattr(args, "data_noise", 0.0) > 0.0:
        seed = getattr(args, "data_noise_segment_seed", None)

        phi_exact = add_deterministic_relative_noise(
            u=phi_exact,
            noise_level=args.data_noise,
            seed=seed,
            offset=0,
        )

        psi_exact = add_deterministic_relative_noise(
            u=psi_exact,
            noise_level=args.data_noise,
            seed=seed,
            offset=100000,
        )

    loss_phi = torch.mean((phi_pred - phi_exact) ** 2)
    loss_psi = torch.mean((psi_pred - psi_exact) ** 2)

    total_data_loss = loss_phi + loss_psi

    return total_data_loss, loss_phi, loss_psi