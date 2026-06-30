"""
Manufactured solutions for the 2D bio-inspired phase-field PINN.

This module defines the exact fields used in manufactured-solution experiments
for the mixed four-output formulation:
    (phi, mu, psi, nu)

The manufactured setup provides:

1. Exact physical fields:
       phi_exact(x, y, t), psi_exact(x, y, t)

2. Consistent auxiliary potentials:
       mu_exact, nu_exact

3. Source terms:
       S_phi, S_psi

so that the exact fields satisfy the forced PDE system:
       phi_t + m_phi * mu - S_phi = 0
       psi_t + div(-M_psi(phi) grad(nu)) - S_psi = 0

Important
---------
The manufactured sources depend explicitly on the target parameter m_phi. This
makes the manufactured setup useful for forward verification and diagnostics, but
not a leakage-free blind inverse benchmark.
"""

import math
from argparse import Namespace
from typing import Tuple

import torch
from torch import Tensor


# Same phase-field normalization constant used in utils_bio_2d.py.
# It is duplicated here to avoid circular imports.
K_SURF = 3.0 * math.sqrt(2.0) / 4.0


# =============================================================================
# Local autograd helpers
# =============================================================================

def grad(u: Tensor, x: Tensor) -> Tensor:
    """
    Compute the first derivative du/dx using PyTorch autograd.
    """
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]


def laplacian(u: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """
    Compute the 2D Laplacian u_xx + u_yy.
    """
    u_x = grad(u, x)
    u_y = grad(u, y)

    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    return u_xx + u_yy


# =============================================================================
# Local constitutive functions
# =============================================================================

def p_interp(phi: Tensor) -> Tensor:
    """
    Smooth interpolation function selecting the vesicle inside/outside.
    """
    return -0.5 * phi**3 + 1.5 * phi


def p_interp_der(phi: Tensor) -> Tensor:
    """
    Derivative of p_interp(phi).
    """
    return 1.5 * (1.0 - phi**2)


def g(phi: Tensor) -> Tensor:
    """
    Double-well potential:
        g(phi) = 1/4 (phi^2 - 1)^2.
    """
    return 0.25 * (phi**2 - 1.0) ** 2


def g_der(phi: Tensor) -> Tensor:
    """
    Derivative of the double-well potential:
        g'(phi) = phi^3 - phi.
    """
    return phi**3 - phi


def f_in(
    psi: Tensor,
    psi_eq: float,
    lambda_in: float,
    beta_in: float,
) -> Tensor:
    """
    Inner osmotic free-energy density.
    """
    return 0.5 * lambda_in * (psi - psi_eq) ** 2 + beta_in


def f_out(
    psi: Tensor,
    psi_eq: float,
    lambda_out: float,
    beta_out: float,
) -> Tensor:
    """
    Outer osmotic free-energy density.
    """
    return 0.5 * lambda_out * (psi - psi_eq) ** 2 + beta_out


# =============================================================================
# Manufactured exact fields
# =============================================================================

def exact_phi_psi(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the exact manufactured physical fields phi and psi.

    Two manufactured profiles are available.

    1. Smooth cosine manufactured solution, enabled by --ms_smooth:
       This profile is smooth and satisfies homogeneous normal derivatives on
       the unit square.

    2. Phase-field-like manufactured solution:
       This profile resembles a growing diffuse circular vesicle with an
       interpolated concentration field.
    """
    if getattr(args, "ms_smooth", False):
        # The cosine profile has zero normal derivative at the boundaries when
        # lx = ly = 1. This is the simplest manufactured verification target.
        spatial = torch.cos(math.pi * x) * torch.cos(math.pi * y)

        phi = 0.2 + 0.3 * spatial * torch.exp(-1.0 * t)
        psi = 0.5 + 0.2 * spatial * torch.exp(-0.7 * t)

        return phi, psi

    dx = x - args.x0
    dy = y - args.y0

    r = torch.sqrt(dx**2 + dy**2)
    radius_t = args.ms_R0 + args.ms_alpha_R * t

    phi = torch.tanh((radius_t - r) / (math.sqrt(2.0) * args.eps))

    psi_in_t = args.ms_psi_in0 + args.ms_beta_in * t
    psi_out_t = args.ms_psi_out0 + args.ms_beta_out * t

    p_phi = p_interp(phi)

    psi = (
        0.5 * (1.0 + p_phi) * psi_in_t
        + 0.5 * (1.0 - p_phi) * psi_out_t
    )

    return phi, psi


def compute_mu_nu_targets(
    phi: Tensor,
    psi: Tensor,
    x: Tensor,
    y: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the exact auxiliary potentials mu and nu from phi and psi.

    These targets are not prescribed independently: they are obtained from the
    same constitutive relations used in the PDE residuals.
    """
    lap_phi = laplacian(phi, x, y)

    p_phi = p_interp(phi)
    p_phi_der = p_interp_der(phi)
    g_phi_der = g_der(phi)

    f_in_values = f_in(
        psi=psi,
        psi_eq=args.psi_in_eq,
        lambda_in=args.lambda_in,
        beta_in=args.beta_in,
    )

    f_out_values = f_out(
        psi=psi,
        psi_eq=args.psi_out_eq,
        lambda_out=args.lambda_out,
        beta_out=args.beta_out,
    )

    mu_target = (
        args.lambda_surf
        * K_SURF
        * ((1.0 / args.eps) * g_phi_der - args.eps * lap_phi)
        + 0.5 * p_phi_der * (f_in_values - f_out_values)
    )

    nu_target = (
        0.5 * (1.0 + p_phi) * args.lambda_in * (psi - args.psi_in_eq)
        + 0.5 * (1.0 - p_phi) * args.lambda_out * (psi - args.psi_out_eq)
    )

    return mu_target, nu_target


def exact_fields_ms(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute all exact manufactured fields:
        phi_exact, mu_exact, psi_exact, nu_exact

    The input coordinates are marked as differentiable because the auxiliary
    potentials require spatial derivatives of phi.
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    phi_exact, psi_exact = exact_phi_psi(x, y, t, args)

    mu_exact, nu_exact = compute_mu_nu_targets(
        phi=phi_exact,
        psi=psi_exact,
        x=x,
        y=y,
        args=args,
    )

    return phi_exact, mu_exact, psi_exact, nu_exact


def exact_fields_ms_target(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute detached manufactured targets for initial-condition losses.

    This version clones and detaches the coordinates before building the target
    graph, then returns detached fields. This avoids autograd graph conflicts
    when the targets are used inside training losses.
    """
    x_target = x.detach().clone().requires_grad_(True)
    y_target = y.detach().clone().requires_grad_(True)
    t_target = t.detach().clone().requires_grad_(True)

    phi_exact, mu_exact, psi_exact, nu_exact = exact_fields_ms(
        x_target,
        y_target,
        t_target,
        args,
    )

    return (
        phi_exact.detach(),
        mu_exact.detach(),
        psi_exact.detach(),
        nu_exact.detach(),
    )


# =============================================================================
# Manufactured source terms
# =============================================================================

def manufactured_sources(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor]:
    """
    Compute source terms for the forced manufactured PDE system.

    The sources are defined so that the manufactured fields satisfy:
        phi_t + m_phi * mu - S_phi = 0

        psi_t + div(-M_psi(phi) grad(nu)) - S_psi = 0

    where:
        M_psi(phi) = 1 - m0 * (phi^2 - 1)^2

    Notes
    -----
    S_phi contains args.m_phi explicitly. This is the parameter-leakage mechanism
    discussed in the report: manufactured inverse recovery of m_phi is therefore
    a diagnostic/forced-recovery experiment, not blind source-free inference.
    """
    # Build an independent coordinate graph for source construction.
    x_source = x.detach().clone().requires_grad_(True)
    y_source = y.detach().clone().requires_grad_(True)
    t_source = t.detach().clone().requires_grad_(True)

    phi_exact, mu_exact, psi_exact, nu_exact = exact_fields_ms(
        x_source,
        y_source,
        t_source,
        args,
    )

    phi_t = grad(phi_exact, t_source)
    psi_t = grad(psi_exact, t_source)

    nu_x = grad(nu_exact, x_source)
    nu_y = grad(nu_exact, y_source)

    m_psi = 1.0 - args.m0 * ((phi_exact**2 - 1.0) ** 2)

    psi_current_x = -m_psi * nu_x
    psi_current_y = -m_psi * nu_y

    div_psi_current = grad(psi_current_x, x_source) + grad(psi_current_y, y_source)

    source_phi = phi_t + args.m_phi * mu_exact
    source_psi = psi_t + div_psi_current

    return source_phi.detach(), source_psi.detach()