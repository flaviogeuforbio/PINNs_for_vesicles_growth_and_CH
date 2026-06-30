"""
Utility functions for the 2D bio-inspired phase-field PINN.

This module contains:

1. Autograd helpers:
   - first derivatives
   - Laplacian

2. Phase-field constitutive functions:
   - interpolation p(phi)
   - double-well potential g(phi)
   - osmotic free-energy densities
   - surface free-energy density

3. Initial-condition construction for the minimal bio-inspired model.

4. Collocation-point generation and residual-based adaptive resampling.

5. Model loading and time-windowed prediction utilities.

6. PDE residuals for the mixed four-output formulation:
       (phi, mu, psi, nu)

The code is used in the bio_phasefield_2d branch for forward runs,
manufactured verification, and manufactured inverse diagnostics.
"""

import math
from argparse import Namespace
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn

from manufactured_bio_2d import exact_fields_ms_target
from models import BioACCHPINN2d


# Constant used in the phase-field free-energy density.
# In the vesicle phase-field model this normalization makes the diffuse
# interface energy approximate the sharp-interface surface measure.
K_SURF = 3.0 * math.sqrt(2.0) / 4.0


# =============================================================================
# Autograd helpers
# =============================================================================

def grad(u: Tensor, x: Tensor) -> Tensor:
    """
    Compute the first derivative du/dx using PyTorch autograd.

    Parameters
    ----------
    u:
        Tensor-valued function evaluated at x.
    x:
        Input tensor with respect to which the derivative is computed.

    Returns
    -------
    Tensor
        The derivative du/dx.
    """
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def laplacian(u: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """
    Compute the 2D Laplacian of u with respect to x and y.

    Parameters
    ----------
    u:
        Scalar field evaluated at collocation points.
    x, y:
        Coordinate tensors with requires_grad=True.

    Returns
    -------
    Tensor
        Laplacian u_xx + u_yy.
    """
    u_x = grad(u, x)
    u_y = grad(u, y)

    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    return u_xx + u_yy


def integral_2d(field: Tensor, x_lin: Tensor, y_lin: Tensor, n_grid: int) -> Tensor:
    """
    Approximate a 2D integral over a tensor-product grid using trapezoidal rule.

    Parameters
    ----------
    field:
        Flattened field with shape (n_grid*n_grid, 1) or compatible.
    x_lin, y_lin:
        One-dimensional grid coordinates.
    n_grid:
        Number of grid points per spatial direction.

    Returns
    -------
    Tensor
        Approximate integral over the 2D domain.
    """
    field_grid = field.reshape(n_grid, n_grid)

    int_y = torch.trapz(field_grid, y_lin, dim=1)
    int_xy = torch.trapz(int_y, x_lin, dim=0)

    return int_xy


# =============================================================================
# Phase-field constitutive functions
# =============================================================================

def p_interp(phi: Tensor) -> Tensor:
    """
    Smooth interpolation function selecting inside/outside of the vesicle.

    p(+1)=+1, p(-1)=-1, and p'(±1)=0.
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
    Osmotic free-energy density inside the vesicle.
    """
    return 0.5 * lambda_in * (psi - psi_eq) ** 2 + beta_in


def f_out(
    psi: Tensor,
    psi_eq: float,
    lambda_out: float,
    beta_out: float,
) -> Tensor:
    """
    Osmotic free-energy density outside the vesicle.
    """
    return 0.5 * lambda_out * (psi - psi_eq) ** 2 + beta_out


def f_surf_density(
    phi: Tensor,
    phi_x: Tensor,
    phi_y: Tensor,
    eps: float,
) -> Tensor:
    """
    Diffuse surface free-energy density for the phase field phi.
    """
    return K_SURF * (
        (1.0 / eps) * g(phi)
        + 0.5 * eps * (phi_x**2 + phi_y**2)
    )


def f_osm_density(
    phi: Tensor,
    psi: Tensor,
    psi_in_eq: float,
    psi_out_eq: float,
    lambda_in: float,
    lambda_out: float,
    beta_in: float,
    beta_out: float,
) -> Tensor:
    """
    Osmotic free-energy density interpolated between vesicle inside and outside.
    """
    p_phi = p_interp(phi)

    f_in_val = f_in(
        psi=psi,
        psi_eq=psi_in_eq,
        lambda_in=lambda_in,
        beta_in=beta_in,
    )

    f_out_val = f_out(
        psi=psi,
        psi_eq=psi_out_eq,
        lambda_out=lambda_out,
        beta_out=beta_out,
    )

    return 0.5 * (1.0 + p_phi) * f_in_val + 0.5 * (1.0 - p_phi) * f_out_val


def compute_energy(
    phi: Tensor,
    psi: Tensor,
    phi_x: Tensor,
    phi_y: Tensor,
    x_lin: Tensor,
    y_lin: Tensor,
    n_grid: int,
    args: Namespace,
    area_target: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Compute the minimal free energy on a 2D grid.

    In the minimal model used in this branch, the energy contains:
        F = F_surf + F_osm

    Optionally, an area penalty can be added if `area_target` is provided and
    `args.lambda_area` exists.

    Parameters
    ----------
    phi, psi:
        Predicted physical fields on a flattened n_grid x n_grid grid.
    phi_x, phi_y:
        Spatial derivatives of phi.
    x_lin, y_lin:
        One-dimensional grid coordinates.
    n_grid:
        Number of points per spatial direction.
    args:
        Runtime configuration.
    area_target:
        Optional target diffuse area.

    Returns
    -------
    total_energy, surf_energy, osm_energy, area_energy
        area_energy is None unless the optional area penalty is enabled.
    """
    surf_density = args.lambda_surf * f_surf_density(
        phi=phi,
        phi_x=phi_x,
        phi_y=phi_y,
        eps=args.eps,
    )

    osm_density = f_osm_density(
        phi=phi,
        psi=psi,
        psi_in_eq=args.psi_in_eq,
        psi_out_eq=args.psi_out_eq,
        lambda_in=args.lambda_in,
        lambda_out=args.lambda_out,
        beta_in=args.beta_in,
        beta_out=args.beta_out,
    )

    surf_energy = integral_2d(surf_density, x_lin, y_lin, n_grid)
    osm_energy = integral_2d(osm_density, x_lin, y_lin, n_grid)

    total_energy = surf_energy + osm_energy

    area_energy = None

    if area_target is not None and hasattr(args, "lambda_area"):
        diffuse_area = integral_2d(
            f_surf_density(phi, phi_x, phi_y, args.eps),
            x_lin,
            y_lin,
            n_grid,
        )

        area_energy = 0.5 * args.lambda_area * (diffuse_area - area_target) ** 2
        total_energy = total_energy + area_energy

    return total_energy, surf_energy, osm_energy, area_energy


# =============================================================================
# Initial conditions
# =============================================================================

def initial_phi(
    x: Tensor,
    y: Tensor,
    eps: float,
    radius: float = 0.28,
    x0: float = 0.5,
    y0: float = 0.5,
    pert_a: float = 0.0,
    pert_mode: int = 4,
) -> Tensor:
    """
    Diffuse circular vesicle initial condition.

    The interface is represented by the standard tanh profile:
        phi = tanh((R(theta) - r)/(sqrt(2)*eps))

    A small angular perturbation can optionally be added to the radius.
    """
    dx = x - x0
    dy = y - y0

    r = torch.sqrt(dx**2 + dy**2)
    theta = torch.atan2(dy, dx)

    r_theta = radius * (1.0 + pert_a * torch.cos(pert_mode * theta))

    return torch.tanh((r_theta - r) / (math.sqrt(2.0) * eps))


def initial_psi(phi: Tensor, psi_in_0: float, psi_out_0: float) -> Tensor:
    """
    Smooth inside/outside initial concentration profile.

    The phase-field interpolation selects the interior and exterior values.
    """
    p_phi = p_interp(phi)
    chi_in = 0.5 * (1.0 + p_phi)

    return chi_in * psi_in_0 + (1.0 - chi_in) * psi_out_0


def initial_fields(
    x: Tensor,
    y: Tensor,
    args: Namespace,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the initial profiles for (phi, mu, psi, nu).

    The physical fields phi and psi are constructed from the diffuse disk IC.
    The auxiliary potentials mu and nu are then computed consistently from the
    constitutive relations of the minimal model.
    """
    x_ic = x.clone().detach().requires_grad_(True)
    y_ic = y.clone().detach().requires_grad_(True)

    phi_0 = initial_phi(
        x=x_ic,
        y=y_ic,
        eps=args.eps,
        radius=getattr(args, "radius", 0.28),
        x0=getattr(args, "x0", 0.5),
        y0=getattr(args, "y0", 0.5),
        pert_a=getattr(args, "pert_a", 0.0),
        pert_mode=getattr(args, "pert_mode", 4),
    )

    psi_0 = initial_psi(
        phi=phi_0,
        psi_in_0=args.psi_in_0,
        psi_out_0=args.psi_out_0,
    )

    lap_phi = laplacian(phi_0, x_ic, y_ic)

    p_phi = p_interp(phi_0)
    p_phi_der = p_interp_der(phi_0)
    g_phi_der = g_der(phi_0)

    f_in_values = f_in(
        psi=psi_0,
        psi_eq=args.psi_in_eq,
        lambda_in=args.lambda_in,
        beta_in=args.beta_in,
    )

    f_out_values = f_out(
        psi=psi_0,
        psi_eq=args.psi_out_eq,
        lambda_out=args.lambda_out,
        beta_out=args.beta_out,
    )

    mu_0 = (
        args.lambda_surf
        * K_SURF
        * ((1.0 / args.eps) * g_phi_der - args.eps * lap_phi)
        + 0.5 * p_phi_der * (f_in_values - f_out_values)
    )

    nu_0 = (
        0.5 * (1.0 + p_phi) * args.lambda_in * (psi_0 - args.psi_in_eq)
        + 0.5 * (1.0 - p_phi) * args.lambda_out * (psi_0 - args.psi_out_eq)
    )

    return phi_0.detach(), mu_0.detach(), psi_0.detach(), nu_0.detach()


# =============================================================================
# Collocation points and adaptive resampling
# =============================================================================

def generate_boundary_points(
    n_bc: int,
    lx: float,
    ly: float,
    tmax: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Generate random boundary collocation points on the four sides of the box.

    Homogeneous Neumann conditions are imposed using the associated outward
    normal vectors.

    Notes
    -----
    If n_bc is not divisible by 4, the remainder is ignored to preserve the
    original branch behavior.
    """
    n_side = n_bc // 4

    x_left = torch.zeros((n_side, 1), device=device)
    y_left = torch.rand((n_side, 1), device=device) * ly
    t_left = torch.rand((n_side, 1), device=device) * tmax
    n_left = torch.tensor([-1.0, 0.0], device=device).repeat(n_side, 1)

    x_right = torch.ones((n_side, 1), device=device) * lx
    y_right = torch.rand((n_side, 1), device=device) * ly
    t_right = torch.rand((n_side, 1), device=device) * tmax
    n_right = torch.tensor([1.0, 0.0], device=device).repeat(n_side, 1)

    x_down = torch.rand((n_side, 1), device=device) * lx
    y_down = torch.zeros((n_side, 1), device=device)
    t_down = torch.rand((n_side, 1), device=device) * tmax
    n_down = torch.tensor([0.0, -1.0], device=device).repeat(n_side, 1)

    x_up = torch.rand((n_side, 1), device=device) * lx
    y_up = torch.ones((n_side, 1), device=device) * ly
    t_up = torch.rand((n_side, 1), device=device) * tmax
    n_up = torch.tensor([0.0, 1.0], device=device).repeat(n_side, 1)

    x_bc = torch.cat([x_left, x_right, x_down, x_up], dim=0)
    y_bc = torch.cat([y_left, y_right, y_down, y_up], dim=0)
    t_bc = torch.cat([t_left, t_right, t_down, t_up], dim=0)
    normal_bc = torch.cat([n_left, n_right, n_down, n_up], dim=0)

    return x_bc, y_bc, t_bc, normal_bc


def generate_coll_points_and_ic(
    args: Namespace,
    device: torch.device,
    ic_fn: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]] = None,
) -> Tuple[dict, Tensor, Tensor, Tensor, Tensor]:
    """
    Generate PDE, BC, IC and optional data collocation points.

    If ic_fn is None, the initial condition is either:
    - the manufactured exact field at t=0, if args.manufactured=True;
    - the bio-inspired diffuse-disk initial condition otherwise.

    If ic_fn is provided, it is used to initialize a new time-window segment from
    the previous segment prediction.
    """
    n_pde = args.n_pde
    n_bc = args.n_bc
    n_ic = args.n_ic
    n_data = args.n_data

    tmax = args.segment_length
    lx = args.lx
    ly = args.ly

    x_pde = torch.rand((n_pde, 1), device=device) * lx
    y_pde = torch.rand((n_pde, 1), device=device) * ly
    t_pde = torch.rand((n_pde, 1), device=device) * tmax

    x_ic = torch.rand((n_ic, 1), device=device) * lx
    y_ic = torch.rand((n_ic, 1), device=device) * ly
    t_ic = torch.zeros((n_ic, 1), device=device)

    x_bc, y_bc, t_bc, normal_bc = generate_boundary_points(
        n_bc=n_bc,
        lx=lx,
        ly=ly,
        tmax=tmax,
        device=device,
    )

    x_data = torch.rand((n_data, 1), device=device) * lx
    y_data = torch.rand((n_data, 1), device=device) * ly
    t_data = torch.rand((n_data, 1), device=device) * tmax

    collocation = {
        "x_pde": x_pde,
        "y_pde": y_pde,
        "t_pde": t_pde,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "t_bc": t_bc,
        "normal_bc": normal_bc,
        "x_ic": x_ic,
        "y_ic": y_ic,
        "t_ic": t_ic,
        "x_data": x_data,
        "y_data": y_data,
        "t_data": t_data,
    }

    if ic_fn is None:
        if getattr(args, "manufactured", False):
            phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = exact_fields_ms_target(
                x_ic,
                y_ic,
                t_ic,
                args,
            )
        else:
            phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = initial_fields(
                x_ic,
                y_ic,
                args,
            )

        phi_ic_true = phi_ic_true.to(device)
        mu_ic_true = mu_ic_true.to(device)
        psi_ic_true = psi_ic_true.to(device)
        nu_ic_true = nu_ic_true.to(device)

    else:
        phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = ic_fn(x_ic, y_ic)

    return collocation, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true


def adaptive_resample_pde_points(
    model: nn.Module,
    args: Namespace,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Residual-based adaptive resampling of PDE collocation points.

    A large candidate set is sampled uniformly. Candidate points are scored using
    a weighted residual magnitude, with R_psi emphasized because the conservative
    concentration residual was empirically the hardest term to optimize.

    The new PDE set is a mixture of:
    - high-score adaptive points;
    - fresh uniform points for global coverage.
    """
    model.eval()

    n_adapt = int(args.adaptive_frac * args.n_pde)
    n_uniform = args.n_pde - n_adapt
    n_candidates = args.n_candidates_resamp

    x_cand = torch.rand((n_candidates, 1), device=device) * args.lx
    y_cand = torch.rand((n_candidates, 1), device=device) * args.ly
    t_cand = torch.rand((n_candidates, 1), device=device) * args.segment_length

    x_cand.requires_grad_(True)
    y_cand.requires_grad_(True)
    t_cand.requires_grad_(True)

    phi, mu, psi, nu = model(x_cand, y_cand, t_cand)

    res_phi, res_mu, res_psi, res_nu = pde_residuals(
        x=x_cand,
        y=y_cand,
        t=t_cand,
        phi=phi,
        psi=psi,
        mu=mu,
        nu=nu,
        args=args,
    )

    score = (
        torch.abs(res_psi)
        + 0.25 * torch.abs(res_mu)
        + 0.25 * torch.abs(res_phi)
        + 0.25 * torch.abs(res_nu)
    )

    score = score.detach().flatten()

    topk_idx = torch.topk(score, k=n_adapt, largest=True).indices

    x_adapt = x_cand.detach()[topk_idx]
    y_adapt = y_cand.detach()[topk_idx]
    t_adapt = t_cand.detach()[topk_idx]

    x_uniform = torch.rand((n_uniform, 1), device=device) * args.lx
    y_uniform = torch.rand((n_uniform, 1), device=device) * args.ly
    t_uniform = torch.rand((n_uniform, 1), device=device) * args.segment_length

    x_pde = torch.cat([x_adapt, x_uniform], dim=0).detach()
    y_pde = torch.cat([y_adapt, y_uniform], dim=0).detach()
    t_pde = torch.cat([t_adapt, t_uniform], dim=0).detach()

    print(
        f"Adaptive PDE resampling done | "
        f"N_candidates={n_candidates} | "
        f"N_adapt={n_adapt} | "
        f"N_uniform={n_uniform} | "
        f"score_max={score.max().item():.4e} | "
        f"score_mean={score.mean().item():.4e}"
    )

    model.train()

    return x_pde, y_pde, t_pde


# =============================================================================
# Model loading and time-windowed prediction
# =============================================================================

def load_model(
    model_checkpoint: str,
    hidden_layers: int,
    hidden_dim: int,
    device: Optional[torch.device] = None,
) -> BioACCHPINN2d:
    """
    Load a single BioACCHPINN2d model checkpoint.
    """
    model = BioACCHPINN2d(
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
    )

    if device is None:
        state_dict = torch.load(model_checkpoint)
    else:
        state_dict = torch.load(model_checkpoint, map_location=device)
        model = model.to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_models(
    check_dir: Path,
    hidden_layers: int,
    hidden_dim: int,
    device: torch.device,
) -> List[BioACCHPINN2d]:
    """
    Load all time-window segment models from a checkpoint directory.

    Checkpoints are expected to be named:
        segment_0.pt, segment_1.pt, ...
    """
    models = []

    check_paths = sorted(
        check_dir.glob("segment_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    if len(check_paths) == 0:
        raise FileNotFoundError(f"No segment_*.pt checkpoints found in {check_dir}")

    print("Loading segment models in order:")

    for check_path in check_paths:
        print(f"  {check_path.name}")

        model = BioACCHPINN2d(
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
        )

        model.load_state_dict(torch.load(check_path, map_location=device))
        model = model.to(device)
        model.eval()

        models.append(model)

    return models


def make_ic_from_previous_model(
    previous_model: nn.Module,
    segment_length: float,
    device: torch.device,
    args: Namespace,
    recompute_potentials: bool = False,
) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Build the initial-condition function for the next time-window segment.

    If recompute_potentials=False, all four fields are copied from the previous
    model prediction at the final local time.

    If recompute_potentials=True, only phi and psi are copied; mu and nu are
    recomputed from the constitutive relations to improve consistency.
    """
    previous_model.eval()

    def ic_fn(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x_eval = x.detach().to(device)
        y_eval = y.detach().to(device)
        t_eval = torch.full_like(x_eval, float(segment_length), device=device)

        if recompute_potentials:
            x_eval = x_eval.requires_grad_(True)
            y_eval = y_eval.requires_grad_(True)

            phi_ic_true, _, psi_ic_true, _ = previous_model(x_eval, y_eval, t_eval)

            lap_phi = laplacian(phi_ic_true, x_eval, y_eval)

            p_phi = p_interp(phi_ic_true)
            p_phi_der = p_interp_der(phi_ic_true)
            g_phi_der = g_der(phi_ic_true)

            f_in_values = f_in(
                psi=psi_ic_true,
                psi_eq=args.psi_in_eq,
                lambda_in=args.lambda_in,
                beta_in=args.beta_in,
            )

            f_out_values = f_out(
                psi=psi_ic_true,
                psi_eq=args.psi_out_eq,
                lambda_out=args.lambda_out,
                beta_out=args.beta_out,
            )

            mu_ic_true = (
                args.lambda_surf
                * K_SURF
                * ((1.0 / args.eps) * g_phi_der - args.eps * lap_phi)
                + 0.5 * p_phi_der * (f_in_values - f_out_values)
            )

            nu_ic_true = (
                0.5 * (1.0 + p_phi) * args.lambda_in * (psi_ic_true - args.psi_in_eq)
                + 0.5 * (1.0 - p_phi) * args.lambda_out * (psi_ic_true - args.psi_out_eq)
            )

        else:
            with torch.no_grad():
                phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = previous_model(
                    x_eval,
                    y_eval,
                    t_eval,
                )

        return (
            phi_ic_true.detach().to(device),
            mu_ic_true.detach().to(device),
            psi_ic_true.detach().to(device),
            nu_ic_true.detach().to(device),
        )

    return ic_fn


def predict_windowed(
    models: List[nn.Module],
    x: Tensor,
    y: Tensor,
    t_global: float,
    segment_length: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Predict fields at a scalar global time using a time-windowed model ensemble.
    """
    segment_idx = int(min(t_global // segment_length, len(models) - 1))
    tau = t_global - segment_idx * segment_length

    t_local = torch.full_like(x, float(tau), device=device)

    model = models[segment_idx]
    model.eval()

    return model(x, y, t_local)


def predict_windowed_tensor(
    models: List[nn.Module],
    x: Tensor,
    y: Tensor,
    t_global: Tensor,
    segment_length: float,
    device: torch.device,
) -> Tensor:
    """
    Predict fields at tensor-valued global times using a time-windowed ensemble.

    Returns
    -------
    Tensor
        Tensor of shape (N, 4), with columns:
        phi, mu, psi, nu.
    """
    x = x.to(device)
    y = y.to(device)
    t_global = t_global.to(device)

    segment_idxs = torch.floor(t_global / segment_length).long()
    segment_idxs = torch.clamp(segment_idxs, min=0, max=len(models) - 1)

    tau = t_global - segment_idxs.float() * segment_length

    out = torch.zeros((x.shape[0], 4), device=device)

    for segment_idx, model in enumerate(models):
        mask = segment_idxs[:, 0] == segment_idx

        if not torch.any(mask):
            continue

        model.eval()

        x_s = x[mask]
        y_s = y[mask]
        tau_s = tau[mask]

        phi_s, mu_s, psi_s, nu_s = model(x_s, y_s, tau_s)

        out[mask, 0:1] = phi_s
        out[mask, 1:2] = mu_s
        out[mask, 2:3] = psi_s
        out[mask, 3:4] = nu_s

    return out


# =============================================================================
# PDE residuals
# =============================================================================

def pde_residuals(
    x: Tensor,
    y: Tensor,
    t: Tensor,
    phi: Tensor,
    psi: Tensor,
    mu: Tensor,
    nu: Tensor,
    args: Namespace,
    m_phi_eff=None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the residuals of the minimal 2D bio-inspired phase-field model.

    Mixed formulation:
        R_phi = phi_t + m_phi * mu

        R_mu  = mu - [lambda_surf*k*(eps^{-1} g'(phi) - eps*Delta phi)
                      + 1/2 p'(phi)(f_in(psi)-f_out(psi))]

        R_psi = psi_t + div(J_psi),
                with J_psi = -M_psi(phi) grad(nu)

        R_nu  = nu - [1/2(1+p(phi))*lambda_in*(psi-psi_in_eq)
                      + 1/2(1-p(phi))*lambda_out*(psi-psi_out_eq)]

    Parameters
    ----------
    x, y, t:
        Space-time collocation coordinates with requires_grad=True.
    phi, psi, mu, nu:
        Network-predicted fields.
    args:
        Runtime configuration.
    m_phi_eff:
        Optional effective mobility. If None, args.m_phi is used.

    Returns
    -------
    res_phi, res_mu, res_psi, res_nu
        PDE residual tensors.
    """
    phi_t = grad(phi, t)
    psi_t = grad(psi, t)

    nu_x = grad(nu, x)
    nu_y = grad(nu, y)

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

    m_psi = 1.0 - args.m0 * ((phi**2 - 1.0) ** 2)

    psi_current_x = -m_psi * nu_x
    psi_current_y = -m_psi * nu_y

    if m_phi_eff is None:
        m_phi_eff = args.m_phi

    res_phi = phi_t + m_phi_eff * mu

    res_mu = mu - (
        args.lambda_surf
        * K_SURF
        * ((1.0 / args.eps) * g_phi_der - args.eps * lap_phi)
        + 0.5 * p_phi_der * (f_in_values - f_out_values)
    )

    res_psi = psi_t + grad(psi_current_x, x) + grad(psi_current_y, y)

    res_nu = nu - (
        0.5 * (1.0 + p_phi) * args.lambda_in * (psi - args.psi_in_eq)
        + 0.5 * (1.0 - p_phi) * args.lambda_out * (psi - args.psi_out_eq)
    )

    return res_phi, res_mu, res_psi, res_nu