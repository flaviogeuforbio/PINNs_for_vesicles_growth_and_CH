"""
Training utilities for the 2D bio-inspired phase-field PINN.

This module contains the helper functions used to train one time segment of the
mixed four-output PINN:
    (x, y, t) -> (phi, mu, psi, nu)

Training is organized in three optional stages:

1. Initial-condition pretraining
   The network is first trained only on the initial condition. This anchors the
   four fields before the PDE residuals are activated.

2. Adam training
   The full composite loss is minimized:
       PDE + BC + IC (+ data, for manufactured inverse runs)

3. L-BFGS refinement
   Optional quasi-Newton refinement after Adam.

For inverse manufactured experiments, the shape mobility is represented as:
    m_phi = exp(log_m_phi)
and optimized jointly with the network parameters.
"""

import time
from argparse import Namespace
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adam

from losses_bio_2d import bc_loss, data_loss_manufactured, ic_loss, pde_loss
from models import BioACCHPINN2d
from utils_bio_2d import adaptive_resample_pde_points, generate_coll_points_and_ic


LossDict = Dict[str, list]


def get_effective_m_phi(args: Namespace, log_m_phi: Optional[Tensor] = None):
    """
    Return the mobility used in the PDE residual.

    In inverse runs, the mobility is trainable and represented as exp(log_m_phi).
    In forward/verification runs, the fixed value from args is used.
    """
    if log_m_phi is not None:
        return torch.exp(log_m_phi)

    return args.m_phi


def compute_weighted_total_loss(
    args: Namespace,
    l_pde: Tensor,
    l_bc: Tensor,
    l_ic: Tensor,
    l_data: Optional[Tensor] = None,
) -> Tensor:
    """
    Combine PDE, BC, IC and optional data losses with their scalar weights.

    The PDE loss returned by `pde_loss` already contains the per-equation weights
    pde_phi_w, pde_mu_w, pde_psi_w and pde_nu_w. Here we apply the global PDE
    weight, together with the BC, IC and optional data weights.
    """
    total_loss = (
        args.pde_weight * l_pde
        + args.bc_weight * l_bc
        + args.ic_weight * l_ic
    )

    if l_data is not None:
        total_loss = total_loss + args.data_weight * l_data

    return total_loss


def init_train_history() -> LossDict:
    """
    Initialize the loss-history dictionary used during Adam training.
    """
    return {
        "total": [],
        "pde": [],
        "pde_phi": [],
        "pde_mu": [],
        "pde_psi": [],
        "pde_nu": [],
        "bc": [],
        "bc_phi": [],
        "bc_mu": [],
        "bc_psi": [],
        "bc_nu": [],
        "ic": [],
        "ic_phi": [],
        "ic_mu": [],
        "ic_psi": [],
        "ic_nu": [],
        "data": [],
        "data_phi": [],
        "data_psi": [],
        "m_phi": [],
    }


def append_epoch_losses(history: LossDict, epoch_losses: Dict[str, float], inverse: bool) -> None:
    """
    Append one epoch of scalar losses to the global training history.
    """
    base_keys = [
        "total",
        "pde",
        "pde_phi",
        "pde_mu",
        "pde_psi",
        "pde_nu",
        "bc",
        "bc_phi",
        "bc_mu",
        "bc_psi",
        "bc_nu",
        "ic",
        "ic_phi",
        "ic_mu",
        "ic_psi",
        "ic_nu",
    ]

    for key in base_keys:
        history[key].append(epoch_losses[key])

    if inverse:
        history["data"].append(epoch_losses["data"])
        history["data_phi"].append(epoch_losses["data_phi"])
        history["data_psi"].append(epoch_losses["data_psi"])
        history["m_phi"].append(epoch_losses["m_phi"])


def train_one_epoch(
    model: nn.Module,
    collocation: Dict[str, Tensor],
    phi_ic_true: Tensor,
    mu_ic_true: Tensor,
    psi_ic_true: Tensor,
    nu_ic_true: Tensor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: Namespace,
    m_phi_eff=None,
) -> Dict[str, float]:
    """
    Perform one Adam training epoch.

    Parameters
    ----------
    model:
        Mixed four-output PINN.
    collocation:
        Dictionary containing PDE, BC, IC and optional data points.
    phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true:
        Initial-condition targets for the four fields.
    optimizer:
        Adam optimizer.
    epoch:
        Current epoch index, used only for logging.
    args:
        Runtime configuration.
    m_phi_eff:
        Effective mobility used in the PDE residual. In inverse runs this is
        usually exp(log_m_phi); otherwise it is args.m_phi.

    Returns
    -------
    dict
        Scalar loss values for the current epoch.
    """
    model.train()

    x_pde = collocation["x_pde"]
    y_pde = collocation["y_pde"]
    t_pde = collocation["t_pde"]

    x_bc = collocation["x_bc"]
    y_bc = collocation["y_bc"]
    t_bc = collocation["t_bc"]
    normal = collocation["normal_bc"]

    x_ic = collocation["x_ic"]
    y_ic = collocation["y_ic"]
    t_ic = collocation["t_ic"]

    inverse = getattr(args, "inverse_m_phi", False)

    optimizer.zero_grad()

    l_pde, l_pde_phi, l_pde_mu, l_pde_psi, l_pde_nu = pde_loss(
        x=x_pde,
        y=y_pde,
        t=t_pde,
        model=model,
        args=args,
        m_phi_eff=m_phi_eff,
    )

    l_bc, l_bc_phi, l_bc_mu, l_bc_psi, l_bc_nu = bc_loss(
        model=model,
        x_bc=x_bc,
        y_bc=y_bc,
        t_bc=t_bc,
        normal=normal,
    )

    l_ic, l_ic_phi, l_ic_mu, l_ic_psi, l_ic_nu = ic_loss(
        model=model,
        x_ic=x_ic,
        y_ic=y_ic,
        t_ic=t_ic,
        phi_ic_true=phi_ic_true,
        mu_ic_true=mu_ic_true,
        psi_ic_true=psi_ic_true,
        nu_ic_true=nu_ic_true,
    )

    l_data = None
    l_data_phi = None
    l_data_psi = None

    if inverse:
        x_data = collocation["x_data"]
        y_data = collocation["y_data"]
        t_data = collocation["t_data"]

        l_data, l_data_phi, l_data_psi = data_loss_manufactured(
            model=model,
            x_data=x_data,
            y_data=y_data,
            t_data=t_data,
            args=args,
        )

    loss = compute_weighted_total_loss(
        args=args,
        l_pde=l_pde,
        l_bc=l_bc,
        l_ic=l_ic,
        l_data=l_data,
    )

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        msg = (
            f"Epoch {epoch:05d} | "
            f"PDE(phi): {l_pde_phi.item():.4e} | "
            f"PDE(mu): {l_pde_mu.item():.4e} | "
            f"PDE(psi): {l_pde_psi.item():.4e} | "
            f"PDE(nu): {l_pde_nu.item():.4e} | "
            f"BC: {l_bc.item():.4e} | "
            f"IC: {l_ic.item():.4e}"
        )

        if inverse:
            msg += (
                f" | Data: {l_data.item():.4e}"
                f" | m_phi: {float(m_phi_eff.detach().cpu()):.6e}"
            )

        print(msg)

    epoch_results = {
        "total": float(loss.item()),
        "pde": float(l_pde.item()),
        "pde_phi": float(l_pde_phi.item()),
        "pde_mu": float(l_pde_mu.item()),
        "pde_psi": float(l_pde_psi.item()),
        "pde_nu": float(l_pde_nu.item()),
        "bc": float(l_bc.item()),
        "bc_phi": float(l_bc_phi.item()),
        "bc_mu": float(l_bc_mu.item()),
        "bc_psi": float(l_bc_psi.item()),
        "bc_nu": float(l_bc_nu.item()),
        "ic": float(l_ic.item()),
        "ic_phi": float(l_ic_phi.item()),
        "ic_mu": float(l_ic_mu.item()),
        "ic_psi": float(l_ic_psi.item()),
        "ic_nu": float(l_ic_nu.item()),
    }

    if inverse:
        epoch_results["data"] = float(l_data.item())
        epoch_results["data_phi"] = float(l_data_phi.item())
        epoch_results["data_psi"] = float(l_data_psi.item())
        epoch_results["m_phi"] = float(m_phi_eff.detach().cpu())

    return epoch_results


def pretrain_initial_condition(
    model: nn.Module,
    collocation: Dict[str, Tensor],
    phi_ic_true: Tensor,
    mu_ic_true: Tensor,
    psi_ic_true: Tensor,
    nu_ic_true: Tensor,
    optimizer: torch.optim.Optimizer,
    pretrain_epochs: int,
) -> LossDict:
    """
    Pretrain the model on the initial condition only.

    This phase gives the PINN a physically meaningful starting point before PDE
    residuals are activated.
    """
    model.train()

    x_ic = collocation["x_ic"]
    y_ic = collocation["y_ic"]
    t_ic = collocation["t_ic"]

    history = {
        "total": [],
        "ic_phi": [],
        "ic_mu": [],
        "ic_psi": [],
        "ic_nu": [],
    }

    for epoch in range(1, pretrain_epochs + 1):
        optimizer.zero_grad()

        l_ic, l_ic_phi, l_ic_mu, l_ic_psi, l_ic_nu = ic_loss(
            model=model,
            x_ic=x_ic,
            y_ic=y_ic,
            t_ic=t_ic,
            phi_ic_true=phi_ic_true,
            mu_ic_true=mu_ic_true,
            psi_ic_true=psi_ic_true,
            nu_ic_true=nu_ic_true,
        )

        l_ic.backward()
        optimizer.step()

        history["total"].append(float(l_ic.item()))
        history["ic_phi"].append(float(l_ic_phi.item()))
        history["ic_mu"].append(float(l_ic_mu.item()))
        history["ic_psi"].append(float(l_ic_psi.item()))
        history["ic_nu"].append(float(l_ic_nu.item()))

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Pretrain epoch {epoch:05d} | "
                f"IC(phi): {l_ic_phi.item():.4e} | "
                f"IC(mu): {l_ic_mu.item():.4e} | "
                f"IC(psi): {l_ic_psi.item():.4e} | "
                f"IC(nu): {l_ic_nu.item():.4e} | "
                f"IC total: {l_ic.item():.4e}"
            )

    return history


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    collocation: Dict[str, Tensor],
    phi_ic_true: Tensor,
    mu_ic_true: Tensor,
    psi_ic_true: Tensor,
    nu_ic_true: Tensor,
    n_epochs: int,
    pretrain_epochs: int,
    args: Namespace,
    device: torch.device,
    log_m_phi: Optional[Tensor] = None,
) -> Tuple[LossDict, Optional[LossDict]]:
    """
    Train a single PINN model with optional IC pretraining and Adam.

    Parameters
    ----------
    model:
        Mixed four-output PINN.
    optimizer:
        Adam optimizer.
    collocation:
        Dictionary of collocation/data points.
    phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true:
        Initial-condition targets.
    n_epochs:
        Number of Adam epochs after pretraining.
    pretrain_epochs:
        Number of IC-only pretraining epochs.
    args:
        Runtime configuration.
    device:
        Torch device.
    log_m_phi:
        Optional trainable logarithmic mobility for inverse runs.

    Returns
    -------
    train_losses, pretrain_losses:
        Loss histories for Adam training and optional pretraining.
    """
    train_losses = init_train_history()

    if pretrain_epochs > 0:
        print("\n" + "=" * 40)
        print("INITIAL-CONDITION PRETRAINING")
        print("=" * 40)

        pretrain_losses = pretrain_initial_condition(
            model=model,
            collocation=collocation,
            phi_ic_true=phi_ic_true,
            mu_ic_true=mu_ic_true,
            psi_ic_true=psi_ic_true,
            nu_ic_true=nu_ic_true,
            optimizer=optimizer,
            pretrain_epochs=pretrain_epochs,
        )
    else:
        pretrain_losses = None

    print("\n" + "=" * 40)
    print("ADAM TRAINING")
    print("=" * 40)

    for epoch in range(1, n_epochs + 1):
        if getattr(args, "adaptive_sampling", False) and epoch == args.adap_warmup_epochs:
            print(f"\nAdaptive residual-based resampling at epoch {epoch}...")

            x_pde_new, y_pde_new, t_pde_new = adaptive_resample_pde_points(
                model=model,
                args=args,
                device=device,
            )

            collocation["x_pde"] = x_pde_new
            collocation["y_pde"] = y_pde_new
            collocation["t_pde"] = t_pde_new

        m_phi_eff = get_effective_m_phi(args=args, log_m_phi=log_m_phi)

        epoch_losses = train_one_epoch(
            model=model,
            collocation=collocation,
            phi_ic_true=phi_ic_true,
            mu_ic_true=mu_ic_true,
            psi_ic_true=psi_ic_true,
            nu_ic_true=nu_ic_true,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            m_phi_eff=m_phi_eff,
        )

        append_epoch_losses(
            history=train_losses,
            epoch_losses=epoch_losses,
            inverse=getattr(args, "inverse_m_phi", False),
        )

    return train_losses, pretrain_losses


def train_lbfgs(
    model: nn.Module,
    collocation: Dict[str, Tensor],
    phi_ic_true: Tensor,
    mu_ic_true: Tensor,
    psi_ic_true: Tensor,
    nu_ic_true: Tensor,
    args: Namespace,
    log_m_phi: Optional[Tensor] = None,
) -> LossDict:
    """
    Perform optional L-BFGS refinement after Adam training.

    The same composite loss used during Adam is minimized. In inverse runs,
    log_m_phi is added to the optimized parameter list.
    """
    print("\n" + "=" * 40)
    print("L-BFGS REFINEMENT")
    print("=" * 40)

    model.train()

    params = list(model.parameters())

    if getattr(args, "inverse_m_phi", False):
        if log_m_phi is None:
            raise ValueError("inverse_m_phi=True but log_m_phi is None.")
        params += [log_m_phi]

    optimizer = torch.optim.LBFGS(
        params=params,
        lr=1.0,
        max_iter=args.lbfgs_iter,
        max_eval=args.lbfgs_iter * 2,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    x_pde = collocation["x_pde"]
    y_pde = collocation["y_pde"]
    t_pde = collocation["t_pde"]

    x_bc = collocation["x_bc"]
    y_bc = collocation["y_bc"]
    t_bc = collocation["t_bc"]
    normal = collocation["normal_bc"]

    x_ic = collocation["x_ic"]
    y_ic = collocation["y_ic"]
    t_ic = collocation["t_ic"]

    inverse = getattr(args, "inverse_m_phi", False)

    history = {
        "total": [],
        "pde": [],
        "bc": [],
        "ic": [],
        "data": [],
        "m_phi": [],
    }

    def closure():
        optimizer.zero_grad()

        m_phi_eff = get_effective_m_phi(args=args, log_m_phi=log_m_phi)

        l_pde, *_ = pde_loss(
            x=x_pde,
            y=y_pde,
            t=t_pde,
            model=model,
            args=args,
            m_phi_eff=m_phi_eff,
        )

        l_bc, *_ = bc_loss(
            model=model,
            x_bc=x_bc,
            y_bc=y_bc,
            t_bc=t_bc,
            normal=normal,
        )

        l_ic, *_ = ic_loss(
            model=model,
            x_ic=x_ic,
            y_ic=y_ic,
            t_ic=t_ic,
            phi_ic_true=phi_ic_true,
            mu_ic_true=mu_ic_true,
            psi_ic_true=psi_ic_true,
            nu_ic_true=nu_ic_true,
        )

        l_data = None

        if inverse:
            x_data = collocation["x_data"]
            y_data = collocation["y_data"]
            t_data = collocation["t_data"]

            l_data, *_ = data_loss_manufactured(
                model=model,
                x_data=x_data,
                y_data=y_data,
                t_data=t_data,
                args=args,
            )

        loss = compute_weighted_total_loss(
            args=args,
            l_pde=l_pde,
            l_bc=l_bc,
            l_ic=l_ic,
            l_data=l_data,
        )

        loss.backward()

        history["total"].append(float(loss.item()))
        history["pde"].append(float(l_pde.item()))
        history["bc"].append(float(l_bc.item()))
        history["ic"].append(float(l_ic.item()))

        if inverse:
            history["data"].append(float(l_data.item()))
            history["m_phi"].append(float(m_phi_eff.detach().cpu()))

        return loss

    optimizer.step(closure)

    print("\nL-BFGS refinement completed.")

    msg = (
        f"Final L-BFGS | "
        f"Total: {history['total'][-1]:.4e} | "
        f"PDE: {history['pde'][-1]:.4e} | "
        f"BC: {history['bc'][-1]:.4e} | "
        f"IC: {history['ic'][-1]:.4e}"
    )

    if inverse:
        msg += (
            f" | Data: {history['data'][-1]:.4e}"
            f" | m_phi: {history['m_phi'][-1]:.6e}"
        )

    print(msg)

    return history


def train_one_segment(
    segment_idx: int,
    ic_fn: Callable,
    args: Namespace,
    device: torch.device,
    log_m_phi: Optional[Tensor] = None,
):
    """
    Train one time-window segment of the 2D bio phase-field PINN.

    Each segment has its own network. For time-windowing, the initial condition
    function `ic_fn` may come either from the analytic initial condition or from
    the final prediction of the previous segment.

    Parameters
    ----------
    segment_idx:
        Index of the time segment.
    ic_fn:
        Function used to generate the segment initial condition.
    args:
        Runtime configuration.
    device:
        Torch device.
    log_m_phi:
        Optional trainable logarithmic mobility for inverse runs.

    Returns
    -------
    model:
        Trained segment model.
    train_losses:
        Adam training loss history.
    pretrain_losses:
        Optional initial-condition pretraining history.
    lbfgs_losses:
        Optional L-BFGS refinement history.
    """
    t_start = segment_idx * args.segment_length
    t_end = (segment_idx + 1) * args.segment_length

    print("\n" + "=" * 80)
    print(f"TRAINING SEGMENT {segment_idx}")
    print(f"Local/global time window: [{t_start:.6f}, {t_end:.6f}]")
    print("=" * 80)

    model = BioACCHPINN2d(
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
    ).to(device)

    model_params = list(model.parameters())

    if log_m_phi is not None:
        optimizer = Adam(
            [
                {"params": model_params, "lr": args.lr},
                {"params": [log_m_phi], "lr": args.m_phi_lr},
            ]
        )
    else:
        optimizer = Adam(model_params, lr=args.lr)

    collocation, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = generate_coll_points_and_ic(
        args=args,
        device=device,
        ic_fn=ic_fn,
    )

    start_time = time.time()

    train_losses, pretrain_losses = train_model(
        model=model,
        optimizer=optimizer,
        collocation=collocation,
        phi_ic_true=phi_ic_true,
        mu_ic_true=mu_ic_true,
        psi_ic_true=psi_ic_true,
        nu_ic_true=nu_ic_true,
        n_epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,
        args=args,
        device=device,
        log_m_phi=log_m_phi,
    )

    lbfgs_losses = None

    if args.lbfgs_iter > 0:
        lbfgs_losses = train_lbfgs(
            model=model,
            collocation=collocation,
            phi_ic_true=phi_ic_true,
            mu_ic_true=mu_ic_true,
            psi_ic_true=psi_ic_true,
            nu_ic_true=nu_ic_true,
            args=args,
            log_m_phi=log_m_phi,
        )

    elapsed = time.time() - start_time
    print(f"Segment {segment_idx} completed in {elapsed:.2f} s.")

    return model, train_losses, pretrain_losses, lbfgs_losses