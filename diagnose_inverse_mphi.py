"""
Least-squares diagnostics for manufactured inverse m_phi runs.

This script analyzes a trained manufactured inverse PINN and estimates the
mobility m_phi a posteriori from the learned fields using the phi-equation:

    phi_t + m_phi * mu - S_phi = 0

Given predicted fields phi_theta and mu_theta, the least-squares estimate is:

    m_phi^LS = <mu_theta, S_phi - phi_t_theta> / <mu_theta, mu_theta>

The same estimator is also evaluated on exact manufactured fields as a sanity
check. Since the manufactured source S_phi contains the true target mobility
m_phi explicitly, this script is a diagnostic tool for the manufactured leakage
setting, not a source-free blind inverse benchmark.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from manufactured_bio_2d import exact_fields_ms, manufactured_sources
from models import BioACCHPINN2d
from utils_bio_2d import grad


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run LS diagnostics for manufactured inverse m_phi experiments."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the manufactured inverse run.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="artifacts/manufactured",
        help="Root directory containing manufactured run artifacts.",
    )
    parser.add_argument(
        "--segment_idx",
        type=int,
        default=0,
        help="Time-window segment checkpoint to diagnose.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=20000,
        help="Number of random space-time points used for LS diagnostics.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional JSON output path. If not provided, results are saved under "
            "<root>/<run_name>/diagnostics/diagnose_inverse_mphi.json."
        ),
    )

    return parser.parse_args()


def load_args_from_config(run_dir: Path) -> argparse.Namespace:
    """
    Load the training configuration saved for a run.
    """
    config_path = run_dir / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return argparse.Namespace(**config)


def load_trained_model(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    segment_idx: int = 0,
) -> nn.Module:
    """
    Load one trained segment checkpoint.
    """
    checkpoint_path = run_dir / "weights" / f"segment_{segment_idx}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = BioACCHPINN2d(
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    return model


def sample_pde_points(
    args: argparse.Namespace,
    n_points: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample random space-time points for diagnostics.

    Notes
    -----
    This preserves the original script behavior: time is sampled in [0, args.tmax].
    For strictly local time-window diagnostics, make sure the diagnosed checkpoint
    was trained on the same time range or adapt this function accordingly.
    """
    x = args.lx * torch.rand(n_points, 1, device=device)
    y = args.ly * torch.rand(n_points, 1, device=device)
    t = args.tmax * torch.rand(n_points, 1, device=device)

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    return x, y, t


def least_squares_m_phi(mu: Tensor, rhs: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute the scalar least-squares estimate:

        m = <mu, rhs> / <mu, mu>

    where rhs = S_phi - phi_t.
    """
    numerator = torch.mean(mu * rhs)
    denominator = torch.mean(mu * mu) + eps

    return numerator / denominator


def rms(u: Tensor) -> Tensor:
    """
    Root mean square of a tensor.
    """
    return torch.sqrt(torch.mean(u**2))


def estimate_m_phi_ls(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    n_points: int = 20000,
) -> Dict[str, float]:
    """
    Estimate m_phi by least squares using predicted and exact manufactured fields.

    Diagnostics computed
    --------------------
    1. m_phi_LS_net_fields:
       Uses network phi_t_theta and mu_theta.

    2. m_phi_LS_exact_fields:
       Uses exact phi_t and exact mu. This should recover args.m_phi because
       the manufactured source was constructed with that value.

    3. m_phi_LS_exact_mu_net_phit:
       Uses exact mu but network phi_t_theta. This isolates errors in phi_t.

    4. m_phi_LS_net_mu_exact_phit:
       Uses network mu_theta but exact phi_t. This isolates errors in mu.
    """
    x, y, t = sample_pde_points(
        args=args,
        n_points=n_points,
        device=device,
    )

    # Network fields and time derivative.
    phi_pred, mu_pred, _, _ = model(x, y, t)
    phi_t_pred = grad(phi_pred, t)

    # Manufactured source built with the target value args.m_phi.
    source_phi, _ = manufactured_sources(x, y, t, args)

    mu_net = mu_pred.detach()
    rhs_net = (source_phi - phi_t_pred).detach()

    m_ls_net_fields = least_squares_m_phi(mu_net, rhs_net)

    residual_true_m = phi_t_pred.detach() + args.m_phi * mu_net - source_phi.detach()
    residual_ls_m = phi_t_pred.detach() + m_ls_net_fields * mu_net - source_phi.detach()

    # Exact manufactured fields: sanity check for the LS estimator.
    phi_exact, mu_exact, _, _ = exact_fields_ms(x, y, t, args)
    phi_t_exact = grad(phi_exact, t)

    mu_exact_detached = mu_exact.detach()
    rhs_exact = (source_phi - phi_t_exact).detach()

    m_ls_exact_fields = least_squares_m_phi(mu_exact_detached, rhs_exact)

    # Mixed diagnostics: isolate whether the LS discrepancy mostly comes from
    # the learned time derivative or from the learned auxiliary potential.
    m_ls_exact_mu_net_phit = least_squares_m_phi(mu_exact_detached, rhs_net)
    m_ls_net_mu_exact_phit = least_squares_m_phi(mu_net, rhs_exact)

    diagnostics = {
        "m_phi_true": float(args.m_phi),
        "m_phi_LS_net_fields": float(m_ls_net_fields.detach().cpu()),
        "m_phi_LS_exact_fields": float(m_ls_exact_fields.detach().cpu()),
        "m_phi_LS_exact_mu_net_phit": float(
            m_ls_exact_mu_net_phit.detach().cpu()
        ),
        "m_phi_LS_net_mu_exact_phit": float(
            m_ls_net_mu_exact_phit.detach().cpu()
        ),
        "rmse_R_phi_true_m": float(rms(residual_true_m).detach().cpu()),
        "rmse_R_phi_LS_m": float(rms(residual_ls_m).detach().cpu()),
        "mu_theta_rms": float(rms(mu_net).detach().cpu()),
        "phi_t_theta_rms": float(rms(phi_t_pred.detach()).detach().cpu()),
        "S_phi_rms": float(rms(source_phi.detach()).detach().cpu()),
        "n_points": int(n_points),
    }

    return diagnostics


def get_output_path(cli_args: argparse.Namespace, run_dir: Path) -> Path:
    """
    Return the JSON output path.
    """
    if cli_args.output_json is not None:
        output_path = Path(cli_args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    return diagnostics_dir / "diagnose_inverse_mphi.json"


def save_diagnostics_json(
    diagnostics: Dict[str, float],
    cli_args: argparse.Namespace,
    run_args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> Path:
    """
    Save LS diagnostics to JSON.
    """
    output = {
        "run_name": cli_args.run_name,
        "root": str(run_dir.parent),
        "device": str(device),
        "segment_idx": cli_args.segment_idx,
        "diagnostics_settings": {
            "n_points": cli_args.n_points,
        },
        "model_parameters": {
            "tmax": run_args.tmax,
            "segment_length": run_args.segment_length,
            "lx": run_args.lx,
            "ly": run_args.ly,
            "eps": run_args.eps,
            "m_phi": run_args.m_phi,
            "m0": run_args.m0,
            "lambda_surf": run_args.lambda_surf,
            "lambda_in": run_args.lambda_in,
            "lambda_out": run_args.lambda_out,
            "beta_in": run_args.beta_in,
            "beta_out": run_args.beta_out,
            "psi_in_eq": run_args.psi_in_eq,
            "psi_out_eq": run_args.psi_out_eq,
            "hidden_layers": run_args.hidden_layers,
            "hidden_dim": run_args.hidden_dim,
        },
        "diagnostics": diagnostics,
    }

    output_path = get_output_path(cli_args, run_dir)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    return output_path


def print_diagnostics(diagnostics: Dict[str, float]) -> None:
    """
    Print diagnostics in a compact readable format.
    """
    print("\nINVERSE m_phi LEAST-SQUARES DIAGNOSTICS")
    print("=" * 60)

    for key, value in diagnostics.items():
        if isinstance(value, int):
            print(f"{key:35s}: {value:d}")
        else:
            print(f"{key:35s}: {value:.8e}")


def main() -> None:
    """
    Entry point.
    """
    cli_args = parse_args()

    if cli_args.n_points <= 0:
        raise ValueError("n_points must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    run_dir = Path(cli_args.root) / cli_args.run_name

    run_args = load_args_from_config(run_dir)

    model = load_trained_model(
        run_dir=run_dir,
        args=run_args,
        device=device,
        segment_idx=cli_args.segment_idx,
    )

    diagnostics = estimate_m_phi_ls(
        model=model,
        args=run_args,
        device=device,
        n_points=cli_args.n_points,
    )

    print_diagnostics(diagnostics)

    output_path = save_diagnostics_json(
        diagnostics=diagnostics,
        cli_args=cli_args,
        run_args=run_args,
        run_dir=run_dir,
        device=device,
    )

    print(f"\nSaved diagnostics to: {output_path}")


if __name__ == "__main__":
    main()