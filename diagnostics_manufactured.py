"""
Diagnostics for manufactured-solution experiments in the 2D bio PINN branch.

This script evaluates trained time-windowed PINN models against the exact
manufactured solution and computes:

1. Field reconstruction errors:
       MSE(phi, mu, psi, nu)
       relative L2(phi, mu, psi, nu)

2. Optional source-term diagnostics:
       RMSE(phi_t), RMSE(m_phi * mu), RMSE(S_phi)
       RMSE(psi_t), RMSE(div J), RMSE(S_psi)

   together with normalized source magnitudes:
       rho_phi = ||S_phi|| / (||phi_t|| + ||m_phi mu||)
       rho_psi = ||S_psi|| / (||psi_t|| + ||div J||)

3. Diagnostic plots and optional JSON export.

Outputs are saved under:
    artifacts/manufactured/<run_name>/figures/
and, if enabled or by default, under:
    artifacts/manufactured/<run_name>/diagnostics/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from manufactured_bio_2d import exact_fields_ms
from utils_bio_2d import grad, load_models, predict_windowed


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute diagnostics for manufactured 2D bio PINN runs."
    )

    # -------------------------------------------------------------------------
    # Run and time-window settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the manufactured run to diagnose.",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=1.0,
        help="Final simulation time.",
    )
    parser.add_argument(
        "--segment_length",
        type=float,
        default=0.5,
        help="Length of each time-window segment.",
    )

    # -------------------------------------------------------------------------
    # Domain settings
    # -------------------------------------------------------------------------
    parser.add_argument("--lx", type=float, default=1.0, help="Domain length in x.")
    parser.add_argument("--ly", type=float, default=1.0, help="Domain length in y.")

    # -------------------------------------------------------------------------
    # Free-energy and model parameters
    # These are overwritten from run_config.json if available.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="Diffuse-interface width parameter.",
    )
    parser.add_argument(
        "--m_phi",
        type=float,
        default=1.0,
        help="Shape mobility parameter.",
    )
    parser.add_argument(
        "--m0",
        type=float,
        default=0.5,
        help="Parameter in the psi mobility function.",
    )
    parser.add_argument(
        "--lambda_surf",
        type=float,
        default=1.0,
        help="Surface free-energy weight.",
    )
    parser.add_argument(
        "--lambda_in",
        type=float,
        default=1.0,
        help="Quadratic coefficient for inner osmotic free energy.",
    )
    parser.add_argument(
        "--lambda_out",
        type=float,
        default=1.0,
        help="Quadratic coefficient for outer osmotic free energy.",
    )
    parser.add_argument(
        "--beta_in",
        type=float,
        default=0.0,
        help="Additive constant in the inner osmotic free-energy density.",
    )
    parser.add_argument(
        "--beta_out",
        type=float,
        default=0.0,
        help="Additive constant in the outer osmotic free-energy density.",
    )
    parser.add_argument(
        "--psi_in_eq",
        type=float,
        default=1.0,
        help="Equilibrium psi value inside the vesicle.",
    )
    parser.add_argument(
        "--psi_out_eq",
        type=float,
        default=0.0,
        help="Equilibrium psi value outside the vesicle.",
    )

    # -------------------------------------------------------------------------
    # Network and diagnostic-grid settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=4,
        help="Number of PINN hidden layers.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Number of neurons in each hidden layer.",
    )
    parser.add_argument(
        "--n_grid",
        type=int,
        default=128,
        help="Grid size used for diagnostics: n_grid x n_grid.",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=11,
        help="Number of diagnostic time slices between 0 and tmax.",
    )

    # -------------------------------------------------------------------------
    # Manufactured-solution parameters
    # These are overwritten from run_config.json if available.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--ms_smooth",
        action="store_true",
        help="Use the smooth manufactured solution.",
    )
    parser.add_argument(
        "--ms_R0",
        type=float,
        default=0.25,
        help="Initial radius for the phase-field-like manufactured solution.",
    )
    parser.add_argument(
        "--x0",
        type=float,
        default=0.5,
        help="x coordinate of the vesicle center.",
    )
    parser.add_argument(
        "--y0",
        type=float,
        default=0.5,
        help="y coordinate of the vesicle center.",
    )
    parser.add_argument(
        "--ms_alpha_R",
        type=float,
        default=0.3,
        help="Radius growth rate for the phase-field-like manufactured solution.",
    )
    parser.add_argument(
        "--ms_psi_in0",
        type=float,
        default=0.3,
        help="Initial inner psi value for the phase-field-like manufactured solution.",
    )
    parser.add_argument(
        "--ms_psi_out0",
        type=float,
        default=0.8,
        help="Initial outer psi value for the phase-field-like manufactured solution.",
    )
    parser.add_argument(
        "--ms_beta_in",
        type=float,
        default=0.2,
        help="Growth rate of inner psi for the phase-field-like manufactured solution.",
    )
    parser.add_argument(
        "--ms_beta_out",
        type=float,
        default=0.0,
        help="Growth rate of outer psi for the phase-field-like manufactured solution.",
    )

    # -------------------------------------------------------------------------
    # Optional diagnostics and output
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--source_diagnostics",
        action="store_true",
        help="Compute diagnostics for the manufactured source terms.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output JSON path. If not provided, diagnostics are saved "
            "under artifacts/manufactured/<run_name>/diagnostics/."
        ),
    )

    return parser.parse_args()


def update_args_from_run_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Load training parameters from run_config.json, if available.

    Only keys already present in the diagnostic CLI namespace are overwritten.
    This keeps diagnostic-only options such as n_grid, n_times, output_json and
    source_diagnostics under direct CLI control unless they are explicitly stored
    in the run configuration.
    """
    config_path = Path("artifacts/manufactured") / args.run_name / "run_config.json"

    if not config_path.exists():
        print(f"WARNING: run_config.json not found at {config_path}")
        return args

    with open(config_path, "r") as f:
        config = json.load(f)

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    print(f"Loaded run configuration from: {config_path}")
    print("Active diagnostics parameters:")
    print(f"  tmax           = {args.tmax}")
    print(f"  segment_length = {args.segment_length}")
    print(f"  eps            = {args.eps}")
    print(f"  m_phi          = {args.m_phi}")
    print(f"  m0             = {args.m0}")
    print(f"  psi_in_eq      = {args.psi_in_eq}")
    print(f"  psi_out_eq     = {args.psi_out_eq}")
    print(f"  hidden_layers  = {args.hidden_layers}")
    print(f"  hidden_dim     = {args.hidden_dim}")

    return args


# =============================================================================
# Error metrics
# =============================================================================

def mse(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean squared error.
    """
    return torch.mean((pred - target) ** 2)


def rmse(u: Tensor) -> Tensor:
    """
    Root mean square of a tensor.
    """
    return torch.sqrt(torch.mean(u**2))


def relative_l2(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Relative L2 error based on RMS norms.
    """
    numerator = torch.sqrt(torch.mean((pred - target) ** 2))
    denominator = torch.sqrt(torch.mean(target**2)) + eps

    return numerator / denominator


# =============================================================================
# Manufactured field diagnostics
# =============================================================================

def evaluate_err_diagnostics(
    models: List[torch.nn.Module],
    times: Tensor,
    n_grid: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, List[float]], Tensor, Tensor]:
    """
    Evaluate pointwise field errors against the manufactured exact solution.

    Parameters
    ----------
    models:
        Trained time-windowed PINN ensemble.
    times:
        Global diagnostic times.
    n_grid:
        Number of grid points per spatial direction.
    args:
        Runtime configuration.
    device:
        Torch device.

    Returns
    -------
    results:
        Time series of MSE and relative L2 errors.
    x_flat, y_flat:
        Flattened diagnostic grid, reused by source diagnostics.
    """
    results = {
        "times": [],
        "mse_phi": [],
        "mse_mu": [],
        "mse_psi": [],
        "mse_nu": [],
        "rel_l2_phi": [],
        "rel_l2_mu": [],
        "rel_l2_psi": [],
        "rel_l2_nu": [],
    }

    x_lin = torch.linspace(0.0, args.lx, n_grid, device=device)
    y_lin = torch.linspace(0.0, args.ly, n_grid, device=device)

    x_mesh, y_mesh = torch.meshgrid(x_lin, y_lin, indexing="ij")

    x_flat = x_mesh.reshape(-1, 1)
    y_flat = y_mesh.reshape(-1, 1)

    for time_value in times:
        time_float = float(time_value.detach().cpu())

        x = x_flat.clone().detach().requires_grad_(True)
        y = y_flat.clone().detach().requires_grad_(True)
        t = torch.full_like(x, time_float, device=device).requires_grad_(True)

        phi_pred, mu_pred, psi_pred, nu_pred = predict_windowed(
            models=models,
            x=x,
            y=y,
            t_global=time_float,
            segment_length=args.segment_length,
            device=device,
        )

        phi_exact, mu_exact, psi_exact, nu_exact = exact_fields_ms(
            x=x,
            y=y,
            t=t,
            args=args,
        )

        results["times"].append(time_float)

        results["mse_phi"].append(float(mse(phi_pred, phi_exact).detach().cpu()))
        results["mse_mu"].append(float(mse(mu_pred, mu_exact).detach().cpu()))
        results["mse_psi"].append(float(mse(psi_pred, psi_exact).detach().cpu()))
        results["mse_nu"].append(float(mse(nu_pred, nu_exact).detach().cpu()))

        results["rel_l2_phi"].append(
            float(relative_l2(phi_pred, phi_exact).detach().cpu())
        )
        results["rel_l2_mu"].append(
            float(relative_l2(mu_pred, mu_exact).detach().cpu())
        )
        results["rel_l2_psi"].append(
            float(relative_l2(psi_pred, psi_exact).detach().cpu())
        )
        results["rel_l2_nu"].append(
            float(relative_l2(nu_pred, nu_exact).detach().cpu())
        )

    return results, x_flat, y_flat


def evaluate_source_diagnostics(
    x_flat: Tensor,
    y_flat: Tensor,
    times: Tensor,
    args: argparse.Namespace,
) -> Dict[str, List[float]]:
    """
    Evaluate manufactured source-term diagnostics.

    This measures how strong the forcing terms are relative to the natural
    unforced PDE contributions.

    For phi:
        S_phi = phi_t + m_phi * mu

    For psi:
        S_psi = psi_t + div J,
        J = -M_psi(phi) grad(nu)
    """
    results = {
        "times": [],
        "rmse_phi_t": [],
        "rmse_m_phi_mu": [],
        "rmse_S_phi": [],
        "rmse_psi_t": [],
        "rmse_div_J": [],
        "rmse_S_psi": [],
        "rho_phi": [],
        "rho_psi": [],
    }

    for time_value in times:
        time_float = float(time_value.detach().cpu())

        x = x_flat.clone().detach().requires_grad_(True)
        y = y_flat.clone().detach().requires_grad_(True)
        t = torch.full_like(x, time_float, device=x.device).requires_grad_(True)

        phi_exact, mu_exact, psi_exact, nu_exact = exact_fields_ms(
            x=x,
            y=y,
            t=t,
            args=args,
        )

        phi_t = grad(phi_exact, t)
        psi_t = grad(psi_exact, t)

        nu_x = grad(nu_exact, x)
        nu_y = grad(nu_exact, y)

        mobility_psi = 1.0 - args.m0 * ((phi_exact**2 - 1.0) ** 2)

        current_x = -mobility_psi * nu_x
        current_y = -mobility_psi * nu_y

        div_current = grad(current_x, x) + grad(current_y, y)

        source_phi = phi_t + args.m_phi * mu_exact
        source_psi = psi_t + div_current

        eps = 1e-12

        rmse_phi_t = rmse(phi_t)
        rmse_m_phi_mu = rmse(args.m_phi * mu_exact)
        rmse_source_phi = rmse(source_phi)

        rmse_psi_t = rmse(psi_t)
        rmse_div_current = rmse(div_current)
        rmse_source_psi = rmse(source_psi)

        rho_phi = rmse_source_phi / (rmse_phi_t + rmse_m_phi_mu + eps)
        rho_psi = rmse_source_psi / (rmse_psi_t + rmse_div_current + eps)

        results["times"].append(time_float)
        results["rmse_phi_t"].append(float(rmse_phi_t.detach().cpu()))
        results["rmse_m_phi_mu"].append(float(rmse_m_phi_mu.detach().cpu()))
        results["rmse_S_phi"].append(float(rmse_source_phi.detach().cpu()))

        results["rmse_psi_t"].append(float(rmse_psi_t.detach().cpu()))
        results["rmse_div_J"].append(float(rmse_div_current.detach().cpu()))
        results["rmse_S_psi"].append(float(rmse_source_psi.detach().cpu()))

        results["rho_phi"].append(float(rho_phi.detach().cpu()))
        results["rho_psi"].append(float(rho_psi.detach().cpu()))

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_errors(results: Dict[str, List[float]], out_dir: Path) -> None:
    """
    Save line plots for relative L2 errors and MSEs.
    """
    times = results["times"]

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["rel_l2_phi"], label=r"$\phi$")
    plt.semilogy(times, results["rel_l2_psi"], label=r"$\psi$")
    plt.semilogy(times, results["rel_l2_mu"], label=r"$\mu$")
    plt.semilogy(times, results["rel_l2_nu"], label=r"$\nu$")
    plt.xlabel("Time")
    plt.ylabel("Relative L2 error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "relative_l2_errors.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["mse_phi"], label=r"$\phi$")
    plt.semilogy(times, results["mse_psi"], label=r"$\psi$")
    plt.semilogy(times, results["mse_mu"], label=r"$\mu$")
    plt.semilogy(times, results["mse_nu"], label=r"$\nu$")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "mse_errors.png", dpi=200)
    plt.close()


def plot_source_diagnostics(results: Dict[str, List[float]], out_dir: Path) -> None:
    """
    Save source-strength time-series plots.
    """
    times = results["times"]

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["rho_phi"], label=r"$\rho_{\phi}$")
    plt.semilogy(times, results["rho_psi"], label=r"$\rho_{\psi}$")
    plt.xlabel("Time")
    plt.ylabel("Normalized source magnitude")
    plt.title(r"Normalized manufactured source magnitude")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "normalized_source_magnitude.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["rmse_S_phi"], label=r"$S_{\phi}$")
    plt.semilogy(times, results["rmse_phi_t"], label=r"$\phi_t$")
    plt.semilogy(times, results["rmse_m_phi_mu"], label=r"$m_\phi\mu$")
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title(r"Manufactured $\phi$-equation source diagnostics")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "source_phi_components.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["rmse_S_psi"], label=r"$S_{\psi}$")
    plt.semilogy(times, results["rmse_psi_t"], label=r"$\psi_t$")
    plt.semilogy(times, results["rmse_div_J"], label=r"$\nabla\cdot J$")
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title(r"Manufactured $\psi$-equation source diagnostics")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "source_psi_components.png", dpi=200)
    plt.close()


def plot_source_maps(
    x_flat: Tensor,
    y_flat: Tensor,
    n_grid: int,
    args: argparse.Namespace,
    out_dir: Path,
    time_value: Optional[float] = None,
) -> None:
    """
    Save spatial maps of manufactured source magnitudes and normalized sources.
    """
    if time_value is None:
        time_value = args.tmax

    device = x_flat.device

    x = x_flat.clone().detach().requires_grad_(True)
    y = y_flat.clone().detach().requires_grad_(True)
    t = torch.full_like(x, float(time_value), device=device).requires_grad_(True)

    phi_exact, mu_exact, psi_exact, nu_exact = exact_fields_ms(
        x=x,
        y=y,
        t=t,
        args=args,
    )

    phi_t = grad(phi_exact, t)
    psi_t = grad(psi_exact, t)

    nu_x = grad(nu_exact, x)
    nu_y = grad(nu_exact, y)

    mobility_psi = 1.0 - args.m0 * ((phi_exact**2 - 1.0) ** 2)

    current_x = -mobility_psi * nu_x
    current_y = -mobility_psi * nu_y

    div_current = grad(current_x, x) + grad(current_y, y)

    source_phi = phi_t + args.m_phi * mu_exact
    source_psi = psi_t + div_current

    eps = 1e-12

    rho_phi_pointwise = torch.abs(source_phi) / (
        torch.abs(phi_t) + torch.abs(args.m_phi * mu_exact) + eps
    )

    rho_psi_pointwise = torch.abs(source_psi) / (
        torch.abs(psi_t) + torch.abs(div_current) + eps
    )

    def to_grid(u: Tensor) -> Tensor:
        return u.detach().cpu().reshape(n_grid, n_grid).T

    maps = {
        r"$\log_{10}(|S_\phi|)$": torch.log10(torch.abs(source_phi) + eps),
        r"$\log_{10}(|S_\psi|)$": torch.log10(torch.abs(source_psi) + eps),
        r"$\rho_\phi(x,y)$": rho_phi_pointwise,
        r"$\rho_\psi(x,y)$": rho_psi_pointwise,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (title, field) in zip(axes.ravel(), maps.items()):
        image = ax.imshow(
            to_grid(field),
            origin="lower",
            extent=[0.0, args.lx, 0.0, args.ly],
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(image, ax=ax)

    fig.suptitle(f"Manufactured source maps at t={float(time_value):.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / f"source_maps_t{float(time_value):.3f}.png", dpi=200)
    plt.close()


# =============================================================================
# JSON export
# =============================================================================

def get_diagnostics_output_path(args: argparse.Namespace) -> Path:
    """
    Return the JSON output path for manufactured diagnostics.
    """
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    diagnostics_dir = Path("artifacts/manufactured") / args.run_name / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    return diagnostics_dir / "diagnostics_manufactured.json"


def build_error_summary(error_diagnostics: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Build a compact summary of manufactured reconstruction errors.
    """
    summary = {}

    for key, values in error_diagnostics.items():
        if key == "times":
            continue

        summary[key] = {
            "initial": values[0],
            "final": values[-1],
            "min": min(values),
            "max": max(values),
        }

    return summary


def build_source_summary(
    source_diagnostics: Optional[Dict[str, List[float]]],
) -> Optional[Dict[str, Any]]:
    """
    Build a compact summary of source diagnostics.
    """
    if source_diagnostics is None:
        return None

    summary = {}

    for key, values in source_diagnostics.items():
        if key == "times":
            continue

        summary[key] = {
            "initial": values[0],
            "final": values[-1],
            "min": min(values),
            "max": max(values),
        }

    return summary


def save_diagnostics_json(
    args: argparse.Namespace,
    device: torch.device,
    error_diagnostics: Dict[str, List[float]],
    source_diagnostics: Optional[Dict[str, List[float]]] = None,
) -> Path:
    """
    Save manufactured diagnostics to JSON.
    """
    output = {
        "run_name": args.run_name,
        "device": str(device),
        "diagnostics_settings": {
            "tmax": args.tmax,
            "segment_length": args.segment_length,
            "lx": args.lx,
            "ly": args.ly,
            "n_grid": args.n_grid,
            "n_times": len(error_diagnostics["times"]),
            "source_diagnostics": getattr(args, "source_diagnostics", False),
        },
        "model_parameters": {
            "eps": args.eps,
            "m_phi": args.m_phi,
            "m0": args.m0,
            "lambda_surf": args.lambda_surf,
            "lambda_in": args.lambda_in,
            "lambda_out": args.lambda_out,
            "beta_in": args.beta_in,
            "beta_out": args.beta_out,
            "psi_in_eq": args.psi_in_eq,
            "psi_out_eq": args.psi_out_eq,
            "hidden_layers": args.hidden_layers,
            "hidden_dim": args.hidden_dim,
        },
        "manufactured_parameters": {
            "ms_smooth": getattr(args, "ms_smooth", False),
            "ms_R0": getattr(args, "ms_R0", None),
            "x0": getattr(args, "x0", None),
            "y0": getattr(args, "y0", None),
            "ms_alpha_R": getattr(args, "ms_alpha_R", None),
            "ms_psi_in0": getattr(args, "ms_psi_in0", None),
            "ms_psi_out0": getattr(args, "ms_psi_out0", None),
            "ms_beta_in": getattr(args, "ms_beta_in", None),
            "ms_beta_out": getattr(args, "ms_beta_out", None),
        },
        "summary": {
            "errors": build_error_summary(error_diagnostics),
            "sources": build_source_summary(source_diagnostics),
        },
        "error_diagnostics": error_diagnostics,
        "source_diagnostics": source_diagnostics,
    }

    output_path = get_diagnostics_output_path(args)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    return output_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Load trained manufactured models, compute diagnostics and save outputs.
    """
    args = parse_args()
    args = update_args_from_run_config(args)

    if args.n_grid <= 1:
        raise ValueError("n_grid must be greater than 1.")
    if args.n_times <= 1:
        raise ValueError("n_times must be greater than 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    check_dir = Path("artifacts/manufactured") / args.run_name / "weights"

    if not check_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")

    out_dir = Path("artifacts/manufactured") / args.run_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(
        check_dir=check_dir,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    diagnostics_times = torch.linspace(0.0, args.tmax, args.n_times)

    error_diagnostics, x_flat, y_flat = evaluate_err_diagnostics(
        models=models,
        times=diagnostics_times,
        n_grid=args.n_grid,
        args=args,
        device=device,
    )

    source_diagnostics = None

    if getattr(args, "source_diagnostics", False):
        source_diagnostics = evaluate_source_diagnostics(
            x_flat=x_flat,
            y_flat=y_flat,
            times=diagnostics_times,
            args=args,
        )

        plot_source_diagnostics(
            results=source_diagnostics,
            out_dir=out_dir,
        )

        plot_source_maps(
            x_flat=x_flat,
            y_flat=y_flat,
            n_grid=args.n_grid,
            args=args,
            out_dir=out_dir,
            time_value=args.tmax,
        )

    plot_errors(
        results=error_diagnostics,
        out_dir=out_dir,
    )

    json_path = save_diagnostics_json(
        args=args,
        device=device,
        error_diagnostics=error_diagnostics,
        source_diagnostics=source_diagnostics,
    )

    print(f"Saved diagnostic plots to: {out_dir}")
    print(f"Saved diagnostic JSON to: {json_path}")


if __name__ == "__main__":
    main()