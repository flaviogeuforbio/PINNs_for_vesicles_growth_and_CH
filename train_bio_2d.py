"""
Main training script for the 2D bio-inspired phase-field PINN.

This script trains the mixed four-output PINN used in the bio_phasefield_2d
branch:
    (x, y, t) -> (phi, mu, psi, nu)

Supported settings:
1. Forward minimal bio-inspired phase-field runs.
2. Manufactured-solution verification runs.
3. Manufactured inverse runs for the mobility parameter m_phi.

For inverse runs, the mobility is represented as:
    m_phi = exp(log_m_phi)
and optimized jointly with the network weights.

Outputs are saved under:
    artifacts/bio-minimal/<run_name>/
or:
    artifacts/manufactured/<run_name>/

depending on whether --manufactured is enabled.
"""

import argparse
import json
import math
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer_bio_2d import train_one_segment
from utils_bio_2d import make_ic_from_previous_model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a 2D bio-inspired phase-field PINN."
    )

    # -------------------------------------------------------------------------
    # Network and time-domain settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the current run.",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=4,
        help="Number of hidden layers in the PINN.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Number of neurons in each hidden layer.",
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
    parser.add_argument(
        "--recompute_pot",
        action="store_true",
        help=(
            "If enabled, mu and nu initial conditions for segments after the "
            "first one are recomputed from phi and psi instead of copied from "
            "the previous model predictions."
        ),
    )

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------
    parser.add_argument("--lx", type=float, default=1.0, help="Domain length in x.")
    parser.add_argument("--ly", type=float, default=1.0, help="Domain length in y.")

    # -------------------------------------------------------------------------
    # Physical parameters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="Diffuse-interface width parameter for phi.",
    )
    parser.add_argument(
        "--m_phi",
        type=float,
        default=1.0,
        help="Shape mobility parameter for phi.",
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
        help="Quadratic well coefficient for psi inside the vesicle.",
    )
    parser.add_argument(
        "--lambda_out",
        type=float,
        default=1.0,
        help="Quadratic well coefficient for psi outside the vesicle.",
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
        help="Equilibrium value of psi inside the vesicle.",
    )
    parser.add_argument(
        "--psi_out_eq",
        type=float,
        default=0.0,
        help="Equilibrium value of psi outside the vesicle.",
    )

    # -------------------------------------------------------------------------
    # Initial-condition parameters for the bio-inspired forward problem
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--psi_in_0",
        type=float,
        default=0.6,
        help="Initial value of psi inside the vesicle.",
    )
    parser.add_argument(
        "--psi_out_0",
        type=float,
        default=0.2,
        help="Initial value of psi outside the vesicle.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.28,
        help="Initial vesicle radius.",
    )
    parser.add_argument(
        "--x0",
        type=float,
        default=0.5,
        help="x coordinate of the initial vesicle center.",
    )
    parser.add_argument(
        "--y0",
        type=float,
        default=0.5,
        help="y coordinate of the initial vesicle center.",
    )

    # -------------------------------------------------------------------------
    # Loss weights
    # -------------------------------------------------------------------------
    parser.add_argument("--pde_weight", type=float, default=1.0, help="Global PDE loss weight.")
    parser.add_argument("--bc_weight", type=float, default=10.0, help="Boundary-condition loss weight.")
    parser.add_argument("--ic_weight", type=float, default=100.0, help="Initial-condition loss weight.")

    parser.add_argument("--pde_phi_w", type=float, default=1.0, help="Weight for R_phi.")
    parser.add_argument("--pde_mu_w", type=float, default=1.0, help="Weight for R_mu.")
    parser.add_argument("--pde_psi_w", type=float, default=1.0, help="Weight for R_psi.")
    parser.add_argument("--pde_nu_w", type=float, default=1.0, help="Weight for R_nu.")

    # -------------------------------------------------------------------------
    # Optimization settings
    # -------------------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=3000, help="Number of Adam epochs.")
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="Number of IC-only pretraining epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--lbfgs_iter",
        type=int,
        default=100,
        help="Number of L-BFGS refinement iterations. Use 0 to disable.",
    )

    # -------------------------------------------------------------------------
    # Collocation points
    # -------------------------------------------------------------------------
    parser.add_argument("--n_pde", type=int, default=10000, help="Number of PDE collocation points.")
    parser.add_argument("--n_bc", type=int, default=2000, help="Number of BC collocation points.")
    parser.add_argument("--n_ic", type=int, default=2000, help="Number of IC collocation points.")

    # -------------------------------------------------------------------------
    # Adaptive residual-based resampling
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--adaptive_sampling",
        action="store_true",
        help="Enable one-stage residual-based adaptive resampling of PDE points.",
    )
    parser.add_argument(
        "--adap_warmup_epochs",
        type=int,
        default=400,
        help="Epoch at which adaptive resampling is performed.",
    )
    parser.add_argument(
        "--n_candidates_resamp",
        type=int,
        default=30000,
        help="Number of candidate points generated for adaptive resampling.",
    )
    parser.add_argument(
        "--adaptive_frac",
        type=float,
        default=0.7,
        help="Fraction of PDE points selected adaptively during resampling.",
    )

    # -------------------------------------------------------------------------
    # Manufactured-solution settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--manufactured",
        action="store_true",
        help="Enable manufactured-solution forcing.",
    )
    parser.add_argument(
        "--ms_smooth",
        action="store_true",
        help="Use the smooth manufactured solution instead of the phase-field-like one.",
    )
    parser.add_argument(
        "--ms_R0",
        type=float,
        default=0.25,
        help="Initial radius for the phase-field-like manufactured solution.",
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
    # Manufactured inverse problem
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--inverse_m_phi",
        action="store_true",
        help="Enable inverse training of the shape mobility m_phi.",
    )
    parser.add_argument(
        "--m_phi_lr",
        type=float,
        default=5e-3,
        help="Learning rate for the logarithmic mobility parameter.",
    )
    parser.add_argument(
        "--m_phi_init",
        type=float,
        default=0.3,
        help="Initial guess for m_phi in inverse runs. The target is args.m_phi.",
    )
    parser.add_argument(
        "--data_weight",
        type=float,
        default=10.0,
        help="Data loss weight for manufactured inverse runs.",
    )
    parser.add_argument(
        "--n_data",
        type=int,
        default=5000,
        help="Number of manufactured data points for inverse runs.",
    )
    parser.add_argument(
        "--data_noise",
        type=float,
        default=0.0,
        help="Relative Gaussian noise level added to manufactured data targets.",
    )

    return parser.parse_args()


def get_output_directory(args: argparse.Namespace) -> Path:
    """
    Return the output directory for the current run.
    """
    if getattr(args, "manufactured", False):
        return Path("artifacts/manufactured") / args.run_name

    return Path("artifacts/bio-minimal") / args.run_name


def create_output_directories(out_dir: Path) -> Dict[str, Path]:
    """
    Create and return the output subdirectories used by this script.
    """
    directories = {
        "root": out_dir,
        "weights": out_dir / "weights",
        "figures": out_dir / "figures",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def save_run_config(args: argparse.Namespace, run_dir: Path) -> None:
    """
    Save the full run configuration to JSON.
    """
    config_path = run_dir / "run_config.json"
    config = vars(args).copy()

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Run configuration saved to: {config_path}")


def save_training_history(histories: List[Dict[str, Any]], run_dir: Path) -> None:
    """
    Save all segment loss histories to JSON.

    The histories contain only Python floats/lists, so they can be serialized
    directly.
    """
    history_path = run_dir / "training_history.json"

    with open(history_path, "w") as f:
        json.dump(histories, f, indent=4)

    print(f"Training history saved to: {history_path}")


def initialize_trainable_m_phi(args: argparse.Namespace, device: torch.device) -> Optional[torch.nn.Parameter]:
    """
    Initialize the trainable logarithmic mobility for inverse runs.

    Returns None for non-inverse runs.
    """
    if not getattr(args, "inverse_m_phi", False):
        return None

    if args.m_phi_init <= 0.0:
        raise ValueError("m_phi_init must be positive because log(m_phi_init) is used.")

    print("=" * 80)
    print("MANUFACTURED INVERSE PROBLEM: inferring m_phi")
    print(f"Target m_phi        = {args.m_phi}")
    print(f"Initial m_phi guess = {args.m_phi_init}")
    print("=" * 80)

    log_m_phi = torch.nn.Parameter(
        torch.tensor(
            np.log(args.m_phi_init),
            dtype=torch.float32,
            device=device,
        )
    )

    return log_m_phi


def assign_segment_noise_seed(
    args: argparse.Namespace,
    segment_idx: int,
    n_segments: int,
) -> None:
    """
    Assign a deterministic-per-segment noise seed for manufactured data.

    The seed is randomly generated once per segment at the beginning of the run
    and then reused at every epoch, so the noisy manufactured targets are fixed
    during training.
    """
    if getattr(args, "data_noise", 0.0) <= 0.0:
        args.data_noise_segment_seed = None
        return

    if not hasattr(args, "data_noise_segment_seeds"):
        args.data_noise_segment_seeds = [secrets.randbits(31) for _ in range(n_segments)]

    args.data_noise_segment_seed = args.data_noise_segment_seeds[segment_idx]

    print(
        f"Data noise enabled | "
        f"segment={segment_idx} | "
        f"noise={args.data_noise} | "
        f"seed={args.data_noise_segment_seed}"
    )


def plot_segment_losses(
    train_losses: Dict[str, List[float]],
    segment_idx: int,
    figures_dir: Path,
) -> None:
    """
    Save a semilog plot of the main training losses for one segment.
    """
    n_epochs = len(train_losses["total"])
    x_epochs = np.arange(1, n_epochs + 1)

    plt.figure(figsize=(8, 5))

    plt.semilogy(x_epochs, train_losses["pde_phi"], label="PDE phi")
    plt.semilogy(x_epochs, train_losses["pde_mu"], label="PDE mu")
    plt.semilogy(x_epochs, train_losses["pde_psi"], label="PDE psi")
    plt.semilogy(x_epochs, train_losses["pde_nu"], label="PDE nu")
    plt.semilogy(x_epochs, train_losses["bc"], label="BC")
    plt.semilogy(x_epochs, train_losses["ic"], label="IC")

    if len(train_losses.get("data", [])) == n_epochs and n_epochs > 0:
        plt.semilogy(x_epochs, train_losses["data"], label="Data")

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(f"Bio AC-CH 2D training losses (segment {segment_idx})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    output_path = figures_dir / f"segment_{segment_idx}_lossplot.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_m_phi_convergence(
    train_losses: Dict[str, List[float]],
    args: argparse.Namespace,
    segment_idx: int,
    figures_dir: Path,
) -> None:
    """
    Save the inferred mobility trajectory for inverse runs.
    """
    m_phi_values = train_losses.get("m_phi", [])

    if not getattr(args, "inverse_m_phi", False) or len(m_phi_values) == 0:
        return

    x_epochs = np.arange(1, len(m_phi_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x_epochs, m_phi_values, label=r"learned $m_\phi$")
    plt.axhline(args.m_phi, linestyle="--", color="black", label=r"target $m_\phi$")
    plt.xlabel("Epoch")
    plt.ylabel(r"$m_\phi$")
    plt.title(f"Manufactured inverse inference: segment {segment_idx}")
    plt.legend()
    plt.grid(True, alpha=0.4)

    output_path = figures_dir / f"segment_{segment_idx}_m_phi_convergence.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_segment_weights(model: torch.nn.Module, segment_idx: int, weights_dir: Path) -> None:
    """
    Save one segment model state_dict.
    """
    output_path = weights_dir / f"segment_{segment_idx}.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Saved segment {segment_idx} weights to: {output_path}")


def main() -> None:
    """
    Train all time-window segments for the selected run.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.segment_length <= 0.0:
        raise ValueError("segment_length must be positive.")

    n_segments = int(math.ceil(args.tmax / args.segment_length))

    print(f"T_max          = {args.tmax}")
    print(f"segment_length = {args.segment_length}")
    print(f"n_segments     = {n_segments}")

    out_dir = get_output_directory(args)
    dirs = create_output_directories(out_dir)

    log_m_phi = initialize_trainable_m_phi(args=args, device=device)

    # Create noise seeds once per run. They are stored in args and saved in
    # run_config.json after training.
    if getattr(args, "data_noise", 0.0) > 0.0:
        args.data_noise_segment_seeds = [secrets.randbits(31) for _ in range(n_segments)]
    else:
        args.data_noise_segment_seeds = []

    models = []
    all_histories = []
    ic_fn = None

    for segment_idx in range(n_segments):
        assign_segment_noise_seed(
            args=args,
            segment_idx=segment_idx,
            n_segments=n_segments,
        )

        model, train_losses, pretrain_losses, lbfgs_losses = train_one_segment(
            segment_idx=segment_idx,
            ic_fn=ic_fn,
            args=args,
            device=device,
            log_m_phi=log_m_phi,
        )

        models.append(model)

        segment_history = {
            "segment_idx": segment_idx,
            "train_losses": train_losses,
            "pretrain_losses": pretrain_losses,
            "lbfgs_losses": lbfgs_losses,
        }

        all_histories.append(segment_history)

        save_segment_weights(
            model=model,
            segment_idx=segment_idx,
            weights_dir=dirs["weights"],
        )

        plot_segment_losses(
            train_losses=train_losses,
            segment_idx=segment_idx,
            figures_dir=dirs["figures"],
        )

        plot_m_phi_convergence(
            train_losses=train_losses,
            args=args,
            segment_idx=segment_idx,
            figures_dir=dirs["figures"],
        )

        # The next segment starts from the final prediction of the current model.
        ic_fn = make_ic_from_previous_model(
            previous_model=model,
            segment_length=args.segment_length,
            device=device,
            args=args,
            recompute_potentials=args.recompute_pot,
        )

    save_run_config(args=args, run_dir=dirs["root"])
    save_training_history(histories=all_histories, run_dir=dirs["root"])

    if getattr(args, "inverse_m_phi", False) and log_m_phi is not None:
        final_m_phi = float(torch.exp(log_m_phi).detach().cpu())
        print(f"Final inferred m_phi: {final_m_phi:.8e}")

    print("Training completed.")


if __name__ == "__main__":
    main()