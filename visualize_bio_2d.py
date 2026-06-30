"""
Visualization script for the 2D bio-inspired phase-field PINN.

This script loads the trained time-windowed PINN ensemble and generates GIF
animations on a regular 2D grid.

Two visualization modes are supported:

1. Bio-minimal forward runs
   The script animates the predicted physical fields:
       phi(x, y, t), psi(x, y, t)

2. Manufactured-solution runs
   The script animates pointwise squared-error maps:
       (phi_theta - phi_exact)^2, (psi_theta - psi_exact)^2

Outputs are saved under:
    artifacts/bio-minimal/<run_name>/animations/
or:
    artifacts/manufactured/<run_name>/animations/
depending on whether --manufactured is enabled.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from manufactured_bio_2d import exact_phi_psi
from utils_bio_2d import load_models, predict_windowed


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate 2D animations for trained bio_phasefield_2d PINNs."
    )

    # -------------------------------------------------------------------------
    # Run and checkpoint settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run to visualize.",
    )
    parser.add_argument(
        "--manufactured",
        action="store_true",
        help="Visualize manufactured-solution pointwise errors.",
    )
    parser.add_argument(
        "--gif_name",
        type=str,
        default="bio_ACCH_2d.gif",
        help="Output GIF file name.",
    )

    # -------------------------------------------------------------------------
    # Domain and time-window settings
    # These values are overwritten from run_config.json if available.
    # -------------------------------------------------------------------------
    parser.add_argument("--tmax", type=float, default=1.0, help="Final time.")
    parser.add_argument(
        "--segment_length",
        type=float,
        default=0.5,
        help="Length of each time-window segment.",
    )
    parser.add_argument("--lx", type=float, default=1.0, help="Domain length in x.")
    parser.add_argument("--ly", type=float, default=1.0, help="Domain length in y.")

    # -------------------------------------------------------------------------
    # Network architecture
    # These values are overwritten from run_config.json if available.
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Plot resolution
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--n_x_plot",
        type=int,
        default=128,
        help="Number of grid points in x for visualization.",
    )
    parser.add_argument(
        "--n_y_plot",
        type=int,
        default=128,
        help="Number of grid points in y for visualization.",
    )
    parser.add_argument(
        "--n_t_plot",
        type=int,
        default=80,
        help="Number of time frames in the animation.",
    )

    # -------------------------------------------------------------------------
    # Manufactured-solution parameters
    # These are overwritten from run_config.json if available.
    # -------------------------------------------------------------------------
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
    # Error-map visualization settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--error_scale",
        type=str,
        default="linear",
        choices=["linear", "log"],
        help="Scale used for manufactured error maps.",
    )
    parser.add_argument(
        "--error_eps",
        type=float,
        default=1e-12,
        help="Small offset used when plotting log-error maps.",
    )

    return parser.parse_args()


def get_artifact_root(args: argparse.Namespace) -> Path:
    """
    Return the root artifact directory for the selected run type.
    """
    if getattr(args, "manufactured", False):
        return Path("artifacts/manufactured")

    return Path("artifacts/bio-minimal")


def update_args_from_run_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Load run parameters from run_config.json, if available.

    This ensures that visualization uses the same physical parameters, network
    architecture and time-window settings used during training. CLI-only plotting
    parameters such as gif_name, n_x_plot, n_y_plot, n_t_plot and error_scale are
    preserved unless the run configuration explicitly contains those keys.
    """
    root = get_artifact_root(args)
    config_path = root / args.run_name / "run_config.json"

    if not config_path.exists():
        print(f"WARNING: run_config.json not found at {config_path}")
        return args

    with open(config_path, "r") as f:
        config = json.load(f)

    for key, value in config.items():
        setattr(args, key, value)

    print(f"Loaded run configuration from: {config_path}")

    return args


def get_checkpoint_dir(args: argparse.Namespace) -> Path:
    """
    Return the directory containing trained segment checkpoints.
    """
    return get_artifact_root(args) / args.run_name / "weights"


def get_animation_dir(args: argparse.Namespace) -> Path:
    """
    Return and create the directory where animations are saved.
    """
    save_dir = get_artifact_root(args) / args.run_name / "animations"
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir


def build_plot_grid(
    lx: float,
    ly: float,
    tmax: float,
    n_x_plot: int,
    n_y_plot: int,
    n_t_plot: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a regular space-time grid for visualization.
    """
    x_grid = torch.linspace(0.0, lx, n_x_plot)
    y_grid = torch.linspace(0.0, ly, n_y_plot)
    t_grid = torch.linspace(0.0, tmax, n_t_plot)

    x_mesh, y_mesh = torch.meshgrid(x_grid, y_grid, indexing="xy")

    return x_grid, y_grid, t_grid, x_mesh, y_mesh


def compute_pts_to_visualize_tw(
    check_dir: Path,
    lx: float,
    ly: float,
    tmax: float,
    segment_length: float,
    hidden_layers: int,
    hidden_dim: int,
    device: torch.device,
    n_x_plot: int,
    n_y_plot: int,
    n_t_plot: int,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate phi and psi on a regular grid using the time-windowed PINN ensemble.

    Returns
    -------
    phi_results, psi_results:
        Arrays with shape (n_t_plot, n_y_plot, n_x_plot).
    x_grid, y_grid, t_grid:
        Grid coordinates.
    """
    models = load_models(
        check_dir=check_dir,
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        device=device,
    )

    x_grid, y_grid, t_grid, x_mesh, y_mesh = build_plot_grid(
        lx=lx,
        ly=ly,
        tmax=tmax,
        n_x_plot=n_x_plot,
        n_y_plot=n_y_plot,
        n_t_plot=n_t_plot,
    )

    x_flat = x_mesh.reshape(-1, 1).to(device)
    y_flat = y_mesh.reshape(-1, 1).to(device)

    phi_results = np.zeros((n_t_plot, n_y_plot, n_x_plot))
    psi_results = np.zeros((n_t_plot, n_y_plot, n_x_plot))

    with torch.no_grad():
        for frame_idx, t_val in enumerate(t_grid):
            phi_pred, _, psi_pred, _ = predict_windowed(
                models=models,
                x=x_flat,
                y=y_flat,
                t_global=float(t_val),
                segment_length=segment_length,
                device=device,
            )

            phi_results[frame_idx] = (
                phi_pred.detach().cpu().numpy().reshape(n_y_plot, n_x_plot)
            )
            psi_results[frame_idx] = (
                psi_pred.detach().cpu().numpy().reshape(n_y_plot, n_x_plot)
            )

    return phi_results, psi_results, x_grid, y_grid, t_grid


def compute_error_pts_manufactured_tw(
    check_dir: Path,
    lx: float,
    ly: float,
    tmax: float,
    segment_length: float,
    hidden_layers: int,
    hidden_dim: int,
    device: torch.device,
    n_x_plot: int,
    n_y_plot: int,
    n_t_plot: int,
    args: argparse.Namespace,
    error_scale: str = "linear",
    error_eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute manufactured-solution pointwise error maps on a regular grid.

    The plotted quantity is either:
        squared error
    or:
        log10(squared error + error_eps)

    depending on `error_scale`.
    """
    models = load_models(
        check_dir=check_dir,
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        device=device,
    )

    x_grid, y_grid, t_grid, x_mesh, y_mesh = build_plot_grid(
        lx=lx,
        ly=ly,
        tmax=tmax,
        n_x_plot=n_x_plot,
        n_y_plot=n_y_plot,
        n_t_plot=n_t_plot,
    )

    x_flat = x_mesh.reshape(-1, 1).to(device)
    y_flat = y_mesh.reshape(-1, 1).to(device)

    phi_err_results = np.zeros((n_t_plot, n_y_plot, n_x_plot))
    psi_err_results = np.zeros((n_t_plot, n_y_plot, n_x_plot))

    for frame_idx, t_val in enumerate(t_grid):
        t_float = float(t_val)

        with torch.no_grad():
            phi_pred, _, psi_pred, _ = predict_windowed(
                models=models,
                x=x_flat,
                y=y_flat,
                t_global=t_float,
                segment_length=segment_length,
                device=device,
            )

        # The exact manufactured fields are evaluated separately so that this
        # function remains independent of the trained model graph.
        x_exact = x_flat.detach().clone().requires_grad_(True)
        y_exact = y_flat.detach().clone().requires_grad_(True)
        t_exact = torch.full_like(x_exact, t_float).requires_grad_(True)

        phi_exact, psi_exact = exact_phi_psi(
            x=x_exact,
            y=y_exact,
            t=t_exact,
            args=args,
        )

        phi_sqerr = (phi_pred.detach() - phi_exact.detach()) ** 2
        psi_sqerr = (psi_pred.detach() - psi_exact.detach()) ** 2

        if error_scale == "log":
            phi_plot = torch.log10(phi_sqerr + error_eps)
            psi_plot = torch.log10(psi_sqerr + error_eps)
        elif error_scale == "linear":
            phi_plot = phi_sqerr
            psi_plot = psi_sqerr
        else:
            raise ValueError(f"Unknown error_scale: {error_scale}")

        phi_err_results[frame_idx] = (
            phi_plot.cpu().numpy().reshape(n_y_plot, n_x_plot)
        )
        psi_err_results[frame_idx] = (
            psi_plot.cpu().numpy().reshape(n_y_plot, n_x_plot)
        )

    return phi_err_results, psi_err_results, x_grid, y_grid, t_grid


def create_animation_2d(
    save_path: Path,
    lx: float,
    ly: float,
    phi_results: np.ndarray,
    psi_results: np.ndarray,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    t_grid: torch.Tensor,
) -> None:
    """
    Create a two-panel GIF animation for phi and psi predictions.
    """
    n_t_plot = phi_results.shape[0]

    fig, (ax_phi, ax_psi) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    phi_min = np.min(phi_results)
    phi_max = np.max(phi_results)
    psi_min = np.min(psi_results)
    psi_max = np.max(psi_results)

    extent = [0.0, lx, 0.0, ly]

    im_phi = ax_phi.imshow(
        phi_results[0],
        origin="lower",
        extent=extent,
        vmin=phi_min,
        vmax=phi_max,
        aspect="equal",
        interpolation="bilinear",
    )

    im_psi = ax_psi.imshow(
        psi_results[0],
        origin="lower",
        extent=extent,
        vmin=psi_min,
        vmax=psi_max,
        aspect="equal",
        interpolation="bilinear",
    )

    ax_phi.set_title(r"Phase field $\phi(x,y,t)$")
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel("y")

    ax_psi.set_title(r"Concentration field $\psi(x,y,t)$")
    ax_psi.set_xlabel("x")
    ax_psi.set_ylabel("y")

    cbar_phi = fig.colorbar(im_phi, ax=ax_phi, fraction=0.046, pad=0.04)
    cbar_phi.set_label(r"$\phi$")

    cbar_psi = fig.colorbar(im_psi, ax=ax_psi, fraction=0.046, pad=0.04)
    cbar_psi.set_label(r"$\psi$")

    fig.suptitle(
        f"Minimal bio-inspired phase-field PINN | t = {float(t_grid[0]):.3f}",
        fontsize=14,
    )

    def animate(frame_idx: int):
        im_phi.set_data(phi_results[frame_idx])
        im_psi.set_data(psi_results[frame_idx])

        fig.suptitle(
            f"Minimal bio-inspired phase-field PINN | "
            f"t = {float(t_grid[frame_idx]):.3f}",
            fontsize=14,
        )

        return im_phi, im_psi

    animation = FuncAnimation(
        fig,
        animate,
        frames=n_t_plot,
        interval=80,
        blit=False,
    )

    animation.save(save_path, fps=10)
    plt.close(fig)


def create_error_animation_2d(
    save_path: Path,
    lx: float,
    ly: float,
    phi_err_results: np.ndarray,
    psi_err_results: np.ndarray,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    t_grid: torch.Tensor,
    error_scale: str = "linear",
) -> None:
    """
    Create a two-panel GIF animation for manufactured pointwise error maps.
    """
    n_t_plot = phi_err_results.shape[0]

    fig, (ax_phi, ax_psi) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    phi_min = np.min(phi_err_results)
    phi_max = np.max(phi_err_results)
    psi_min = np.min(psi_err_results)
    psi_max = np.max(psi_err_results)

    extent = [0.0, lx, 0.0, ly]

    im_phi = ax_phi.imshow(
        phi_err_results[0],
        origin="lower",
        extent=extent,
        vmin=phi_min,
        vmax=phi_max,
        aspect="equal",
        interpolation="bilinear",
    )

    im_psi = ax_psi.imshow(
        psi_err_results[0],
        origin="lower",
        extent=extent,
        vmin=psi_min,
        vmax=psi_max,
        aspect="equal",
        interpolation="bilinear",
    )

    if error_scale == "log":
        phi_title = r"$\log_{10}((\phi_\theta-\phi_{\mathrm{ex}})^2+\epsilon)$"
        psi_title = r"$\log_{10}((\psi_\theta-\psi_{\mathrm{ex}})^2+\epsilon)$"
        colorbar_label = "log squared error"
    elif error_scale == "linear":
        phi_title = r"$(\phi_\theta-\phi_{\mathrm{ex}})^2$"
        psi_title = r"$(\psi_\theta-\psi_{\mathrm{ex}})^2$"
        colorbar_label = "squared error"
    else:
        raise ValueError(f"Unknown error_scale: {error_scale}")

    ax_phi.set_title(phi_title)
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel("y")

    ax_psi.set_title(psi_title)
    ax_psi.set_xlabel("x")
    ax_psi.set_ylabel("y")

    cbar_phi = fig.colorbar(im_phi, ax=ax_phi, fraction=0.046, pad=0.04)
    cbar_phi.set_label(colorbar_label)

    cbar_psi = fig.colorbar(im_psi, ax=ax_psi, fraction=0.046, pad=0.04)
    cbar_psi.set_label(colorbar_label)

    fig.suptitle(
        f"Manufactured solution pointwise error | t = {float(t_grid[0]):.3f}",
        fontsize=14,
    )

    def animate(frame_idx: int):
        im_phi.set_data(phi_err_results[frame_idx])
        im_psi.set_data(psi_err_results[frame_idx])

        fig.suptitle(
            f"Manufactured solution pointwise error | "
            f"t = {float(t_grid[frame_idx]):.3f}",
            fontsize=14,
        )

        return im_phi, im_psi

    animation = FuncAnimation(
        fig,
        animate,
        frames=n_t_plot,
        interval=80,
        blit=False,
    )

    animation.save(save_path, fps=10)
    plt.close(fig)


def main() -> None:
    """
    Load trained checkpoints and generate the requested animation.
    """
    args = parse_args()
    args = update_args_from_run_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    check_dir = get_checkpoint_dir(args)

    if not check_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")

    save_dir = get_animation_dir(args)
    save_path = save_dir / args.gif_name

    if getattr(args, "manufactured", False):
        phi_err_results, psi_err_results, x_grid, y_grid, t_grid = (
            compute_error_pts_manufactured_tw(
                check_dir=check_dir,
                lx=args.lx,
                ly=args.ly,
                tmax=args.tmax,
                segment_length=args.segment_length,
                hidden_layers=args.hidden_layers,
                hidden_dim=args.hidden_dim,
                device=device,
                n_x_plot=args.n_x_plot,
                n_y_plot=args.n_y_plot,
                n_t_plot=args.n_t_plot,
                args=args,
                error_scale=args.error_scale,
                error_eps=args.error_eps,
            )
        )

        create_error_animation_2d(
            save_path=save_path,
            lx=args.lx,
            ly=args.ly,
            phi_err_results=phi_err_results,
            psi_err_results=psi_err_results,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
            error_scale=args.error_scale,
        )

    else:
        phi_results, psi_results, x_grid, y_grid, t_grid = compute_pts_to_visualize_tw(
            check_dir=check_dir,
            lx=args.lx,
            ly=args.ly,
            tmax=args.tmax,
            segment_length=args.segment_length,
            hidden_layers=args.hidden_layers,
            hidden_dim=args.hidden_dim,
            device=device,
            n_x_plot=args.n_x_plot,
            n_y_plot=args.n_y_plot,
            n_t_plot=args.n_t_plot,
        )

        create_animation_2d(
            save_path=save_path,
            lx=args.lx,
            ly=args.ly,
            phi_results=phi_results,
            psi_results=psi_results,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
        )

    print(f"Animation saved to: {save_path}")


if __name__ == "__main__":
    main()