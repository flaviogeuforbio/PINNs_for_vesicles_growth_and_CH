"""
Diagnostics for the 2D bio-inspired phase-field PINN.

This script evaluates trained time-windowed PINN models on a regular 2D grid and
computes integral diagnostics for the minimal bio-inspired phase-field model:

1. Total free energy:
       F = F_surf + F_osm

2. Surface and osmotic energy contributions.

3. Total conserved-field mass:
       integral psi dx dy

4. Diffuse internal vesicle area:
       integral 0.5 * (1 + p(phi)) dx dy

5. Global field ranges:
       min/max phi, min/max psi

The diagnostics are saved as a JSON file under:
    artifacts/bio-minimal/<run_name>/diagnostics/diagnostics_bio_2d.json

unless --output_json is explicitly provided.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import Tensor

from utils_bio_2d import (
    compute_energy,
    grad,
    integral_2d,
    load_models,
    p_interp,
    predict_windowed,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute diagnostics for a trained 2D bio phase-field PINN."
    )

    # -------------------------------------------------------------------------
    # Run and time-window settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run to diagnose.",
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
    # These are overwritten from run_config.json if available, except n_grid and
    # n_times, which are diagnostic choices.
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
    # Output settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output JSON path. If not provided, the file is saved under "
            "artifacts/bio-minimal/<run_name>/diagnostics/."
        ),
    )

    return parser.parse_args()


def update_args_from_run_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    Load training parameters from run_config.json, if available.

    Only keys already present in the diagnostic CLI namespace are overwritten.
    This keeps diagnostic-only options such as n_grid, n_times and output_json
    under direct CLI control.
    """
    config_path = Path("artifacts/bio-minimal") / args.run_name / "run_config.json"

    if not config_path.exists():
        print(f"WARNING: run_config.json not found at {config_path}")
        return args

    with open(config_path, "r") as f:
        config = json.load(f)

    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    print(f"Loaded run configuration from: {config_path}")

    return args


def evaluate_diagnostics_tw(
    models: List[torch.nn.Module],
    times: Tensor,
    n_grid: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Evaluate integral diagnostics for a time-windowed model ensemble.

    Parameters
    ----------
    models:
        List of trained segment models.
    times:
        One-dimensional tensor of global times at which diagnostics are computed.
    n_grid:
        Number of grid points per spatial direction.
    args:
        Runtime configuration.
    device:
        Torch device.

    Returns
    -------
    dict
        Raw diagnostic time series.
    """
    results = {
        "times": [],
        "energy": [],
        "surf_energy": [],
        "osm_energy": [],
        "mass_psi": [],
        "V_in": [],
        "phi_min": [],
        "phi_max": [],
        "psi_min": [],
        "psi_max": [],
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

        phi, _, psi, _ = predict_windowed(
            models=models,
            x=x,
            y=y,
            t_global=time_float,
            segment_length=args.segment_length,
            device=device,
        )

        phi_x = grad(phi, x)
        phi_y = grad(phi, y)

        energy, surf_energy, osm_energy, _ = compute_energy(
            phi=phi,
            psi=psi,
            phi_x=phi_x,
            phi_y=phi_y,
            x_lin=x_lin,
            y_lin=y_lin,
            n_grid=n_grid,
            args=args,
        )

        mass_psi = integral_2d(
            field=psi,
            x_lin=x_lin,
            y_lin=y_lin,
            n_grid=n_grid,
        )

        chi_in = 0.5 * (1.0 + p_interp(phi))

        v_in = integral_2d(
            field=chi_in,
            x_lin=x_lin,
            y_lin=y_lin,
            n_grid=n_grid,
        )

        results["times"].append(time_float)
        results["energy"].append(float(energy.detach().cpu()))
        results["surf_energy"].append(float(surf_energy.detach().cpu()))
        results["osm_energy"].append(float(osm_energy.detach().cpu()))
        results["mass_psi"].append(float(mass_psi.detach().cpu()))
        results["V_in"].append(float(v_in.detach().cpu()))
        results["phi_min"].append(float(phi.min().detach().cpu()))
        results["phi_max"].append(float(phi.max().detach().cpu()))
        results["psi_min"].append(float(psi.min().detach().cpu()))
        results["psi_max"].append(float(psi.max().detach().cpu()))

    return results


def _relative_change(final_value: float, initial_value: float) -> float:
    """
    Compute relative change with a small-denominator guard.
    """
    denom = abs(initial_value)

    if denom < 1e-14:
        return float("nan")

    return (final_value - initial_value) / denom


def compute_summary(diagnostics: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Build a compact summary from raw diagnostic time series.
    """
    mass0 = diagnostics["mass_psi"][0]
    v_in0 = diagnostics["V_in"][0]
    energy0 = diagnostics["energy"][0]
    surf0 = diagnostics["surf_energy"][0]
    osm0 = diagnostics["osm_energy"][0]

    delta_mass = [value - mass0 for value in diagnostics["mass_psi"]]
    delta_v_in = [value - v_in0 for value in diagnostics["V_in"]]
    delta_energy = [value - energy0 for value in diagnostics["energy"]]
    delta_surf = [value - surf0 for value in diagnostics["surf_energy"]]
    delta_osm = [value - osm0 for value in diagnostics["osm_energy"]]

    final_mass = diagnostics["mass_psi"][-1]
    final_v_in = diagnostics["V_in"][-1]
    final_energy = diagnostics["energy"][-1]
    final_surf = diagnostics["surf_energy"][-1]
    final_osm = diagnostics["osm_energy"][-1]

    summary = {
        "initial": {
            "mass_psi": mass0,
            "V_in": v_in0,
            "energy": energy0,
            "surf_energy": surf0,
            "osm_energy": osm0,
        },
        "final": {
            "time": diagnostics["times"][-1],
            "mass_psi": final_mass,
            "V_in": final_v_in,
            "energy": final_energy,
            "surf_energy": final_surf,
            "osm_energy": final_osm,
            "phi_min": diagnostics["phi_min"][-1],
            "phi_max": diagnostics["phi_max"][-1],
            "psi_min": diagnostics["psi_min"][-1],
            "psi_max": diagnostics["psi_max"][-1],
        },
        "final_deltas": {
            "mass_psi": delta_mass[-1],
            "V_in": delta_v_in[-1],
            "energy": delta_energy[-1],
            "surf_energy": delta_surf[-1],
            "osm_energy": delta_osm[-1],
        },
        "final_relative_changes": {
            "mass_psi": _relative_change(final_mass, mass0),
            "V_in": _relative_change(final_v_in, v_in0),
            "energy": _relative_change(final_energy, energy0),
            "surf_energy": _relative_change(final_surf, surf0),
            "osm_energy": _relative_change(final_osm, osm0),
        },
        "max_abs_deltas": {
            "mass_psi": max(abs(value) for value in delta_mass),
            "V_in": max(abs(value) for value in delta_v_in),
            "energy": max(abs(value) for value in delta_energy),
            "surf_energy": max(abs(value) for value in delta_surf),
            "osm_energy": max(abs(value) for value in delta_osm),
        },
        "global_ranges": {
            "phi_min": min(diagnostics["phi_min"]),
            "phi_max": max(diagnostics["phi_max"]),
            "psi_min": min(diagnostics["psi_min"]),
            "psi_max": max(diagnostics["psi_max"]),
        },
    }

    return summary


def build_time_records(diagnostics: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """
    Convert raw diagnostic arrays into one record per diagnostic time.
    """
    mass0 = diagnostics["mass_psi"][0]
    v_in0 = diagnostics["V_in"][0]
    energy0 = diagnostics["energy"][0]
    surf0 = diagnostics["surf_energy"][0]
    osm0 = diagnostics["osm_energy"][0]

    records = []

    for (
        time_value,
        mass_psi,
        v_in,
        energy,
        surf_energy,
        osm_energy,
        psi_min,
        psi_max,
        phi_min,
        phi_max,
    ) in zip(
        diagnostics["times"],
        diagnostics["mass_psi"],
        diagnostics["V_in"],
        diagnostics["energy"],
        diagnostics["surf_energy"],
        diagnostics["osm_energy"],
        diagnostics["psi_min"],
        diagnostics["psi_max"],
        diagnostics["phi_min"],
        diagnostics["phi_max"],
    ):
        record = {
            "time": time_value,
            "mass_psi": mass_psi,
            "delta_mass_psi": mass_psi - mass0,
            "relative_change_mass_psi": _relative_change(mass_psi, mass0),
            "V_in": v_in,
            "delta_V_in": v_in - v_in0,
            "relative_change_V_in": _relative_change(v_in, v_in0),
            "energy": energy,
            "delta_energy": energy - energy0,
            "relative_change_energy": _relative_change(energy, energy0),
            "surf_energy": surf_energy,
            "delta_surf_energy": surf_energy - surf0,
            "relative_change_surf_energy": _relative_change(surf_energy, surf0),
            "osm_energy": osm_energy,
            "delta_osm_energy": osm_energy - osm0,
            "relative_change_osm_energy": _relative_change(osm_energy, osm0),
            "psi_min": psi_min,
            "psi_max": psi_max,
            "phi_min": phi_min,
            "phi_max": phi_max,
        }

        records.append(record)

    return records


def get_output_path(args: argparse.Namespace) -> Path:
    """
    Return the JSON output path for diagnostics.
    """
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    diagnostics_dir = Path("artifacts/bio-minimal") / args.run_name / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    return diagnostics_dir / "diagnostics_bio_2d.json"


def save_diagnostics_json(
    diagnostics: Dict[str, List[float]],
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """
    Save raw diagnostics, compact summary and time records to JSON.
    """
    summary = compute_summary(diagnostics)
    time_records = build_time_records(diagnostics)

    output = {
        "run_name": args.run_name,
        "device": str(device),
        "diagnostics_settings": {
            "tmax": args.tmax,
            "segment_length": args.segment_length,
            "lx": args.lx,
            "ly": args.ly,
            "n_grid": args.n_grid,
            "n_times": len(diagnostics["times"]),
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
        "summary": summary,
        "time_series": time_records,
        "raw_arrays": diagnostics,
    }

    output_path = get_output_path(args)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    return output_path


def main() -> None:
    """
    Load trained models, compute diagnostics and save them to JSON.
    """
    args = parse_args()
    args = update_args_from_run_config(args)

    if args.n_grid <= 1:
        raise ValueError("n_grid must be greater than 1.")
    if args.n_times <= 1:
        raise ValueError("n_times must be greater than 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    check_dir = Path("artifacts/bio-minimal") / args.run_name / "weights"

    if not check_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")

    models = load_models(
        check_dir=check_dir,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    diagnostics_times = torch.linspace(0.0, args.tmax, args.n_times)

    diagnostics = evaluate_diagnostics_tw(
        models=models,
        times=diagnostics_times,
        n_grid=args.n_grid,
        args=args,
        device=device,
    )

    output_path = save_diagnostics_json(
        diagnostics=diagnostics,
        args=args,
        device=device,
    )

    print(f"Diagnostics saved to: {output_path}")


if __name__ == "__main__":
    main()