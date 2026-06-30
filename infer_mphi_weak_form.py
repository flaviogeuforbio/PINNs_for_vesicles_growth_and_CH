"""
Weak-form least-squares inference of m_phi for manufactured inverse runs.

This script estimates the shape mobility m_phi from the integrated phi-equation:

    phi_t + m_phi * mu - S_phi = 0

Integrated over a local time interval [t_a, t_b], this gives:

    phi(t_b) - phi(t_a) + m_phi * int_{t_a}^{t_b} mu dt
        - int_{t_a}^{t_b} S_phi dt = 0

Therefore:

    m_phi * A = B

with:

    A = int_{t_a}^{t_b} mu dt
    B = int_{t_a}^{t_b} S_phi dt - [phi(t_b) - phi(t_a)]

The weak least-squares estimate is:

    m_phi^weak = <A, B> / <A, A>

The same estimator is computed both using the learned PINN fields and using the
exact manufactured fields as a sanity check.

Important
---------
This script is part of the manufactured inverse diagnostics. Since S_phi contains
the target mobility m_phi explicitly, this is not a source-free blind inverse
benchmark.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from manufactured_bio_2d import exact_fields_ms, manufactured_sources
from utils_bio_2d import load_models, predict_windowed_tensor


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Weak-form LS inference of m_phi for manufactured PINN runs."
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
        "--n_samples",
        type=int,
        default=30000,
        help="Number of random local time intervals used for weak inference.",
    )
    parser.add_argument(
        "--n_quad",
        type=int,
        default=8,
        help="Number of Gauss-Legendre quadrature nodes per interval.",
    )
    parser.add_argument(
        "--dt_weak",
        type=float,
        default=0.025,
        help="Length of the local weak-form integration interval.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional JSON output path. If not provided, results are saved under "
            "<root>/<run_name>/diagnostics/."
        ),
    )

    return parser.parse_args()


def load_args_from_config(run_dir: Path) -> argparse.Namespace:
    """
    Load the training configuration saved for a run.
    """
    config_path = run_dir / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return argparse.Namespace(**config)


def sample_weak_intervals(
    args: argparse.Namespace,
    device: torch.device,
    n_samples: int,
    dt_weak: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample random space points and local time intervals [t_a, t_b].

    The same spatial point (x, y) is used at both endpoints and along the
    quadrature interval.
    """
    x = torch.rand(n_samples, 1, device=device) * args.lx
    y = torch.rand(n_samples, 1, device=device) * args.ly

    t_a = torch.rand(n_samples, 1, device=device) * (args.tmax - dt_weak)
    t_b = t_a + dt_weak

    return x, y, t_a, t_b


def build_gauss_legendre_quadrature(
    n_quad: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Build Gauss-Legendre nodes and weights on the reference interval [-1, 1].
    """
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)

    nodes = torch.tensor(
        nodes_np,
        dtype=torch.float32,
        device=device,
    ).reshape(1, n_quad)

    weights = torch.tensor(
        weights_np,
        dtype=torch.float32,
        device=device,
    ).reshape(1, n_quad)

    return nodes, weights


def build_quadrature_points(
    x: Tensor,
    y: Tensor,
    t_a: Tensor,
    t_b: Tensor,
    nodes: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Expand sampled intervals into flattened quadrature points.

    Returns
    -------
    xq, yq, tq_flat:
        Flattened tensors with shape (n_samples * n_quad, 1).
    """
    n_quad = nodes.shape[1]
    dt_weak = t_b - t_a
    t_mid = 0.5 * (t_a + t_b)

    tq = t_mid + 0.5 * dt_weak * nodes

    xq = x.repeat_interleave(n_quad, dim=0)
    yq = y.repeat_interleave(n_quad, dim=0)
    tq_flat = tq.reshape(-1, 1)

    return xq, yq, tq_flat


def weak_ls_estimate(A: Tensor, B: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute the scalar weak least-squares estimate:

        m = <A, B> / <A, A>.
    """
    numerator = torch.mean(A * B)
    denominator = torch.mean(A * A) + eps

    return numerator / denominator


def rms(u: Tensor) -> Tensor:
    """
    Root mean square of a tensor.
    """
    return torch.sqrt(torch.mean(u**2))


def relative_error(value: float, target: float, eps: float = 1e-12) -> float:
    """
    Absolute relative error with a small-denominator guard.
    """
    return abs(value - target) / (abs(target) + eps)


def infer_mphi_weak_local(
    models: List[nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    n_samples: int,
    n_quad: int,
    dt_weak: float,
) -> Dict[str, float]:
    """
    Infer m_phi using local weak-form least squares.

    Parameters
    ----------
    models:
        Trained time-windowed PINN ensemble.
    args:
        Run configuration loaded from run_config.json.
    device:
        Torch device.
    n_samples:
        Number of random intervals.
    n_quad:
        Number of Gauss-Legendre quadrature nodes.
    dt_weak:
        Length of each local time interval.

    Returns
    -------
    dict
        Weak-form inference results and diagnostic RMS values.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_quad <= 0:
        raise ValueError("n_quad must be positive.")
    if dt_weak <= 0.0:
        raise ValueError(f"dt_weak must be positive, got {dt_weak}.")
    if dt_weak >= args.tmax:
        raise ValueError(f"dt_weak={dt_weak} must be smaller than tmax={args.tmax}.")

    x, y, t_a, t_b = sample_weak_intervals(
        args=args,
        device=device,
        n_samples=n_samples,
        dt_weak=dt_weak,
    )

    # Endpoint predictions give delta phi = phi(t_b) - phi(t_a).
    with torch.no_grad():
        out_a = predict_windowed_tensor(
            models=models,
            x=x,
            y=y,
            t_global=t_a,
            segment_length=args.segment_length,
            device=device,
        )
        phi_a = out_a[:, 0:1]

        out_b = predict_windowed_tensor(
            models=models,
            x=x,
            y=y,
            t_global=t_b,
            segment_length=args.segment_length,
            device=device,
        )
        phi_b = out_b[:, 0:1]

    delta_phi = phi_b - phi_a

    nodes, weights = build_gauss_legendre_quadrature(
        n_quad=n_quad,
        device=device,
    )

    xq, yq, tq_flat = build_quadrature_points(
        x=x,
        y=y,
        t_a=t_a,
        t_b=t_b,
        nodes=nodes,
    )

    # Evaluate learned mu on quadrature points.
    with torch.no_grad():
        out_q = predict_windowed_tensor(
            models=models,
            x=xq,
            y=yq,
            t_global=tq_flat,
            segment_length=args.segment_length,
            device=device,
        )
        mu_q = out_q[:, 1:2]

    # Manufactured source term. This is what makes the diagnostic forced rather
    # than source-free.
    source_phi_q, _ = manufactured_sources(xq, yq, tq_flat, args)

    mu_q = mu_q.reshape(n_samples, n_quad)
    source_phi_q = source_phi_q.reshape(n_samples, n_quad)

    int_mu = 0.5 * dt_weak * torch.sum(weights * mu_q, dim=1, keepdim=True)
    int_source_phi = (
        0.5 * dt_weak * torch.sum(weights * source_phi_q, dim=1, keepdim=True)
    )

    A = int_mu
    B = int_source_phi - delta_phi

    mphi_weak_net = weak_ls_estimate(A, B)

    weak_res_true_net = args.m_phi * A - B
    weak_res_est_net = mphi_weak_net * A - B

    # Exact manufactured sanity check.
    phi_a_exact, _, _, _ = exact_fields_ms(x, y, t_a, args)
    phi_b_exact, _, _, _ = exact_fields_ms(x, y, t_b, args)

    delta_phi_exact = phi_b_exact - phi_a_exact

    _, mu_exact_q, _, _ = exact_fields_ms(xq, yq, tq_flat, args)
    source_phi_exact_q, _ = manufactured_sources(xq, yq, tq_flat, args)

    mu_exact_q = mu_exact_q.reshape(n_samples, n_quad)
    source_phi_exact_q = source_phi_exact_q.reshape(n_samples, n_quad)

    int_mu_exact = (
        0.5 * dt_weak * torch.sum(weights * mu_exact_q, dim=1, keepdim=True)
    )
    int_source_phi_exact = (
        0.5
        * dt_weak
        * torch.sum(weights * source_phi_exact_q, dim=1, keepdim=True)
    )

    A_exact = int_mu_exact
    B_exact = int_source_phi_exact - delta_phi_exact

    mphi_weak_exact = weak_ls_estimate(A_exact, B_exact)

    weak_res_true_exact = args.m_phi * A_exact - B_exact
    weak_res_est_exact = mphi_weak_exact * A_exact - B_exact

    mphi_weak_net_float = float(mphi_weak_net.detach().cpu())
    mphi_weak_exact_float = float(mphi_weak_exact.detach().cpu())

    results = {
        "m_phi_true": float(args.m_phi),
        "m_phi_weak_local_net": mphi_weak_net_float,
        "m_phi_weak_local_exact": mphi_weak_exact_float,
        "rel_error_net": relative_error(mphi_weak_net_float, float(args.m_phi)),
        "rel_error_exact": relative_error(mphi_weak_exact_float, float(args.m_phi)),
        "rmse_weak_res_true_m_net": float(rms(weak_res_true_net).detach().cpu()),
        "rmse_weak_res_est_m_net": float(rms(weak_res_est_net).detach().cpu()),
        "rmse_weak_res_true_m_exact": float(rms(weak_res_true_exact).detach().cpu()),
        "rmse_weak_res_est_m_exact": float(rms(weak_res_est_exact).detach().cpu()),
        "A_int_mu_rms": float(rms(A).detach().cpu()),
        "B_rhs_rms": float(rms(B).detach().cpu()),
        "delta_phi_rms": float(rms(delta_phi).detach().cpu()),
        "int_S_phi_rms": float(rms(int_source_phi).detach().cpu()),
        "A_exact_int_mu_rms": float(rms(A_exact).detach().cpu()),
        "B_exact_rhs_rms": float(rms(B_exact).detach().cpu()),
        "delta_phi_exact_rms": float(rms(delta_phi_exact).detach().cpu()),
        "int_S_phi_exact_rms": float(rms(int_source_phi_exact).detach().cpu()),
        "dt_weak": float(dt_weak),
        "n_samples": int(n_samples),
        "n_quad": int(n_quad),
    }

    return results


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

    return diagnostics_dir / f"weak_form_mphi_inference_dt{cli_args.dt_weak:.4f}.json"


def save_results(results: Dict[str, float], output_path: Path) -> None:
    """
    Save weak-form inference results to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved weak-form inference results to: {output_path}")


def print_results(results: Dict[str, float]) -> None:
    """
    Print weak-form inference results in a readable format.
    """
    print("\nRESULTS")
    print("=" * 80)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.8e}")
        else:
            print(f"{key:35s}: {value}")


def main() -> None:
    """
    Entry point.
    """
    cli_args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cli_args.root) / cli_args.run_name
    check_dir = run_dir / "weights"

    print("=" * 80)
    print("WEAK-FORM PARAMETER INFERENCE: m_phi")
    print("=" * 80)
    print(f"Run directory : {run_dir}")
    print(f"Device        : {device}")
    print(f"dt_weak       : {cli_args.dt_weak}")
    print(f"n_samples     : {cli_args.n_samples}")
    print(f"n_quad        : {cli_args.n_quad}")

    args = load_args_from_config(run_dir)

    models = load_models(
        check_dir=check_dir,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    inference_results = infer_mphi_weak_local(
        models=models,
        args=args,
        device=device,
        n_samples=cli_args.n_samples,
        n_quad=cli_args.n_quad,
        dt_weak=cli_args.dt_weak,
    )

    print_results(inference_results)

    output_path = get_output_path(
        cli_args=cli_args,
        run_dir=run_dir,
    )

    save_results(
        results=inference_results,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()