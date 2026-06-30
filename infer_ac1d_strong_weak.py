"""
Post-hoc strong/weak inference for the source-free AC1D blind inverse PINN.

This script loads a trained run produced by train_inverse_ac1d.py and estimates
m_phi from the learned field phi_theta(x,t), without using any manufactured
source term.

Strong form:
    -phi_t = m_phi * mu

    m_phi = <mu, -phi_t> / <mu, mu>

Weak form:
    phi(t_b) - phi(t_a) = -m_phi * int_{t_a}^{t_b} mu dt

    A = int_{t_a}^{t_b} mu dt
    B = -(phi(t_b) - phi(t_a))

    m_phi = <A, B> / <A, A>

where:

    mu = k * [ (1/eps) * (phi^3 - phi) - eps * phi_xx ]

The purpose of this script is diagnostic: it checks whether the learned field
phi_theta contains enough physically consistent information to recover m_phi
after training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from model import SimpleAC1D


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Post-hoc strong/weak m_phi inference from a trained AC1D PINN."
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help=(
            "Run directory or run name. If a path exists as given, it is used "
            "directly; otherwise it is interpreted relative to --out_root."
        ),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="artifacts",
        help="Root artifact directory used when --run_dir is a run name.",
    )
    parser.add_argument(
        "--n_samples_strong",
        type=int,
        default=30000,
        help="Number of random collocation points for strong-form inference.",
    )
    parser.add_argument(
        "--n_samples_weak",
        type=int,
        default=10000,
        help="Number of random local intervals for weak-form inference.",
    )
    parser.add_argument(
        "--n_quad",
        type=int,
        default=8,
        help="Number of Gauss-Legendre quadrature nodes for the weak form.",
    )
    parser.add_argument(
        "--dt_weak",
        type=float,
        default=0.025,
        help="Length of each weak-form local integration interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for post-hoc sampling.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional JSON output path. If not provided, results are saved under "
            "<run_dir>/diagnostics/."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Basic sanity checks for post-hoc inference settings.
    """
    if args.n_samples_strong <= 0:
        raise ValueError("n_samples_strong must be positive.")
    if args.n_samples_weak <= 0:
        raise ValueError("n_samples_weak must be positive.")
    if args.n_quad <= 0:
        raise ValueError("n_quad must be positive.")
    if args.dt_weak <= 0.0:
        raise ValueError("dt_weak must be positive.")


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """
    Resolve the run directory.

    This preserves compatibility with the original interface:

        --run_dir my_run

    which resolves to:

        artifacts/my_run

    while also allowing a direct path:

        --run_dir artifacts/my_run
    """
    candidate = Path(args.run_dir)

    if candidate.exists():
        return candidate

    return Path(args.out_root) / args.run_dir


# =============================================================================
# Autograd and physics
# =============================================================================

def grad(u: Tensor, z: Tensor) -> Tensor:
    """
    Compute du/dz using PyTorch autograd.
    """
    return torch.autograd.grad(
        u,
        z,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def mu_from_phi(
    phi: Tensor,
    x: Tensor,
    k: float,
    eps: float,
) -> Tensor:
    """
    Compute the Allen-Cahn chemical potential:

        mu = k * [ (1/eps) * (phi^3 - phi) - eps * phi_xx ].
    """
    phi_x = grad(phi, x)
    phi_xx = grad(phi_x, x)

    mu = k * (
        (1.0 / eps) * (phi**3 - phi)
        - eps * phi_xx
    )

    return mu


def rms(u: Tensor) -> Tensor:
    """
    Root mean square of a tensor.
    """
    return torch.sqrt(torch.mean(u**2))


def scalar_ls_estimate(A: Tensor, B: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute the scalar least-squares estimate:

        m = <A, B> / <A, A>.
    """
    numerator = torch.mean(A * B)
    denominator = torch.mean(A * A) + eps

    return numerator / denominator


def relative_error(value: float, target: float, eps: float = 1e-12) -> float:
    """
    Absolute relative error with a small-denominator guard.
    """
    return abs(value - target) / (abs(target) + eps)


# =============================================================================
# Loading
# =============================================================================

def load_config(run_dir: Path) -> Dict[str, Any]:
    """
    Load run_config.json from a trained run.
    """
    config_path = run_dir / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def load_model(
    run_dir: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """
    Load the trained SimpleAC1D checkpoint.
    """
    model_path = run_dir / "weights" / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found at: {model_path}")

    model = SimpleAC1D(
        hidden_layers=int(config["hidden_layers"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Load metrics.json if available.
    """
    metrics_path = run_dir / "metrics.json"

    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


# =============================================================================
# Strong-form inference
# =============================================================================

def infer_strong(
    model: nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    n_samples: int,
) -> Dict[str, float]:
    """
    Estimate m_phi from the strong-form relation:

        -phi_t = m_phi * mu.

    The estimate is computed on random points sampled uniformly in:
        x in [0, 1],
        t in [0, T].
    """
    x = torch.rand(n_samples, 1, device=device)
    t = torch.rand(n_samples, 1, device=device) * float(config["t_final"])

    x.requires_grad_(True)
    t.requires_grad_(True)

    phi = model(x, t)
    phi_t = grad(phi, t)

    mu = mu_from_phi(
        phi=phi,
        x=x,
        k=float(config["k"]),
        eps=float(config["eps"]),
    )

    A = mu
    B = -phi_t

    m_strong = scalar_ls_estimate(A, B)
    residual = m_strong * A - B

    results = {
        "m_phi_strong_net": float(m_strong.detach().cpu()),
        "rmse_strong_residual": float(rms(residual).detach().cpu()),
        "A_mu_rms_strong": float(rms(A).detach().cpu()),
        "B_minus_phi_t_rms_strong": float(rms(B).detach().cpu()),
    }

    return results


# =============================================================================
# Weak-form inference
# =============================================================================

def build_gauss_legendre_quadrature(
    n_quad: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Build Gauss-Legendre nodes and weights on [-1, 1].
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


def sample_weak_intervals(
    n_samples: int,
    t_final: float,
    dt_weak: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample spatial points and time intervals [t_a, t_a + dt_weak].
    """
    if dt_weak >= t_final:
        raise ValueError(f"dt_weak={dt_weak} must be smaller than t_final={t_final}.")

    x = torch.rand(n_samples, 1, device=device)

    t_a = torch.rand(n_samples, 1, device=device) * (t_final - dt_weak)
    t_b = t_a + dt_weak

    return x, t_a, t_b


def build_weak_quadrature_points(
    x: Tensor,
    t_a: Tensor,
    t_b: Tensor,
    nodes: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Expand sampled intervals into flattened quadrature points.
    """
    n_quad = nodes.shape[1]
    dt_weak = t_b - t_a
    t_mid = 0.5 * (t_a + t_b)

    t_q = t_mid + 0.5 * dt_weak * nodes

    x_q = x.repeat_interleave(n_quad, dim=0).detach().clone().requires_grad_(True)
    t_q_flat = t_q.reshape(-1, 1).detach().clone().requires_grad_(True)

    return x_q, t_q_flat


def infer_weak(
    model: nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    n_samples: int,
    n_quad: int,
    dt_weak: float,
) -> Dict[str, float]:
    """
    Estimate m_phi from the weak-form relation:

        phi(t_b) - phi(t_a) = -m_phi * int_{t_a}^{t_b} mu dt.

    The estimate is computed by Gauss-Legendre quadrature over randomly sampled
    local time intervals.
    """
    t_final = float(config["t_final"])

    if dt_weak <= 0.0:
        raise ValueError(f"dt_weak must be positive, got {dt_weak}.")
    if dt_weak >= t_final:
        raise ValueError(f"dt_weak={dt_weak} must be smaller than t_final={t_final}.")

    x, t_a, t_b = sample_weak_intervals(
        n_samples=n_samples,
        t_final=t_final,
        dt_weak=dt_weak,
        device=device,
    )

    with torch.no_grad():
        phi_a = model(x, t_a)
        phi_b = model(x, t_b)

    delta_phi = phi_b - phi_a

    nodes, weights = build_gauss_legendre_quadrature(
        n_quad=n_quad,
        device=device,
    )

    x_q, t_q_flat = build_weak_quadrature_points(
        x=x,
        t_a=t_a,
        t_b=t_b,
        nodes=nodes,
    )

    phi_q = model(x_q, t_q_flat)

    mu_q = mu_from_phi(
        phi=phi_q,
        x=x_q,
        k=float(config["k"]),
        eps=float(config["eps"]),
    )

    mu_q = mu_q.reshape(n_samples, n_quad)

    int_mu = 0.5 * dt_weak * torch.sum(weights * mu_q, dim=1, keepdim=True)

    A = int_mu
    B = -delta_phi

    m_weak = scalar_ls_estimate(A, B)
    residual = m_weak * A - B

    results = {
        "m_phi_weak_net": float(m_weak.detach().cpu()),
        "rmse_weak_residual": float(rms(residual).detach().cpu()),
        "A_int_mu_rms_weak": float(rms(A).detach().cpu()),
        "B_minus_delta_phi_rms_weak": float(rms(B).detach().cpu()),
        "dt_weak": float(dt_weak),
        "n_quad": int(n_quad),
    }

    return results


# =============================================================================
# Saving and reporting
# =============================================================================

def build_results(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    strong_results: Dict[str, float],
    weak_results: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Merge all post-hoc inference results into one dictionary.
    """
    m_phi_true = float(config["m_phi_true"])
    m_phi_trainable = metrics.get("m_phi_final", None)

    results = {
        "m_phi_true": m_phi_true,
        "m_phi_trainable": m_phi_trainable,
        **strong_results,
        **weak_results,
        "rel_error_strong_net": relative_error(
            strong_results["m_phi_strong_net"],
            m_phi_true,
        ),
        "rel_error_weak_net": relative_error(
            weak_results["m_phi_weak_net"],
            m_phi_true,
        ),
        "n_samples_strong": int(args.n_samples_strong),
        "n_samples_weak": int(args.n_samples_weak),
        "seed": int(args.seed),
    }

    if m_phi_trainable is not None:
        results["rel_error_trainable"] = relative_error(
            float(m_phi_trainable),
            m_phi_true,
        )

    return results


def get_output_path(args: argparse.Namespace, run_dir: Path) -> Path:
    """
    Return the JSON output path.
    """
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    return diagnostics_dir / f"strong_weak_inference_dt{args.dt_weak:.4f}.json"


def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """
    Save post-hoc inference results to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("=" * 80)
    print(f"Saved results to: {output_path}")
    print("=" * 80)


def print_header(
    run_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """
    Print run information.
    """
    print("=" * 80)
    print("AC1D POST-HOC STRONG/WEAK INFERENCE")
    print("=" * 80)
    print(f"run_dir          : {run_dir}")
    print(f"device           : {device}")
    print(f"m_phi_true       : {float(config['m_phi_true']):.8e}")
    print(f"n_samples_strong : {args.n_samples_strong}")
    print(f"n_samples_weak   : {args.n_samples_weak}")
    print(f"dt_weak          : {args.dt_weak}")
    print(f"n_quad           : {args.n_quad}")
    print("=" * 80)


def print_results(results: Dict[str, Any]) -> None:
    """
    Print results in a readable format.
    """
    print("\nRESULTS")
    print("=" * 80)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.8e}")
        else:
            print(f"{key:35s}: {value}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Load a trained AC1D PINN and run post-hoc strong/weak inference.
    """
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = resolve_run_dir(args)
    config = load_config(run_dir)
    metrics = load_metrics(run_dir)

    model = load_model(
        run_dir=run_dir,
        config=config,
        device=device,
    )

    print_header(
        run_dir=run_dir,
        device=device,
        config=config,
        args=args,
    )

    strong_results = infer_strong(
        model=model,
        config=config,
        device=device,
        n_samples=args.n_samples_strong,
    )

    weak_results = infer_weak(
        model=model,
        config=config,
        device=device,
        n_samples=args.n_samples_weak,
        n_quad=args.n_quad,
        dt_weak=args.dt_weak,
    )

    results = build_results(
        config=config,
        metrics=metrics,
        strong_results=strong_results,
        weak_results=weak_results,
        args=args,
    )

    print_results(results)

    output_path = get_output_path(
        args=args,
        run_dir=run_dir,
    )

    save_results(
        results=results,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()