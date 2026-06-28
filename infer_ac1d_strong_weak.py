"""
infer_ac1d_strong_weak.py

Minimal post-hoc strong/weak inference for the blind AC1D PINN.

This script loads a trained PINN run produced by train_inverse_ac1d.py and
computes two least-squares estimates of m_phi from the learned field phi_theta:

Strong form:
    -phi_t = m_phi * mu

    m_phi = <mu, -phi_t> / <mu, mu>

Weak form:
    phi(t_b) - phi(t_a) = -m_phi * int_{t_a}^{t_b} mu dt

    A = int_{t_a}^{t_b} mu dt
    B = -(phi(t_b) - phi(t_a))

    m_phi = <A, B> / <A, A>

No source term is used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import SimpleAC1D


# =============================================================================
# AUTOGRAD + PHYSICS
# =============================================================================

def grad(u, z):
    return torch.autograd.grad(
        u,
        z,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]


def mu_from_phi(phi, x, k: float, eps: float):
    phi_x = grad(phi, x)
    phi_xx = grad(phi_x, x)

    mu = k * (
        (1.0 / eps) * (phi**3 - phi)
        - eps * phi_xx
    )

    return mu


# =============================================================================
# LOADING
# =============================================================================

def load_config(run_dir: Path):
    config_path = run_dir / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def load_model(run_dir: Path, config, device):
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


def load_metrics(run_dir: Path):
    metrics_path = run_dir / "metrics.json"

    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics


# =============================================================================
# STRONG FORM INFERENCE
# =============================================================================

def infer_strong(model, config, device, n_samples: int):
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

    eps_num = 1e-12

    m_strong = torch.mean(A * B) / (torch.mean(A * A) + eps_num)
    residual = m_strong * A - B

    results = {
        "m_phi_strong_net": float(m_strong.detach().cpu()),
        "rmse_strong_residual": float(torch.sqrt(torch.mean(residual**2)).detach().cpu()),
        "A_mu_rms_strong": float(torch.sqrt(torch.mean(A**2)).detach().cpu()),
        "B_minus_phi_t_rms_strong": float(torch.sqrt(torch.mean(B**2)).detach().cpu()),
    }

    return results


# =============================================================================
# WEAK FORM INFERENCE
# =============================================================================

def infer_weak(model, config, device, n_samples: int, n_quad: int, dt_weak: float):
    t_final = float(config["t_final"])

    if dt_weak <= 0.0:
        raise ValueError(f"dt_weak must be positive, got {dt_weak}")

    if dt_weak >= t_final:
        raise ValueError(f"dt_weak={dt_weak} must be smaller than t_final={t_final}")

    x = torch.rand(n_samples, 1, device=device)

    t_a = torch.rand(n_samples, 1, device=device) * (t_final - dt_weak)
    t_b = t_a + dt_weak

    with torch.no_grad():
        phi_a = model(x, t_a)
        phi_b = model(x, t_b)

    delta_phi = phi_b - phi_a

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

    t_mid = 0.5 * (t_a + t_b)
    t_q = t_mid + 0.5 * dt_weak * nodes

    x_q = x.repeat_interleave(n_quad, dim=0).detach().clone().requires_grad_(True)
    t_q_flat = t_q.reshape(-1, 1).detach().clone().requires_grad_(True)

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

    eps_num = 1e-12

    m_weak = torch.mean(A * B) / (torch.mean(A * A) + eps_num)
    residual = m_weak * A - B

    results = {
        "m_phi_weak_net": float(m_weak.detach().cpu()),
        "rmse_weak_residual": float(torch.sqrt(torch.mean(residual**2)).detach().cpu()),
        "A_int_mu_rms_weak": float(torch.sqrt(torch.mean(A**2)).detach().cpu()),
        "B_minus_delta_phi_rms_weak": float(torch.sqrt(torch.mean(B**2)).detach().cpu()),
        "dt_weak": float(dt_weak),
        "n_quad": int(n_quad),
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples_strong", type=int, default=30000)
    parser.add_argument("--n_samples_weak", type=int, default=10000)
    parser.add_argument("--n_quad", type=int, default=8)
    parser.add_argument("--dt_weak", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    metrics = load_metrics(run_dir)

    model = load_model(run_dir, config, device)

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

    m_phi_true = float(config["m_phi_true"])

    m_phi_trainable = metrics.get("m_phi_final", None)

    results = {
        "m_phi_true": m_phi_true,
        "m_phi_trainable": m_phi_trainable,
        **strong_results,
        **weak_results,
        "rel_error_strong_net": abs(strong_results["m_phi_strong_net"] - m_phi_true) / abs(m_phi_true),
        "rel_error_weak_net": abs(weak_results["m_phi_weak_net"] - m_phi_true) / abs(m_phi_true),
        "n_samples_strong": int(args.n_samples_strong),
        "n_samples_weak": int(args.n_samples_weak),
        "seed": int(args.seed),
    }

    if m_phi_trainable is not None:
        results["rel_error_trainable"] = abs(float(m_phi_trainable) - m_phi_true) / abs(m_phi_true)

    print("\nRESULTS")
    print("=" * 80)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.8e}")
        else:
            print(f"{key:35s}: {value}")

    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"strong_weak_inference_dt{args.dt_weak:.4f}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("=" * 80)
    print(f"Saved results to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()