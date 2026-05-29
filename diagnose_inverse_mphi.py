import argparse
import json
from pathlib import Path

import torch
import numpy as np

from models import BioACCHPINN2d
from manufactured_bio_2d import manufactured_sources, exact_fields_ms
from utils_bio_2d import grad


def load_args_from_config(run_dir):
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return argparse.Namespace(**config)


def load_trained_model(run_dir, args, device):
    weights_dir = run_dir / "weights"

    ckpt_path = weights_dir / "segment_0.pt"

    model = BioACCHPINN2d(
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state)

    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    return model


def sample_pde_points(args, n_points, device):
    x = args.lx * torch.rand(n_points, 1, device=device)
    y = args.ly * torch.rand(n_points, 1, device=device)
    t = args.tmax * torch.rand(n_points, 1, device=device)

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    return x, y, t


def estimate_m_phi_ls(model, args, device, n_points=20000):
    x, y, t = sample_pde_points(args, n_points, device)

    #model predictions
    phi, mu, psi, nu = model(x, y, t)

    phi_t = grad(phi, t)

    # Sorgente manufactured costruita col valore vero args.m_phi.
    S_phi, _ = manufactured_sources(x, y, t, args)

    a = mu.detach()
    b = (S_phi - phi_t).detach()

    eps = 1e-12

    m_ls = torch.mean(a * b) / (torch.mean(a * a) + eps)

    r_true = phi_t.detach() + args.m_phi * a - S_phi.detach()
    r_ls = phi_t.detach() + m_ls * a - S_phi.detach()

    #sanity check: using exact fields we should get m_ls = args.m_phi
    phi_ex, mu_ex, psi_ex, nu_ex = exact_fields_ms(x, y, t, args)
    phi_ex_t = grad(phi_ex, t)

    a_ex = mu_ex.detach()
    b_ex = (S_phi - phi_ex_t).detach()

    m_ls_exact = torch.mean(a_ex * b_ex) / (torch.mean(a_ex * a_ex) + eps)

    #mixed diagnostics to detect the source of error (from mu or phi_t)
    m_ls_exact_mu_net_phit = torch.mean(a_ex * b) / (torch.mean(a_ex * a_ex) + eps)
    m_ls_net_mu_exact_phit = torch.mean(a * b_ex) / (torch.mean(a * a) + eps)

    diagnostics = {
        "m_phi_true": float(args.m_phi),
        "m_phi_LS_net_fields": float(m_ls.detach().cpu()),
        "m_phi_LS_exact_fields": float(m_ls_exact.detach().cpu()),
        "m_phi_LS_exact_mu_net_phit": float(m_ls_exact_mu_net_phit.detach().cpu()),
        "m_phi_LS_net_mu_exact_phit": float(m_ls_net_mu_exact_phit.detach().cpu()),
        "rmse_R_phi_true_m": float(torch.sqrt(torch.mean(r_true ** 2)).detach().cpu()),
        "rmse_R_phi_LS_m": float(torch.sqrt(torch.mean(r_ls ** 2)).detach().cpu()),
        "mu_theta_rms": float(torch.sqrt(torch.mean(a ** 2)).detach().cpu()),
        "phi_t_theta_rms": float(torch.sqrt(torch.mean(phi_t.detach() ** 2)).detach().cpu()),
        "S_phi_rms": float(torch.sqrt(torch.mean(S_phi.detach() ** 2)).detach().cpu()),
    }

    return diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--root", type=str, default="artifacts/manufactured")
    # parser.add_argument("--n_points", type=int, default=20000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.root) / args.run_name

    args = load_args_from_config(run_dir)
    model = load_trained_model(run_dir, args, device)

    diagnostics = estimate_m_phi_ls(
        model=model,
        args=args,
        device=device
    )

    print("\nINVERSE m_phi LEAST-SQUARES DIAGNOSTICS")
    print("=" * 60)
    for k, v in diagnostics.items():
        print(f"{k:35s}: {v:.8e}")


if __name__ == "__main__":
    main()