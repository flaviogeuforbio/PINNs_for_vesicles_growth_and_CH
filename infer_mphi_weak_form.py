import argparse
import json 
from pathlib import Path 

import torch 
import numpy as np 

from models import BioACCHPINN2d
from manufactured_bio_2d import manufactured_sources, exact_fields_ms
from utils_bio_2d import load_models, predict_windowed_tensor

#function to load args from run config (this will be different from CLI data related to this specific script)
def load_args_from_config(run_dir: Path):
    config_path = run_dir / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return argparse.Namespace(**config)


def infer_mphi_weak_local(
    models, 
    args, 
    device, 
    n_samples: int,
    n_quad: int, 
    dt_weak: float
):
    
    #sanity checks
    if dt_weak <= 0:
        raise ValueError(f"dt_weak must be positive, got {dt_weak}")

    if dt_weak >= args.tmax:
        raise ValueError(
            f"dt_weak={dt_weak} must be smaller than tmax={args.tmax}"
        )
    
    #sample space points and time intervals (of length = dt_weak) to perform integration  
    x = torch.rand(n_samples, 1, device = device) * args.lx
    y = torch.rand(n_samples, 1, device = device) * args.ly

    t_a = torch.rand(n_samples, 1, device = device) * (args.tmax - dt_weak)
    t_b = t_a + dt_weak

    #compute phi values in {t_a, t_b} and the delta phi(b) - phi(a)
    with torch.no_grad():
        out_a = predict_windowed_tensor(
            models, 
            x, y, t_a,
            args.segment_length,
            device
        )
        phi_a = out_a[:, 0:1]

        out_b = predict_windowed_tensor(
            models, 
            x, y, t_b, 
            args.segment_length, 
            device
        )
        phi_b = out_b[:, 0:1]

    delta_phi = phi_b - phi_a

    #Gauss-Legendre quadrature on each local interval 
    #computing nodes and weights
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n_quad)

    nodes = torch.tensor(
        nodes_np, 
        dtype = torch.float32, 
        device = device
    ).reshape(1, n_quad)

    weights = torch.tensor(
        weights_np, 
        dtype = torch.float32, 
        device = device
    ).reshape(1, n_quad)

    #compute 'quadrature' times and (x, y) 
    t_mid = 0.5 * (t_a + t_b)
    tq = t_mid + 0.5 * dt_weak * nodes #this has shape [n_samples, n_quad]

    xq = x.repeat_interleave(n_quad, dim = 0)
    yq = y.repeat_interleave(n_quad, dim = 0)
    tq_flat = tq.reshape(-1, 1)

    #evaluate mu_theta and S_phi on quadrature points 
    with torch.no_grad():
        out = predict_windowed_tensor(
            models, 
            xq, yq, tq_flat, 
            args.segment_length, 
            device
        )
        mu_q = out[:, 1:2]

    S_phi_q, _ = manufactured_sources(
        xq, yq, tq_flat, args
    )

    #performing integrals 
    mu_q = mu_q.reshape(n_samples, n_quad)
    S_phi_q = S_phi_q.reshape(n_samples, n_quad)

    int_mu = 0.5 * dt_weak * torch.sum(weights * mu_q, dim = 1, keepdim = True)
    int_S_phi = 0.5 * dt_weak * torch.sum(weights * S_phi_q, dim = 1, keepdim = True)

    #computing weak least-squares estimate
    A = int_mu
    B = int_S_phi - delta_phi

    eps = 1e-12

    mphi_weak = torch.mean(A * B) / (torch.mean(A * A) + eps)

    weak_res_true = args.m_phi * A - B #residual with ground truth m_phi
    weak_res_est = mphi_weak * A - B #residual with weak-LS estimate of m_phi

    #exact fields sanity check
    phi_a_ex, *_ = exact_fields_ms(x, y, t_a, args)
    phi_b_ex, *_ = exact_fields_ms(x, y, t_b, args)

    delta_phi_ex = phi_b_ex - phi_a_ex

    _, mu_ex_q, *_ = exact_fields_ms(xq, yq, tq_flat, args)
    S_phi_ex_q, _ = manufactured_sources(xq, yq, tq_flat, args)

    mu_ex_q = mu_ex_q.reshape(n_samples, n_quad)
    S_phi_ex_q = S_phi_ex_q.reshape(n_samples, n_quad)

    int_mu_ex = 0.5 * dt_weak * torch.sum(weights * mu_ex_q, dim = 1, keepdim = True)
    int_S_phi_ex = 0.5 * dt_weak * torch.sum(weights * S_phi_ex_q, dim = 1, keepdim = True)

    A_ex = int_mu_ex
    B_ex = int_S_phi_ex - delta_phi_ex

    mphi_weak_exact = torch.mean(A_ex * B_ex) / (torch.mean(A_ex * A_ex) + eps)

    weak_res_exact_true = args.m_phi * A_ex - B_ex
    weak_res_exact_est = mphi_weak_exact * A_ex - B_ex

    #inference results & diagnostics
    results = {
        "m_phi_true": float(args.m_phi),
        "m_phi_weak_local_net": float(mphi_weak.detach().cpu()),
        "m_phi_weak_local_exact": float(mphi_weak_exact.detach().cpu()),

        "rel_error_net": float(
            abs(mphi_weak.detach().cpu().item() - args.m_phi) / abs(args.m_phi)
        ),
        "rel_error_exact": float(
            abs(mphi_weak_exact.detach().cpu().item() - args.m_phi) / abs(args.m_phi)
        ),

        "rmse_weak_res_true_m_net": float(
            torch.sqrt(torch.mean(weak_res_true ** 2)).detach().cpu()
        ),
        "rmse_weak_res_est_m_net": float(
            torch.sqrt(torch.mean(weak_res_est ** 2)).detach().cpu()
        ),

        "rmse_weak_res_true_m_exact": float(
            torch.sqrt(torch.mean(weak_res_exact_true ** 2)).detach().cpu()
        ),
        "rmse_weak_res_est_m_exact": float(
            torch.sqrt(torch.mean(weak_res_exact_est ** 2)).detach().cpu()
        ),

        "A_int_mu_rms": float(torch.sqrt(torch.mean(A ** 2)).detach().cpu()),
        "B_rhs_rms": float(torch.sqrt(torch.mean(B ** 2)).detach().cpu()),
        "delta_phi_rms": float(torch.sqrt(torch.mean(delta_phi ** 2)).detach().cpu()),
        "int_S_phi_rms": float(torch.sqrt(torch.mean(int_S_phi ** 2)).detach().cpu()),

        "dt_weak": float(dt_weak),
        "n_samples": int(n_samples),
        "n_quad": int(n_quad),
    }

    return results


def save_results(results, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved weak-form inference results to: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--root", type=str, default="artifacts/manufactured")
    parser.add_argument("--segment_idx", type=int, default=0)

    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--n_quad", type=int, default=8)
    parser.add_argument("--dt_weak", type=float, default=0.025)

    cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cli.root) / cli.run_name
    check_dir = run_dir / "weights"

    print("=" * 80)
    print("WEAK-FORM PARAMETER INFERENCE: m_phi")
    print("=" * 80)
    print(f"Run directory : {run_dir}")
    print(f"Device        : {device}")
    print(f"dt_weak       : {cli.dt_weak}")
    print(f"n_samples     : {cli.n_samples}")
    print(f"n_quad        : {cli.n_quad}")

    args = load_args_from_config(run_dir)
    models = load_models(
        check_dir, 
        args.hidden_layers, 
        args.hidden_dim, 
        device
    )

    inference_results = infer_mphi_weak_local(
        models=models,
        args=args,
        device=device,
        n_samples=cli.n_samples,
        n_quad=cli.n_quad,
        dt_weak=cli.dt_weak,
    )

    print("\nRESULTS")
    print("=" * 80)
    for key, value in inference_results.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.8e}")
        else:
            print(f"{key:35s}: {value}")

    out_path = run_dir / "figures" / f"weak_form_mphi_inference_dt{cli.dt_weak:.4f}.json"
    save_results(inference_results, out_path)


if __name__ == "__main__":
    main()





