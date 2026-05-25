import torch 
import json
import argparse
import matplotlib.pyplot as plt

from utils_bio_2d import grad, predict_windowed
from manufactured_bio_2d import exact_fields_ms

#function to pars data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--segment_length", type=float, default=0.5, help = "Time segment dimension of each network in the ensemble (time windowing)")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
    
    #--free energy parameters
    parser.add_argument("--eps", type=float, default=0.05, help = "Interface penalty parameter (for phi field)")
    parser.add_argument("--m_phi", type=float, default=1.0, help = "Mobility parameter for phi")
    parser.add_argument("--m0", type=float, default=0.5, help = "Parameter in psi mobility formula")
    parser.add_argument("--lambda_surf", type=float, default=1.0, help = "Surface free energy weight")
    parser.add_argument("--lambda_in", type=float, default=1.0, help = "Constant factor in formula for phi free energy inside the vesicle")
    parser.add_argument("--lambda_out", type=float, default=1.0, help = "Constant factor in formula for phi free energy outside the vesicle")
    parser.add_argument("--beta_in", type=float, default=0.0, help = "Additive constant in formula for phi free energy inside the vesicle")
    parser.add_argument("--beta_out", type=float, default=0.0, help = "Additive constant in formula for phi free energy outside the vesicle")
    parser.add_argument("--psi_in_eq", type=float, default=1.0, help = "Equilibrium value for psi inside the vesicle")
    parser.add_argument("--psi_out_eq", type=float, default=0.0, help = "Equilibrium value for psi outside the vesicle")

    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of PINN hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons for each PINN hidden layers")
    parser.add_argument("--n_grid", type=int, default=128, help = "(n_grid x n_grid): dimension of the discrete grid needed to compute mass, energy... for diagnostics")

    args = parser.parse_args()

    return args

#function to compute mean squared error
def mse(pred, true): 
    return torch.mean((pred - true) ** 2)

#function to compute relative error L2
def relative_l2(pred, true, eps=1e-12):
    numerator = torch.sqrt(torch.mean((pred - true) ** 2))
    denominator = torch.sqrt(torch.mean(true ** 2)) + eps

    return numerator / denominator

#function to perform the diagnostics in manufactured solutions configuration 
def evaluate_diagnostics_ms(
    models, 
    times, 
    n_grid,
    args, 
    device = "cpu"
):
    
    results = {
        "times": [], 
        "mse_phi": [], 
        "mse_mu": [], 
        "mse_psi": [], 
        "mse_nu": [], 
        "rel_l2_phi": [], 
        "rel_l2_mu": [], 
        "rel_l2_psi": [], 
        "rel_l2_nu": []
    }

    x_lin = torch.linspace(0.0, args.lx, n_grid, device=device)
    y_lin = torch.linspace(0.0, args.ly, n_grid, device=device)

    #creating the discrete 2-dimensional grid
    X, Y = torch.meshgrid(x_lin, y_lin, indexing = "ij")

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    for time_value in times:
        x = x_flat.clone().detach().requires_grad_(True)
        y = y_flat.clone().detach().requires_grad_(True)

        #calculate fields predictions
        phi_pred, mu_pred, psi_pred, nu_pred = predict_windowed(models, x, y, float(time_value), args.segment_length, device)

        #calculate exact fields
        t = torch.full_like(x, float(time_value), device = device).requires_grad_(True)
        phi_ex, mu_ex, psi_ex, nu_ex = exact_fields_ms(x, y, t, args)

        #updating results
        results["times"].append(float(time_value))
        results["mse_phi"].append(float(mse(phi_pred, phi_ex).detach().cpu()))
        results["mse_mu"].append(float(mse(mu_pred, mu_ex).detach().cpu()))
        results["mse_psi"].append(float(mse(psi_pred, psi_ex).detach().cpu()))
        results["mse_nu"].append(float(mse(nu_pred, nu_ex).detach().cpu()))
        results["rel_l2_phi"].append(float(relative_l2(phi_pred, phi_ex).detach().cpu()))
        results["rel_l2_mu"].append(float(relative_l2(mu_pred, mu_ex).detach().cpu()))
        results["rel_l2_psi"].append(float(relative_l2(psi_pred, psi_ex).detach().cpu()))
        results["rel_l2_nu"].append(float(relative_l2(nu_pred, nu_ex).detach().cpu()))

        
    return results

#function to load args from run_config.json (generated at the end of the training)
def update_args_from_run_config(args):
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
    print(f"  tmax          = {args.tmax}")
    print(f"  segment_length= {args.segment_length}")
    print(f"  eps           = {args.eps}")
    print(f"  m_phi         = {args.m_phi}")
    print(f"  m0            = {args.m0}")
    print(f"  psi_in_eq     = {args.psi_in_eq}")
    print(f"  psi_out_eq    = {args.psi_out_eq}")
    print(f"  hidden_layers = {args.hidden_layers}")
    print(f"  hidden_dim    = {args.hidden_dim}")

    return args


def plot_errors(results, out_dir):
    times = results["times"]

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["rel_l2_phi"], label=r"$\phi$")
    plt.semilogy(times, results["rel_l2_psi"], label=r"$\psi$")
    plt.semilogy(times, results["rel_l2_mu"], label=r"$\mu$")
    plt.semilogy(times, results["rel_l2_nu"], label=r"$\nu$")
    plt.xlabel("time")
    plt.ylabel("relative L2 error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "relative_l2_errors.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["mse_phi"], label=r"$\phi$")
    plt.semilogy(times, results["mse_psi"], label=r"$\psi$")
    plt.semilogy(times, results["mse_mu"], label=r"$\mu$")
    plt.semilogy(times, results["mse_nu"], label=r"$\nu$")
    plt.xlabel("time")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mse_errors.png", dpi=200)
    plt.close()



if __name__ == "__main__":
    from pathlib import Path

    from utils_bio_2d import load_models

    args = parse_args()
    args = update_args_from_run_config(args)

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    check_dir = Path("artifacts/manufactured") / args.run_name / "weights"

    if not check_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")

    out_dir = Path("artifacts/manufactured") / args.run_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    #loading segment models
    models = load_models(
        check_dir = check_dir, 
        hidden_layers = args.hidden_layers, 
        hidden_dim = args.hidden_dim, 
        device = device
    )
    
    diagnostics_times = torch.linspace(0.0, args.tmax, 11)

    diagnostics = evaluate_diagnostics_ms(
        models = models, 
        times = diagnostics_times,  
        n_grid = args.n_grid,
        args = args, 
        device = device
    )

    plot_errors(diagnostics, out_dir)
    print(f"Saved diagnostics to: {out_dir}")