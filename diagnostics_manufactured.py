import torch 
import json
import argparse
import matplotlib.pyplot as plt

from utils_bio_2d import grad, predict_windowed
from manufactured_bio_2d import exact_fields_ms, manufactured_sources

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

    parser.add_argument("--source_diagnostics", action="store_true", help = "if True, it is performed source terms diagnostics too. The idea is to assess how intense are the external sources for the current manufactured solution")

    parser.add_argument("--ms_smooth", action="store_true", help = "if True, a simpler smooth manufactured solution is used (instead of the phase-field-like solution)")
    parser.add_argument("--ms_R0", type=float, default=0.25, help = "Initial vesicle radius (IC of phase-field-like manufactured solution)")
    parser.add_argument("--x0", type=float, default=0.5, help = "x coordinate of the center of the vesicle IC (phi field)")
    parser.add_argument("--y0", type=float, default=0.5, help = "y coordinate of the center of the vesicle IC (phi field)")
    parser.add_argument("--ms_alpha_R", type=float, default=0.3, help = "Growth rate of vesicle radius in phase-field-like manufactured solution")
    parser.add_argument("--ms_psi_in0", type=float, default=0.3, help = "Initial concentration psi inside the vesicle (for phase-field-like manufactured solution)")
    parser.add_argument("--ms_psi_out0", type=float, default=0.8, help = "Initial concentration psi outside the vesicle (for phase-field-like manufactured solution)")
    parser.add_argument("--ms_beta_in", type=float, default=0.2, help = "Growth rate of concentration psi inside the vesicle (for phase-field-like manufactured solution)")
    parser.add_argument("--ms_beta_out", type=float, default=0.0, help = "Growth rate of concentration psi outside the vesicle (for phase-field-like manufactured solution)")

    args = parser.parse_args()

    return args

#function to compute mean squared error
def mse(pred, true): 
    return torch.mean((pred - true) ** 2)

#function to compute root mean squared error
def rmse(u):
    return torch.sqrt(torch.mean(u ** 2))

#function to compute relative error L2
def relative_l2(pred, true, eps=1e-12):
    numerator = torch.sqrt(torch.mean((pred - true) ** 2))
    denominator = torch.sqrt(torch.mean(true ** 2)) + eps

    return numerator / denominator

#function to evaluate the diagnostics in manufactured solutions configuration 
def evaluate_err_diagnostics(
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

        
    return results, x_flat, y_flat

#function to evaluate the diagnostics for external source terms. The idea is to check how 'far' is the 
#forced PDE from the original one (minimal biological model)
def evaluate_source_diagnostics(x_flat, y_flat, times, args):

    results = {
        "times": [],
        "rmse_phi_t": [], 
        "rmse_m_phi_mu": [], 
        "rmse_S_phi": [], 

        "rmse_psi_t": [],
        "rmse_div_J": [], 
        "rmse_S_psi": [],

        "rho_phi": [], 
        "rho_psi": []
    }

    for time_value in times:
        t = torch.full_like(x_flat, float(time_value), device=x_flat.device).requires_grad_(True)
        
        x = x_flat.clone().detach().requires_grad_(True)
        y = y_flat.clone().detach().requires_grad_(True)

        phi_ex, mu_ex, psi_ex, nu_ex = exact_fields_ms(x, y, t, args) #calculate exact fields

        #derivatives and useful quantities
        phi_t = grad(phi_ex, t)
        psi_t = grad(psi_ex, t)

        nu_x = grad(nu_ex, x)
        nu_y = grad(nu_ex, y)

        M_psi = 1.0 - args.m0 * ((phi_ex ** 2 - 1.0) ** 2)

        J_x = -M_psi * nu_x
        J_y = -M_psi * nu_y

        div_J = grad(J_x, x) + grad(J_y, y)

        #calculating source terms
        S_phi = phi_t + args.m_phi * mu_ex
        S_psi = psi_t + div_J

        #updating results
        results["times"].append(float(time_value))
        
        results["rmse_phi_t"].append(rmse(phi_t).item())
        results["rmse_m_phi_mu"].append(rmse(args.m_phi * mu_ex).item())
        results["rmse_S_phi"].append(rmse(S_phi).item())

        results["rmse_psi_t"].append(rmse(psi_t).item())
        results["rmse_div_J"].append(rmse(div_J).item())
        results["rmse_S_psi"].append(rmse(S_psi).item())

        #adding pho value (normalized S_phi/S_psi rmse)
        eps = 1e-12

        rho_phi = rmse(S_phi) / (rmse(phi_t) + rmse(args.m_phi * mu_ex) + eps)
        rho_psi = rmse(S_psi) / (rmse(psi_t) + rmse(div_J) + eps)

        results["rho_phi"].append(rho_phi.item())
        results["rho_psi"].append(rho_psi.item())

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

#function to plot error diagnostics (predicted fields vs exact manufactured fields)
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

#function to plot a spatial maps of the manufactured source terms magnitude and relative magnitude (point-wise)
def plot_source_maps(x_flat, y_flat, n_grid, args, out_dir, time_value=None):
    if time_value is None:
        time_value = args.tmax

    device = x_flat.device

    x = x_flat.clone().detach().requires_grad_(True)
    y = y_flat.clone().detach().requires_grad_(True)
    t = torch.full_like(x, float(time_value), device=device).requires_grad_(True)

    phi_ex, mu_ex, psi_ex, nu_ex = exact_fields_ms(x, y, t, args)

    phi_t = grad(phi_ex, t)
    psi_t = grad(psi_ex, t)

    nu_x = grad(nu_ex, x)
    nu_y = grad(nu_ex, y)

    M_psi = 1.0 - args.m0 * ((phi_ex ** 2 - 1.0) ** 2)

    J_x = -M_psi * nu_x
    J_y = -M_psi * nu_y

    div_J = grad(J_x, x) + grad(J_y, y)

    S_phi = phi_t + args.m_phi * mu_ex
    S_psi = psi_t + div_J

    eps = 1e-12

    rho_phi_pointwise = torch.abs(S_phi) / (
        torch.abs(phi_t) + torch.abs(args.m_phi * mu_ex) + eps
    )

    rho_psi_pointwise = torch.abs(S_psi) / (
        torch.abs(psi_t) + torch.abs(div_J) + eps
    )

    def to_grid(u):
        return u.detach().cpu().reshape(n_grid, n_grid).T

    maps = {
        r"$\log_{10}(|S_\phi|)$": torch.log10(torch.abs(S_phi) + eps),
        r"$\log_{10}(|S_\psi|)$": torch.log10(torch.abs(S_psi) + eps),
        r"$\rho_\phi(x,y)$": rho_phi_pointwise,
        r"$\rho_\psi(x,y)$": rho_psi_pointwise,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, (title, field) in zip(axes.ravel(), maps.items()):
        im = ax.imshow(
            to_grid(field),
            origin="lower",
            extent=[0.0, args.lx, 0.0, args.ly],
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    fig.suptitle(f"Manufactured source maps at t={float(time_value):.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / f"source_maps_t{float(time_value):.3f}.png", dpi=200)
    plt.close()

#function to plot source terms diagnostics (to assess how strong is the external forcing over the original PDE)
def plot_source_diag(results, out_dir):
    times = results["times"]

    plt.figure(figsize=(8, 5))
    plt.semilogy(times, results["rho_phi"], label=r"$\rho_{\phi}$")
    plt.semilogy(times, results["rho_psi"], label=r"$\rho_{\psi}$")
    plt.xlabel("time")
    plt.title("Normalized sources $S_{\phi}$,$S_{\psi}$ magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "normalized_source_magnitude.png", dpi=200)
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

    error_diagnostics, x_flat, y_flat = evaluate_err_diagnostics(
        models = models, 
        times = diagnostics_times,  
        n_grid = args.n_grid,
        args = args, 
        device = device
    )

    if getattr(args, "source_diagnostics", False):
        source_diagnostics = evaluate_source_diagnostics(x_flat, y_flat, diagnostics_times, args)

        plot_source_diag(source_diagnostics, out_dir)

        plot_source_maps(
            x_flat, 
            y_flat, 
            args.n_grid, 
            args, 
            out_dir, 
            time_value = args.tmax
        )

    plot_errors(error_diagnostics, out_dir)
    print(f"Saved diagnostics to: {out_dir}")