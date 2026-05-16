import torch 

from utils_coupled_2d import grad, integral_2d, compute_energy
from models import CoupledACCHPINN2d

#function to pars data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=False, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    parser.add_argument("--checkpoint_name", type=str, required=True, help = "Model checkpoint name")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    # parser.add_argument("--segment_length", type=float, default=0.5, help = "Time segment dimension of each network in the ensemble (time windowing)")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
    parser.add_argument("--eps_phi", type=float, default=0.05, help = "Interface penalty term epsilon for phi")
    parser.add_argument("--eps_c", type=float, default=0.05, help = "Interface penalty term epsilon for c")
    parser.add_argument("--gamma", type=float, default=0.05, help = "Coupling parameter (int = gamma * phi * c)")
    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of PINN hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons for each PINN hidden layers")
    parser.add_argument("--n_grid", type=int, default=128, help = "(n_grid x n_grid): dimension of the discrete grid needed to compute mass, energy... for diagnostics")

    args = parser.parse_args()

    return args

#function to evaluate diagnostics in baseline setup
def evaluate_diagnostics(
    model,
    times,
    L_x, L_y,
    eps_phi,
    eps_c,
    gamma,
    n_grid,
    device="cpu",
):
    model.eval()

    results = {
        "times": [],
        "energy": [],
        "mass_c": [],
        "mean_phi": [],
        "phi_min": [],
        "phi_max": [],
        "c_min": [],
        "c_max": [],
    }

    x_lin = torch.linspace(0.0, L_x, n_grid, device=device)
    y_lin = torch.linspace(0.0, L_y, n_grid, device=device)

    #creating the discrete 2-dimensional grid
    X, Y = torch.meshgrid(x_lin, y_lin, indexing = "ij")

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    for time_value in times:
        x = x_flat.clone().detach().requires_grad_(True)
        y = y_flat.clone().detach().requires_grad_(True)

        t = torch.full_like(x, float(time_value), device=device)

        phi, c, mu = model(x, y, t)

        phi_x = grad(phi, x)
        phi_y = grad(phi, y)

        c_x = grad(c, x)
        c_y = grad(c, y)

        energy = compute_energy(
            phi=phi,
            c=c,
            phi_x=phi_x,
            phi_y=phi_y,
            c_x=c_x,
            c_y=c_y,
            x_lin=x_lin, 
            y_lin=y_lin,
            n_grid=n_grid,
            eps_phi=eps_phi,
            eps_c=eps_c,
            gamma=gamma,
        )

        mass_c = integral_2d(c, x_lin, y_lin, n_grid)
        mean_phi = integral_2d(phi, x_lin, y_lin, n_grid)

        results["times"].append(float(time_value))
        results["energy"].append(float(energy.detach().cpu()))
        results["mass_c"].append(float(mass_c.detach().cpu()))
        results["mean_phi"].append(float(mean_phi.detach().cpu()))
        results["phi_min"].append(float(phi.min().detach().cpu()))
        results["phi_max"].append(float(phi.max().detach().cpu()))
        results["c_min"].append(float(c.min().detach().cpu()))
        results["c_max"].append(float(c.max().detach().cpu()))

    return results


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from utils_coupled_2d import load_model

    args = parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    check_dir = Path("artifacts/baseline/weights")
    check_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        model_checkpoint = check_dir / args.checkpoint_name,
        hidden_layers = args.hidden_layers, 
        hidden_dim = args.hidden_dim
    )
    model = model.to(device)
    
    diagnostics_times = torch.linspace(0.0, args.tmax, 11)

    diagnostics = evaluate_diagnostics(
        model = model, 
        times = diagnostics_times, 
        L_x = args.lx, L_y = args.ly, 
        eps_phi = args.eps_phi,
        eps_c = args.eps_c, 
        gamma = args.gamma, 
        n_grid = args.n_grid,
        device = device
    )


    print("\nDIAGNOSTICHE FISICHE")
    print("=" * 80)

    m_c0 = diagnostics["mass_c"][0]
    m_phi0 = diagnostics["mean_phi"][0]
    e0 = diagnostics["energy"][0]
    # a0 = diagnostics["mode1_amp"][0]

    for t, m_c, m_phi, e, cmin, cmax, phimin, phimax in zip(
        diagnostics["times"],
        diagnostics["mass_c"],
        diagnostics["mean_phi"],
        diagnostics["energy"],
        diagnostics["c_min"],
        diagnostics["c_max"],
        diagnostics["phi_min"],
        diagnostics["phi_max"]
    ):
        print(
            f"t={t:.3f} | "
            f"c_mass={m_c:+.8e} | Δmass={m_c-m_c0:+.2e} | "
            f"mean_phi={m_phi:+.8e} | Δmean={m_phi-m_phi0:+.2e} | "
            f"energy={e:.8e} | ΔE={e-e0:+.2e} | "
            f"c∈[{cmin:+.4f}, {cmax:+.4f}]"
            f"phi∈[{phimin:+.4f}, {phimax:+.4f}]"
        )