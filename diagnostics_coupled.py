import torch 

from utils_coupled import compute_integral, compute_energy, predict_windowed
from models import CoupledACCHPINN

#function to pars data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--segment_length", type=float, default=0.5, help = "Time segment dimension of each network in the ensemble (time windowing)")
    parser.add_argument("--eps_phi", type=float, default=0.05, help = "Interface penalty term epsilon for phi")
    parser.add_argument("--eps_c", type=float, default=0.05, help = "Interface penalty term epsilon for c")
    parser.add_argument("--gamma", type=float, default=0.05, help = "Coupling parameter (int = gamma * phi * c)")
    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of PINN hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons for each PINN hidden layers")

    args = parser.parse_args()

    return args


def evaluate_diagnostics(
    model,
    times,
    L,
    eps_phi,
    eps_c,
    gamma,
    n_grid=512,
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

    for time_value in times:
        x = torch.linspace(0.0, L, n_grid, device=device).reshape(-1, 1)
        x.requires_grad_(True)

        t = torch.full_like(x, float(time_value), device=device)

        phi, c, mu = model(x, t)

        phi_x = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]

        c_x = torch.autograd.grad(
            c, x,
            grad_outputs=torch.ones_like(c),
            create_graph=True,
            retain_graph=True,
        )[0]

        energy = compute_energy(
            x=x,
            phi=phi,
            c=c,
            phi_x=phi_x,
            c_x=c_x,
            eps_phi=eps_phi,
            eps_c=eps_c,
            gamma=gamma,
        )

        mass_c = compute_integral(x, c)
        mean_phi = compute_integral(x, phi)

        results["times"].append(float(time_value))
        results["energy"].append(float(energy.detach().cpu()))
        results["mass_c"].append(float(mass_c.detach().cpu()))
        results["mean_phi"].append(float(mean_phi.detach().cpu()))
        results["phi_min"].append(float(phi.min().detach().cpu()))
        results["phi_max"].append(float(phi.max().detach().cpu()))
        results["c_min"].append(float(c.min().detach().cpu()))
        results["c_max"].append(float(c.max().detach().cpu()))

    return results


def evaluate_diagnostics_tw(
    models,
    times,
    L,
    segment_length,
    eps_phi,
    eps_c,
    gamma,
    n_grid=512,
    device="cpu",
):

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

    for time_value in times:
        x = torch.linspace(0.0, L, n_grid, device=device).reshape(-1, 1)
        x.requires_grad_(True)

        phi, c, _ = predict_windowed(models, x, float(time_value), segment_length, device)

        phi_x = torch.autograd.grad(
            phi, x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True,
        )[0]

        c_x = torch.autograd.grad(
            c, x,
            grad_outputs=torch.ones_like(c),
            create_graph=True,
            retain_graph=True,
        )[0]

        energy = compute_energy(
            x=x,
            phi=phi,
            c=c,
            phi_x=phi_x,
            c_x=c_x,
            eps_phi=eps_phi,
            eps_c=eps_c,
            gamma=gamma,
        )

        mass_c = compute_integral(x, c)
        mean_phi = compute_integral(x, phi)

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

    from utils_coupled import load_model, load_models

    args = parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    L = 1.0 #box dimension

    check_dir = Path("artifacts/coupled_tw") / args.run_name / "weights"
    check_dir.mkdir(parents=True, exist_ok=True)

    # #!!!!
    # model = load_model(model_checkpoint = check_dir / args.checkpoint_name)
    # model = model.to(device)
    
    #loading segment models
    models = load_models(
        check_dir = check_dir, 
        hidden_layers = args.hidden_layers, 
        hidden_dim = args.hidden_dim, 
        device = device
    )
    
    diagnostics_times = torch.linspace(0.0, args.tmax, 11)

    diagnostics = evaluate_diagnostics_tw(
        models = models, 
        times = diagnostics_times, 
        L = L, 
        segment_length = args.segment_length,
        eps_phi = args.eps_phi,
        eps_c = args.eps_c, 
        gamma = args.gamma, 
        n_grid = 512,
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