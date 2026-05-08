import torch 

from utils_coupled import compute_integral, compute_energy
from models import CoupledACCHPINN

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

#function to load the PINN best model
def load_model(model_checkpoint: str, hidden_layers: int = 4, hidden_dim: int = 128):
    model = CoupledACCHPINN(hidden_layers = hidden_layers, hidden_dim = hidden_dim)
    model.load_state_dict(torch.load(model_checkpoint))

    return model


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_name", type=str, required = True, help = "File name of model checkpoint used")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--eps_phi", type=float, default=0.05, help = "Interface penalty term epsilon for phi")
    parser.add_argument("--eps_c", type=float, default=0.05, help = "Interface penalty term epsilon for c")
    parser.add_argument("--gamma", type=float, default=0.05, help = "Coupling parameter (int = gamma * phi * c)")

    args = parser.parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    T_max = args.tmax #simulation end time
    L = 1.0 #box dimension
    eps_phi = args.eps_phi #interface penalty term for phi
    eps_c = args.eps_c #interface penalty term for c
    gamma = args.gamma #coupling par.

    check_dir = Path("artifacts/coupled/weights")
    check_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_checkpoint = check_dir / args.checkpoint_name)
    model = model.to(device)
    diagnostics_times = torch.linspace(0.0, T_max, 11)

    diagnostics = evaluate_diagnostics(
        model = model, 
        times = diagnostics_times, 
        L = L, 
        eps_phi = eps_phi,
        eps_c = eps_c, 
        gamma = gamma, 
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