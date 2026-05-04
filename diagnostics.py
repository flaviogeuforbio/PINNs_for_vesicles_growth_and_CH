import torch 

from utils import compute_mass, compute_energy
from models import CahnHilliardPINN

#function to compute mass/energy plots vs reference times (check the physics behind the output of the net)
def compute_mass_energy(model, times, L, epsilon, n_grid = 512):
    model.eval()

    #initiating dictionary with diagnostics results 
    results = {
        "times": [],
        "mass": [],
        "energy": [],
        "c_min": [],
        "c_max": []
    }

    for time_value in times:
        x = torch.linspace(0.0, L, n_grid).reshape(-1, 1) #creating the discrete grid
        x.requires_grad_(True)

        t = torch.full_like(x, float(time_value)) #to pass (x, t) format data to the net

        c, mu = model(x, t) #calculating predictions

        c_x = torch.autograd.grad(
            c, x, 
            grad_outputs = torch.ones_like(c),
            retain_graph = True, 
            create_graph = True
        )[0]

        #compute mass and energy
        mass = compute_mass(x, c)
        energy = compute_energy(x, c, c_x, epsilon)

        results["times"].append(float(time_value))
        results["mass"].append(mass.detach().item())
        results["energy"].append(energy.detach().item())
        results["c_min"].append(c.min().detach().item())
        results["c_max"].append(c.max().detach().item())

    return results


#function to load the PINN best model
def load_model(model_checkpoint: str, hidden_layers: int = 4, hidden_dim: int = 128):
    model = CahnHilliardPINN(hidden_layers = hidden_layers, hidden_dim = hidden_dim)
    model.load_state_dict(torch.load(model_checkpoint))

    return model


if __name__ == "__main__":
    T_max = 2.0 #simulation end time
    L = 1.0 #box dimension
    epsilon = 0.05 #interface penalty term

    model = load_model(model_checkpoint = "artifacts/ch_baseline_lbfgs_tmax2.pt")
    diagnostics_times = torch.linspace(0.0, T_max, 11)

    diagnostics = compute_mass_energy(
        model = model, 
        times = diagnostics_times, 
        L = L, 
        epsilon = epsilon,
        n_grid = 512
    )


    print("\nDIAGNOSTICHE FISICHE")
    print("=" * 80)

    m0 = diagnostics["mass"][0]
    e0 = diagnostics["energy"][0]
    # a0 = diagnostics["mode1_amp"][0]

    for t, m, e, cmin, cmax in zip(
        diagnostics["times"],
        diagnostics["mass"],
        diagnostics["energy"],
        diagnostics["c_min"],
        diagnostics["c_max"],
    ):
        print(
            f"t={t:.3f} | "
            f"mass={m:+.8e} | Δmass={m-m0:+.2e} | "
            f"energy={e:.8e} | ΔE={e-e0:+.2e} | "
            f"c∈[{cmin:+.4f}, {cmax:+.4f}]"
        )