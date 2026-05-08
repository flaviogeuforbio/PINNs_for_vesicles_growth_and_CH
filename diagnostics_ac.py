import torch 

from utils import compute_energy, compute_meanphi
from models import AllenCahnPINN

#function to compute mass/energy plots vs reference times (check the physics behind the output of the net)
def compute_energy_diagnostics(model, times, L, epsilon, n_grid, device):
    model.eval()

    #initiating dictionary with diagnostics results 
    results = {
        "times": [],
        "energy": [],
        "mean_phi": [],
        "phi_min": [],
        "phi_max": []
    }

    for time_value in times:
        x = torch.linspace(0.0, L, n_grid).reshape(-1, 1).to(device) #creating the discrete grid
        x.requires_grad_(True)

        t = torch.full_like(x, float(time_value)).to(device) #to pass (x, t) format data to the net

        phi = model(x, t) #calculating phield prediction

        phi_x = torch.autograd.grad(
            phi, x, 
            grad_outputs = torch.ones_like(phi),
            retain_graph = True, 
            create_graph = True
        )[0]

        #compute mass and energy
        energy = compute_energy(x, phi, phi_x, epsilon)
        mean_phi = compute_meanphi(x, phi)

        results["times"].append(float(time_value))
        results["energy"].append(energy.detach().cpu().item())
        results["mean_phi"].append(mean_phi.detach().cpu().item())
        results["phi_min"].append(phi.min().detach().cpu().item())
        results["phi_max"].append(phi.max().detach().cpu().item())

    return results


#function to load the PINN best model
def load_model(model_checkpoint: str, hidden_layers: int = 4, hidden_dim: int = 128):
    model = AllenCahnPINN(hidden_layers = hidden_layers, hidden_dim = hidden_dim)
    model.load_state_dict(torch.load(model_checkpoint))

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_checkpoint", type=str, required = True)
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--epsilon", type=float, default=0.05, help = "Interface penalty term epsilon")
    
    args = parser.parse_args()

    T_max = args.tmax #simulation end time
    L = 1.0 #box dimension
    epsilon = args.epsilon #interface penalty term

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    model_checkpoint = "/".join(["artifacts", args.model_checkpoint])
    model = load_model(model_checkpoint = model_checkpoint)
    model = model.to(device)
    diagnostics_times = torch.linspace(0.0, T_max, 11)

    diagnostics = compute_energy_diagnostics(
        model = model, 
        times = diagnostics_times, 
        L = L, 
        epsilon = epsilon,
        n_grid = 512,
        device = device
    )


    print("\nDIAGNOSTICHE FISICHE")
    print("=" * 80)

    e0 = diagnostics["energy"][0]
    m0 = diagnostics["mean_phi"][0]
    # a0 = diagnostics["mode1_amp"][0]

    for t, e, m, phimin, phimax in zip(
        diagnostics["times"],
        diagnostics["energy"],
        diagnostics["mean_phi"],
        diagnostics["phi_min"],
        diagnostics["phi_max"],
    ):
        print(
            f"t={t:.3f} | "
            f"energy={e:.8e} | ΔE={e-e0:+.2e} | "
            f"mean_phi={m:.8e} | Δm={m-m0:+.2e} | "
            f"phi∈[{phimin:+.4f}, {phimax:+.4f}]"
        )