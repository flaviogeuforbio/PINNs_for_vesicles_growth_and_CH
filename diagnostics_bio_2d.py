import torch 

from utils_bio_2d import grad, integral_2d, compute_energy, predict_windowed, p_interp

#function to pars data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=True, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    parser.add_argument("--checkpoint_name", type=str, required=False, help = "Model checkpoint name")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--segment_length", type=float, default=0.5, help = "Time segment dimension of each network in the ensemble (time windowing)")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
    
    #--free energy parameters
    parser.add_argument("--eps", type=float, default=0.05, help = "Interface penalty parameter (for phi field)")
    parser.add_argument("--lambda_surf", type=float, default=1.0, help = "Surface free energy density constant factor")
    parser.add_argument("--lambda_in", type=float, default=1.0, help = "constant factor for free energy inside the vesicle (psi)")
    parser.add_argument("--lambda_out", type=float, default=1.0, help = "constant factor for free energy outside the vesicle (psi)")
    parser.add_argument("--beta_in", type=float, default=0.0)
    parser.add_argument("--beta_out", type=float, default=0.0)
    parser.add_argument("--psi_in_eq", type=float, default=1.0)
    parser.add_argument("--psi_out_eq", type=float, default=0.0)

    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of PINN hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons for each PINN hidden layers")
    parser.add_argument("--n_grid", type=int, default=128, help = "(n_grid x n_grid): dimension of the discrete grid needed to compute mass, energy... for diagnostics")

    args = parser.parse_args()

    return args


#time windowing version
def evaluate_diagnostics_tw(
    models,
    times,
    n_grid,
    args, 
    device="cpu",
):

    results = {
        "times": [],
        "energy": [],
        "surf_energy": [], 
        "osm_energy": [],
        "mass_psi": [],
        "V_in": [],
        "phi_min": [],
        "phi_max": [],
        "psi_min": [],
        "psi_max": [],
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

        phi, _, psi, _ = predict_windowed(models, x, y, float(time_value), args.segment_length, device)

        phi_x = grad(phi, x)
        phi_y = grad(phi, y)

        #for the moment we neglect area_energy
        energy, surf_energy, osm_energy, _ = compute_energy(
            phi = phi, 
            psi = psi,
            phi_x = phi_x, 
            phi_y = phi_y, 
            x_lin = x_lin, 
            y_lin = y_lin, 
            n_grid = n_grid, 
            args = args
        )

        # mass_psi = integral_2d(psi, x_lin, y_lin, n_grid) 
        mean_phi = integral_2d(phi, x_lin, y_lin, n_grid) #this function requires meshgrid with indexing ij!!

        #using the interpolating function p(phi) to compute a more authentic estimation of the vesicle internal volume (surface in 2d)
        chi_in = 0.5 * (1 + p_interp(phi))
        V_in = integral_2d(chi_in, x_lin, y_lin, n_grid) 

        results["times"].append(float(time_value))
        results["energy"].append(float(energy.detach().cpu()))
        results["surf_energy"].append(float(surf_energy.detach().cpu()))
        results["osm_energy"].append(float(osm_energy.detach().cpu()))
        results["V_in"].append(float(V_in.detach().cpu()))
        results["mean_phi"].append(float(mean_phi.detach().cpu()))
        results["phi_min"].append(float(phi.min().detach().cpu()))
        results["phi_max"].append(float(phi.max().detach().cpu()))
        results["psi_min"].append(float(psi.min().detach().cpu()))
        results["psi_max"].append(float(psi.max().detach().cpu()))

    return results


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from utils_bio_2d import load_model, load_models

    args = parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    check_dir = Path("artifacts/timewindowing") / args.run_name / "weights"
    check_dir.mkdir(parents=True, exist_ok=True)

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
        n_grid = args.n_grid,
        args = args, 
        device = device
    )


    print("\nDIAGNOSTICHE FISICHE")
    print("=" * 80)

    m_psi0 = diagnostics["mass_psi"][0]
    v_in0 = diagnostics["V_in"][0]
    e0 = diagnostics["energy"][0]
    esurf0 = diagnostics["surf_energy"][0]
    eosm0 = diagnostics["osm_energy"][0]
    # a0 = diagnostics["mode1_amp"][0]

    for t, m_psi, v_in, e, esurf, eosm, psimin, psimax, phimin, phimax in zip(
        diagnostics["times"],
        diagnostics["mass_psi"],
        diagnostics["V_in"],
        diagnostics["energy"],
        diagnostics["surf_energy"],
        diagnostics["osm_energy"],
        diagnostics["psi_min"],
        diagnostics["psi_max"],
        diagnostics["phi_min"],
        diagnostics["phi_max"]
    ):
        print(
            f"t={t:.3f} | "
            f"psi_mass={m_psi:+.8e} | Δmass={m_psi-m_psi0:+.2e} | "
            f"V_in={v_in:+.8e} | ΔV_in={v_in-v_in0:+.2e} | "
            f"F_tot={e:.8e} | ΔF={e-e0:+.2e} | "
            f"F_surf={esurf:.8e} | ΔF={esurf-esurf0:+.2e} | "
            f"F_osm={eosm:.8e} | ΔF={eosm-eosm0:+.2e} | "
            f"c∈[{psimin:+.4f}, {psimax:+.4f}]"
            f"phi∈[{phimin:+.4f}, {phimax:+.4f}]"
        )