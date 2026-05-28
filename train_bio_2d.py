import torch 
import argparse
import json

from trainer_bio_2d import train_one_segment 

#function to parse data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of hidden layers in ACCH PINN")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons in each hidden layer in ACCH PINN")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--segment_length", type=float, default=0.5, help = "Segment lenght of each model in simple version of time marching (ensemble of networks)")
    parser.add_argument("--recompute_pot", action="store_true", help = "if called, IC nu,mu potentials for segment 1> are recomputed starting from previous model predictions")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
    parser.add_argument("--eps", type=float, default=0.05, help = "Interface penalty term epsilon for phi")
    parser.add_argument("--m_phi", type=float, default=1.0, help = "Mobility parameter for phi")
    parser.add_argument("--m0", type=float, default=0.5, help = "Parameter in psi mobility formula")
    parser.add_argument("--lambda_surf", type=float, default=1.0, help = "Surface free energy weight")
    parser.add_argument("--lambda_in", type=float, default=1.0, help = "Constant factor in formula for phi free energy inside the vesicle")
    parser.add_argument("--lambda_out", type=float, default=1.0, help = "Constant factor in formula for phi free energy outside the vesicle")
    parser.add_argument("--beta_in", type=float, default=0.0, help = "Additive constant in formula for phi free energy inside the vesicle")
    parser.add_argument("--beta_out", type=float, default=0.0, help = "Additive constant in formula for phi free energy outside the vesicle")
    parser.add_argument("--psi_in_eq", type=float, default=1.0, help = "Equilibrium value for psi inside the vesicle")
    parser.add_argument("--psi_out_eq", type=float, default=0.0, help = "Equilibrium value for psi outside the vesicle")
    parser.add_argument("--psi_in_0", type=float, default=0.6, help = "IC for psi inside the vesicle (homogeneous concentration)")
    parser.add_argument("--psi_out_0", type=float, default=0.2, help = "IC for psi outside the vesicle (homogeneous concentration)")
    parser.add_argument("--radius", type=float, default=0.28, help = "Radius of the (circular) IC for the vesicle (phi field)")
    parser.add_argument("--x0", type=float, default=0.5, help = "x coordinate of the center of the vesicle IC (phi field)")
    parser.add_argument("--y0", type=float, default=0.5, help = "y coordinate of the center of the vesicle IC (phi field)")
    parser.add_argument("--pde_weight", type=float, default=1.0, help = "PDE loss term weight")
    parser.add_argument("--bc_weight", type=float, default=10.0, help = "BC loss term weight")
    parser.add_argument("--ic_weight", type=float, default=100.0, help = "IC loss term weight")
    parser.add_argument("--pde_phi_w", type=float, default=1.0, help = "PDE (phi) loss term weight")
    parser.add_argument("--pde_mu_w", type=float, default=1.0, help = "PDE (mu) loss term weight")
    parser.add_argument("--pde_psi_w", type=float, default=1.0, help = "PDE (psi) loss term weight")
    parser.add_argument("--pde_nu_w", type=float, default=1.0, help = "PDE (nu) loss term weight")
    parser.add_argument("--epochs", type=int, default=3000, help = "N. of training epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=0, help = "N. of pre-training epochs (IC loss only)")
    parser.add_argument("--lr", type=float, default=1e-3, help = "Adam learning rate")
    parser.add_argument("--lbfgs_iter", type=int, default=100, help = "N. of L-BFGS iterations")
    parser.add_argument("--n_pde", type=int, default=10000, help = "N. of PDE collocation points")
    parser.add_argument("--n_bc", type=int, default=2000, help = "N. of BC collocation points")
    parser.add_argument("--n_ic", type=int, default=2000, help = "N. of IC collocation points")
    parser.add_argument("--adaptive_sampling", action="store_true", help = "if True, two-stage adaptive resampling of collocation points is activated")
    parser.add_argument("--adap_warmup_epochs", type=int, default=400, help = "N. of pre-adaptation epochs in training before performing adaptive resampling of coll. points")
    parser.add_argument("--n_candidates_resamp", type=int, default=30000, help = "N. of candidate points generated in adaptive resampling phase")
    parser.add_argument("--adaptive_frac", type=float, default=0.7, help = "Fraction of adaptive resampled points in total resampled points (adaptive + uniform)")
    parser.add_argument("--run_name", type=str, required=True, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)") 
    
    #manufactured solution
    parser.add_argument("--manufactured", action="store_true", help = "if True, a manufactured solution is used to validate the model (PDE forced with external sources)")
    parser.add_argument("--ms_smooth", action="store_true", help = "if True, a simpler smooth manufactured solution is used (instead of the phase-field-like solution)")
    parser.add_argument("--ms_R0", type=float, default=0.25, help = "Initial vesicle radius (IC of phase-field-like manufactured solution)")
    parser.add_argument("--ms_alpha_R", type=float, default=0.3, help = "Growth rate of vesicle radius in phase-field-like manufactured solution")
    parser.add_argument("--ms_psi_in0", type=float, default=0.3, help = "Initial concentration psi inside the vesicle (for phase-field-like manufactured solution)")
    parser.add_argument("--ms_psi_out0", type=float, default=0.8, help = "Initial concentration psi outside the vesicle (for phase-field-like manufactured solution)")
    parser.add_argument("--ms_beta_in", type=float, default=0.2, help = "Growth rate of concentration psi inside the vesicle (for phase-field-like manufactured solution)")
    parser.add_argument("--ms_beta_out", type=float, default=0.0, help = "Growth rate of concentration psi outside the vesicle (for phase-field-like manufactured solution)")

    #inverse problem
    parser.add_argument("--inverse_m_phi", action="store_true", help = "if True, the network is trained to solve the inverse problem for m_phi parameter")
    parser.add_argument("--m_phi_lr", type=float, default=5e-3, help = "Learning rate for m_phi parameter")
    parser.add_argument("--m_phi_init", type=float, default=0.3, help = "Initial value (guess) for m_phi in inverse problem, target value is args.m_phi")
    parser.add_argument("--data_weight", type=float, default=10.0, help = "Data loss term weight")
    parser.add_argument("--n_data", type=int, default=5000, help = "N. of data collocation points")
    parser.add_argument("--data_noise", type=float, default=0.0, help = "...")

    return parser.parse_args()

#function to save all CLI arguments used for the current run
def save_run_config(args, run_dir):
    config_path = run_dir / "run_config.json"
    config = vars(args).copy()

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Run configuration saved to: {config_path}")

if __name__ == "__main__":
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from utils_bio_2d import make_ic_from_previous_model

    #parse data from CLI
    args = parse_args()

    #setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #calculating number of networks
    n_segments = int(math.ceil(args.tmax / args.segment_length))
    print(f"T_max = {args.tmax}")
    print(f"segment_length = {args.segment_length}")
    print(f"n_segments = {n_segments}")

    #setting directories to save output data
    out_dir = Path("artifacts/manufactured") / args.run_name if getattr(args, "manufactured", False) else Path("artifacts/bio-minimal") / args.run_name
    weights_dir = out_dir / "weights"
    lossplots_dir = out_dir / "figures"

    weights_dir.mkdir(parents=True, exist_ok=True)
    lossplots_dir.mkdir(parents=True, exist_ok=True)

    models = []
    all_histories = []

    ic_fn = None

    #creating trainable parameter for inverse problem
    if getattr(args, "inverse_m_phi", False):
        print("=" * 80)
        print("INVERSE PROBLEM: inferring m_phi")
        print(f"True m_phi   = {args.m_phi}")
        print(f"Initial guess = {args.m_phi_init}")
        print("=" * 80)

        log_m_phi = torch.nn.Parameter(
            torch.tensor(np.log(args.m_phi_init), dtype=torch.float32, device = device)
        )

    else: 
        log_m_phi = None

    for segment_idx in range(n_segments):
        #training the single segment
        model, train_losses, pretrain_losses, lbfgs_losses = train_one_segment(
            segment_idx, 
            ic_fn, 
            args, 
            device,
            log_m_phi
        )

        #updating models and histories array
        models.append(model)
        all_histories.append(
            {
                "train_losses": train_losses, 
                "pretrain_losses": pretrain_losses, 
                "lbfgs_losses": lbfgs_losses
            }
        )

        #saving single segment model weights
        torch.save(
            model.state_dict(), 
            weights_dir / f"segment_{segment_idx}.pt"
        )

        #plot train loss vs epoch
        x_epochs = np.arange(1, args.epochs + 1)

        plt.figure(figsize=(8, 5))

        plt.semilogy(x_epochs, train_losses["pde_phi"], label="PDE phi")
        plt.semilogy(x_epochs, train_losses["pde_mu"], label="PDE mu")
        plt.semilogy(x_epochs, train_losses["pde_psi"], label = "PDE psi")
        plt.semilogy(x_epochs, train_losses["pde_nu"], label = "PDE nu")
        plt.semilogy(x_epochs, train_losses["bc"], label="BC")
        plt.semilogy(x_epochs, train_losses["ic"], label="IC")

        plt.xlabel("Epochs")
        plt.ylabel("Train loss")
        plt.title(f"Bio AC-CH 2D training losses (segment {segment_idx})")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.savefig(lossplots_dir / f"segment_{segment_idx}_lossplot.png", dpi=200, bbox_inches="tight")
        plt.close()

        #plot learned m_phi vs epoch (for inverse problem)
        if getattr(args, "inverse_m_phi", False) and len(train_losses["m_phi"]) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(1, len(train_losses["m_phi"]) + 1), train_losses["m_phi"], label=r"learned $m_\phi$")
            plt.axhline(args.m_phi, linestyle="--", color="black", label=r"true $m_\phi$")
            plt.xlabel("Epoch")
            plt.ylabel(r"$m_\phi$")
            plt.title(f"Inverse parameter inference: segment {segment_idx}")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.savefig(lossplots_dir / f"segment_{segment_idx}_m_phi_convergence.png", dpi=200, bbox_inches="tight")
            plt.close()

        #creating new ic for next segment model using current model predictions
        ic_fn = make_ic_from_previous_model(
            previous_model = model, 
            segment_length = args.segment_length, 
            device = device,
            args = args, 
            recompute_potentials = args.recompute_pot
        )

    #save run configuration 
    save_run_config(args, out_dir)