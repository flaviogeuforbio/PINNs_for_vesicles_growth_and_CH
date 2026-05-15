import time
import argparse
import torch 
from torch.optim import Adam 

from models import CoupledACCHPINN
from train_coupled import parse_args, train_model, train_lbfgs
from utils_coupled import generate_coll_points_and_ic

#function to parse data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of hidden layers in ACCH PINN")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons in each hidden layer in ACCH PINN")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--segment_length", type=float, default=0.5, help = "Segment lenght of each model in simple version of time marching (ensemble of networks)")
    parser.add_argument("--l", type=float, default=1.0, help = "Box dimension (lenght in 1d)")
    parser.add_argument("--eps_phi", type=float, default=0.05, help = "Interface penalty term epsilon for phi")
    parser.add_argument("--eps_c", type=float, default=0.05, help = "Interface penalty term epsilon for c")
    parser.add_argument("--m_phi", type=float, default=1.0, help = "Mobility parameter for phi")
    parser.add_argument("--m_c", type=float, default=0.1, help = "Mobility parameter for c")
    parser.add_argument("--gamma", type=float, default=0.05, help = "Coupling parameter (int = gamma * c * phi)")
    parser.add_argument("--pde_weight", type=float, default=1.0, help = "PDE loss term weight")
    parser.add_argument("--bc_weight", type=float, default=10.0, help = "BC loss term weight")
    parser.add_argument("--ic_weight", type=float, default=100.0, help = "IC loss term weight")
    parser.add_argument("--pde_phi_w", type=float, default=1.0, help = "PDE (phi) loss term weight")
    parser.add_argument("--pde_c_w", type=float, default=2.0, help = "PDE (c) loss term weight")
    parser.add_argument("--pde_mu_w", type=float, default=2.0, help = "PDE (mu) loss term weight")
    parser.add_argument("--epochs", type=int, default=3000, help = "N. of training epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=0, help = "N. of pre-training epochs (IC loss only)")
    parser.add_argument("--lr", type=float, default=1e-3, help = "Adam learning rate")
    parser.add_argument("--lbfgs_iter", type=int, default=100, help = "N. of L-BFGS iterations")
    parser.add_argument("--n_pde", type=int, default=10000, help = "N. of PDE collocation points")
    parser.add_argument("--n_bc", type=int, default=2000, help = "N. of BC collocation points")
    parser.add_argument("--n_ic", type=int, default=2000, help = "N. of IC collocation points")
    parser.add_argument("--run_name", type=str, required=True, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    
    return parser.parse_args()

def train_one_segment(
        segment_idx, 
        ic_fn, 
        args, 
        device
):
    print("\n" + "=" * 80)
    print(f"TRAINING SEGMENT {segment_idx}")
    print(f"Local time window: tau in [{segment_idx * args.segment_length}, {(segment_idx + 1) * args.segment_length}]")
    print("=" * 80)

    #creating model and optimizer
    model = CoupledACCHPINN(
        hidden_layers = args.hidden_layers, 
        hidden_dim = args.hidden_dim
    ).to(device)

    optimizer = Adam(model.parameters(), lr = args.lr) 

    #generate collocation points + ic (true or from previous model)
    collocation, phi_ic_true, c_ic_true, mu_ic_true = generate_coll_points_and_ic(
        N_pde = args.n_pde,
        N_bc = args.n_bc, 
        N_ic = args.n_ic, 
        L = args.l, 
        T_max = args.segment_length, 
        eps_c = args.eps_c, 
        gamma = args.gamma, 
        device = device, 
        ic_fn = ic_fn
    )

    #start the training phase (main loop with Adam + refinement with L-BFGS algorithm)
    start_time = time.time()
    train_losses, pretrain_losses = train_model(
        model, 
        optimizer, 
        args.m_phi, args.m_c, 
        args.eps_phi, args.eps_c,
        args.gamma, 
        collocation,
        phi_ic_true, 
        c_ic_true, 
        mu_ic_true,
        args.epochs,
        pretrain_epochs = args.pretrain_epochs,
        ic_weight = args.ic_weight, 
        bc_weight = args.bc_weight, 
        pde_weight = args.pde_weight,
        pde_phi_w = args.pde_phi_w, 
        pde_c_w = args.pde_c_w, 
        pde_mu_w = args.pde_mu_w
    )

    #L-BFGS
    if args.lbfgs_iter > 0:
        lbfgs_losses = train_lbfgs(
            model=model,
            collocation=collocation,
            phi_ic_true=phi_ic_true,
            c_ic_true=c_ic_true,
            mu_ic_true=mu_ic_true,
            M_phi=args.m_phi, M_c=args.m_c,
            eps_phi=args.eps_phi, eps_c=args.eps_c,
            gamma=args.gamma,
            max_iter=args.lbfgs_iter, 
            ic_weight=args.ic_weight,
            bc_weight=args.bc_weight,
            pde_weight=args.pde_weight,
        )
    print(f"Tempo di esecuzione: {time.time() - start_time:.2f}s")

    return model, train_losses, pretrain_losses, lbfgs_losses


if __name__ == "__main__":
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from utils_coupled import make_ic_from_previous_model

    #parse data from CLI
    args = parse_args()

    #setting device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #calculating number of networks
    n_segments = int(math.ceil(args.tmax / args.segment_length))
    print(f"T_max = {args.tmax}")
    print(f"segment_length = {args.segment_length}")
    print(f"n_segments = {n_segments}")

    #setting directories to save output data
    out_dir = Path("artifacts/coupled_tw") / args.run_name
    weights_dir = out_dir / "weights"
    lossplots_dir = out_dir / "figures"

    weights_dir.mkdir(parents=True, exist_ok=True)
    lossplots_dir.mkdir(parents=True, exist_ok=True)

    models = []
    all_histories = []

    ic_fn = None

    for segment_idx in range(n_segments):
        #training the single segment
        model, train_losses, pretrain_losses, lbfgs_losses = train_one_segment(
            segment_idx, 
            ic_fn, 
            args, 
            device
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
        plt.semilogy(x_epochs, train_losses["pde_c"], label="PDE c")
        plt.semilogy(x_epochs, train_losses["pde_mu"], label="PDE mu")
        plt.semilogy(x_epochs, train_losses["bc"], label="BC")
        plt.semilogy(x_epochs, train_losses["ic"], label="IC")

        plt.xlabel("Epochs")
        plt.ylabel("Train loss")
        plt.title("Coupled AC-CH 1D (t.w.) training losses")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.savefig(lossplots_dir / f"segment_{segment_idx}_lossplot.png", dpi=200, bbox_inches="tight")
        plt.close()

        #creating new ic for next segment model using current model predictions
        ic_fn = make_ic_from_previous_model(
            previous_model = model, 
            segment_length = args.segment_length, 
            device = device
        )

        