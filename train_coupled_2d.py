from pathlib import Path
import torch
import time
import argparse

from losses_coupled_2d import pde_loss, bc_loss, ic_loss
from utils_coupled_2d import generate_coll_points_and_ic

#function to parse data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of hidden layers in ACCH PINN")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons in each hidden layer in ACCH PINN")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
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
    parser.add_argument("--checkpoint_name", type=str, required=True, help = "Model checkpoint name")
    parser.add_argument("--lossplot_name", type=str, required=True, help = "Training losses plot name")
    
    return parser.parse_args()


#function to train the model for just one iteration 
def train_one_epoch(
        model, 
        collocation,
        phi_ic_true, 
        c_ic_true, 
        mu_ic_true, 
        optimizer, 
        M_phi, M_c, 
        eps_phi, eps_c,
        gamma, #coupling parameter 
        epoch,
        ic_weight, #ic loss term weight
        bc_weight, #bc loss term weight 
        pde_weight, #pde loss term weight (set to 0 in pre-training phase)
        pde_phi_w, pde_c_w, pde_mu_w #weights for each separated pde loss term (phi, c, mu residuals)
):
    model.train()

    #unpacking collocation dictionary to extract all collocation points
    x_pde, y_pde, t_pde = collocation["x_pde"], collocation["y_pde"], collocation["t_pde"]
    x_bc, y_bc, t_bc, normal = collocation["x_bc"], collocation["y_bc"], collocation["t_bc"], collocation["normal_bc"]
    x_ic, y_ic, t_ic = collocation["x_ic"], collocation["y_ic"], collocation["t_ic"]

    optimizer.zero_grad() #zero-ing the gradients

    #calculating the total loss and backpropagating
    loss_pde, loss_pde_phi, loss_pde_c, loss_pde_mu = pde_loss(x_pde, y_pde, t_pde, model, M_phi, M_c, eps_phi, eps_c, gamma)
    loss_bc, loss_bc_phi, loss_bc_c, loss_bc_mu = bc_loss(model, x_bc, y_bc, t_bc, normal)
    loss_ic, loss_ic_phi, loss_ic_c, loss_ic_mu = ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true)

    loss = pde_weight * (pde_phi_w * loss_pde_phi + pde_c_w * loss_pde_c + pde_mu_w * loss_pde_mu) + bc_weight * loss_bc + ic_weight * loss_ic
    loss.backward()

    optimizer.step() #updating the gradients

    #printing results 
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:05d} | Loss PDE (phi): {loss_pde_phi.item():.4e} | Loss PDE (c): {loss_pde_c.item():.4e} | Loss PDE (mu): {loss_pde_mu.item():.4e} | Loss BC: {loss_bc.item():.4e} | Loss IC: {(loss_ic_c + loss_ic_mu).item():.4e}")

    return {
        "total": loss.item(),
        "pde": loss_pde.item(),
        "pde_phi": loss_pde_phi.item(),
        "pde_c": loss_pde_c.item(),
        "pde_mu": loss_pde_mu.item(),
        "bc": loss_bc.item(),
        "bc_phi": loss_bc_phi.item(),
        "bc_c": loss_bc_c.item(),
        "bc_mu": loss_bc_mu.item(),
        "ic": loss_ic.item(),
        "ic_phi": loss_ic_phi.item(),
        "ic_c": loss_ic_c.item(),
        "ic_mu": loss_ic_mu.item()
    }


#function to pre-train the model on IC only 
def pretrain_initial_condition(
        model, 
        collocation, 
        phi_ic_true, 
        c_ic_true, 
        mu_ic_true, 
        optimizer, 
        pretrain_epochs: int
):
    model.train()

    x_ic, y_ic, t_ic = collocation["x_ic"], collocation["y_ic"], collocation["t_ic"]

    #initiating the pre-train loss history
    history = {
        "total": [],
        "ic_phi": [],
        "ic_c": [],
        "ic_mu": []
    }

    for epoch in range(1, pretrain_epochs + 1):
        optimizer.zero_grad()

        #calculating the loss, backprop & updating gradients
        loss_ic, loss_ic_phi, loss_ic_c, loss_ic_mu = ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true)

        loss_ic.backward()
        optimizer.step()

        #updating history
        history["total"].append(loss_ic.item())
        history["ic_phi"].append(loss_ic_phi.item())
        history["ic_c"].append(loss_ic_c.item())
        history["ic_mu"].append(loss_ic_mu.item())

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Pretrain epoch {epoch:05d} | "
                f"IC phi: {loss_ic_phi.item():.4e} | "
                f"IC c: {loss_ic_c.item():.4e} | "
                f"IC mu: {loss_ic_mu.item():.4e} | "
                f"IC total: {loss_ic.item():.4e}"
            )

    return history
    

#function to train the model for a fixed n. of iterations (and return the losses history)
def train_model(
    model, 
    optimizer, 
    M_phi, M_c, 
    eps_phi, eps_c,
    gamma,
    collocation, #collocation points dictionary
    phi_ic_true, #true initial non-cons. field profile (phi(x, 0))
    c_ic_true, #true initial concentration profile (c(x, 0))
    mu_ic_true, #true initial potential profile (mu(x, 0))
    n_epochs: int,
    pretrain_epochs: int,
    ic_weight: float, #ic loss term weight
    bc_weight: float, #bc loss term weight
    pde_weight: float, #pde loss term weight
    pde_phi_w: float, # individual pde terms weight (phi, c, mu residuals) 
    pde_c_w: float,   # ---
    pde_mu_w: float   # ---
):
    #initiate losses history
    train_losses = {
        "total": [],
        "pde": [],
        "pde_phi": [],
        "pde_c": [],
        "pde_mu": [],
        "bc": [],
        "bc_phi": [],
        "bc_c": [],
        "bc_mu": [],
        "ic": [],
        "ic_phi": [],
        "ic_c": [],
        "ic_mu": []
    }

    if pretrain_epochs > 0:
        #PRE-TRAINING PHASE (just IC training, setting pde loss term weight to 0)
        #-------------------------------------
        print("\n" + "="*40)
        print("PRE-TRAINING CONDIZIONE INIZIALE")
        print("="*40)

        #pre-train the model and save losses history on initial conditions 
        pretrain_losses = pretrain_initial_condition(
            model, 
            collocation, 
            phi_ic_true, c_ic_true, mu_ic_true, 
            optimizer, 
            pretrain_epochs
        )
    else:
        pretrain_losses = None

    #TRAINING PHASE 
    #-------------------------------------
    print("\n" + "="*40)
    print("ADAM TRAINING")
    print("="*40)

    for epoch in range(1, n_epochs + 1):

        epoch_losses = train_one_epoch(
            model, 
            collocation,
            phi_ic_true, 
            c_ic_true,
            mu_ic_true,  
            optimizer, 
            M_phi, M_c, 
            eps_phi, eps_c,
            gamma, 
            epoch,
            ic_weight = ic_weight, 
            bc_weight = bc_weight,
            pde_weight = pde_weight,
            pde_phi_w = pde_phi_w, 
            pde_c_w = pde_c_w, 
            pde_mu_w = pde_mu_w
        ) #train one epoch and calculate losses terms

        #update losses history with current values
        train_losses["total"].append(epoch_losses["total"])
        train_losses["pde"].append(epoch_losses["pde"])
        train_losses["pde_phi"].append(epoch_losses["pde_phi"])
        train_losses["pde_c"].append(epoch_losses["pde_c"])
        train_losses["pde_mu"].append(epoch_losses["pde_mu"])
        train_losses["bc"].append(epoch_losses["bc"])
        train_losses["bc_phi"].append(epoch_losses["bc_phi"])
        train_losses["bc_c"].append(epoch_losses["bc_c"])
        train_losses["bc_mu"].append(epoch_losses["bc_mu"])
        train_losses["ic"].append(epoch_losses["ic"])
        train_losses["ic_phi"].append(epoch_losses["ic_phi"]) 
        train_losses["ic_c"].append(epoch_losses["ic_c"])
        train_losses["ic_mu"].append(epoch_losses["ic_mu"])  

    # return pretrain_losses, train_losses
    return train_losses, pretrain_losses


#function to perform the training refinement with l-bfgs algorithm (post-Adam)
def train_lbfgs(
    model,
    collocation,
    phi_ic_true,
    c_ic_true,
    mu_ic_true,
    M_phi, M_c,
    eps_phi, eps_c,
    gamma,
    max_iter: int,
    ic_weight: float,
    bc_weight: float,
    pde_weight: float,
    pde_phi_w: float, 
    pde_c_w: float, 
    pde_mu_w: float 
):
    
    #REFINEMENT PHASE 
    #-------------------------------------
    print("\n" + "="*40)
    print("L-BFGS REFINEMENT TRAINING")
    print("="*40)
    model.train()

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    x_pde, y_pde, t_pde = collocation["x_pde"], collocation["y_pde"], collocation["t_pde"]
    x_bc, y_bc, t_bc, normal = collocation["x_bc"], collocation["y_bc"], collocation["t_bc"], collocation["normal_bc"]
    x_ic, y_ic, t_ic = collocation["x_ic"], collocation["y_ic"], collocation["t_ic"]

    history = {
        "total": [],
        "pde": [],
        "bc": [],
        "ic": [],
    }

    def closure():
        optimizer.zero_grad()

        loss_pde, loss_pde_phi, loss_pde_c, loss_pde_mu = pde_loss(x_pde, y_pde, t_pde, model, M_phi, M_c, eps_phi, eps_c, gamma)
        loss_bc, *_ = bc_loss(model, x_bc, y_bc, t_bc, normal)
        loss_ic, *_ = ic_loss(
            model, x_ic, y_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true
        )

        loss = (
            pde_weight * (pde_phi_w * loss_pde_phi + pde_c_w * loss_pde_c + pde_mu_w * loss_pde_mu)
            + bc_weight * loss_bc
            + ic_weight * loss_ic
        )

        loss.backward()

        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["bc"].append(loss_bc.item())
        history["ic"].append(loss_ic.item())

        return loss

    optimizer.step(closure)

    print("\nL-BFGS refinement completed.")
    print(
        f"Final L-BFGS | "
        f"Total: {history['total'][-1]:.4e} | "
        f"PDE: {history['pde'][-1]:.4e} | "
        f"BC: {history['bc'][-1]:.4e} | "
        f"IC: {history['ic'][-1]:.4e}"
    )

    return history


#function to save the model weights
def save_model(model, file_name: str):
    checkpoint_dir = Path("artifacts/baseline/weights")
    checkpoint_dir.mkdir(parents=True, exist_ok = True) #check if folder exists, if not create it
    checkpoint_path = checkpoint_dir / file_name

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModello salvato con successo in {checkpoint_path}")


if __name__ == "__main__":
    from torch.optim import Adam
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse

    from models import CoupledACCHPINN2d 

    #adding CLI parser
    args = parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)   

    #physical parameters
    L_x = args.lx #box length on x
    L_y = args.ly #box length on y
    T_max = args.tmax #simulation end time
    M_phi = args.m_phi #phi mobility
    M_c = args.m_c #c mobility
    
    eps_phi = args.eps_phi #interface penalty term for phi
    eps_c = args.eps_c #interface penalty term for phi

    gamma = args.gamma

    #number of collocation points 
    N_pde = args.n_pde #inside the domain 
    N_bc = args.n_bc #on the boundaries 
    N_ic = args.n_ic #for initial time

    #create model and set Adam optimizer
    model = CoupledACCHPINN2d(hidden_layers = args.hidden_layers, hidden_dim = args.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr = args.lr)

    n_epochs = args.epochs
    collocation, phi_ic_true, c_ic_true, mu_ic_true = generate_coll_points_and_ic(N_pde, N_bc, N_ic, L_x, L_y, T_max, eps_c, gamma, device) #generate training collocation pts

    start_time = time.time()
    #training the model for n_epochs
    train_losses, pretrain_history = train_model(
        model, 
        optimizer, 
        M_phi, M_c, 
        eps_phi, eps_c,
        gamma, 
        collocation,
        phi_ic_true, 
        c_ic_true, 
        mu_ic_true,
        n_epochs,
        pretrain_epochs = args.pretrain_epochs,
        ic_weight = args.ic_weight, 
        bc_weight = args.bc_weight, 
        pde_weight = args.pde_weight,
        pde_phi_w = args.pde_phi_w, 
        pde_c_w = args.pde_c_w, 
        pde_mu_w = args.pde_mu_w
    )

    #refining the training with lbfgs
    if args.lbfgs_iter > 0:
        lbfgs_losses = train_lbfgs(
            model=model,
            collocation=collocation,
            phi_ic_true=phi_ic_true,
            c_ic_true=c_ic_true,
            mu_ic_true=mu_ic_true,
            M_phi=M_phi, M_c=M_c,
            eps_phi=eps_phi, eps_c=eps_c,
            gamma=gamma,
            max_iter=args.lbfgs_iter, 
            ic_weight=args.ic_weight,
            bc_weight=args.bc_weight,
            pde_weight=args.pde_weight,
            pde_phi_w=args.pde_phi_w,
            pde_c_w=args.pde_c_w,
            pde_mu_w=args.pde_mu_w
        )
    print(f"Tempo di esecuzione: {time.time() - start_time:.2f}s")

    #save the model weights
    save_model(model, file_name = args.checkpoint_name)

    #plot train loss vs epoch
    x_epochs = np.arange(1, n_epochs + 1)

    out_dir = Path("artifacts/baseline/lossplots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.semilogy(x_epochs, train_losses["pde_phi"], label="PDE phi")
    plt.semilogy(x_epochs, train_losses["pde_c"], label="PDE c")
    plt.semilogy(x_epochs, train_losses["pde_mu"], label="PDE mu")
    plt.semilogy(x_epochs, train_losses["bc"], label="BC")
    plt.semilogy(x_epochs, train_losses["ic"], label="IC")

    plt.xlabel("Epochs")
    plt.ylabel("Train loss")
    plt.title("Coupled AC-CH 1D training losses")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.savefig(out_dir / args.lossplot_name, dpi=200, bbox_inches="tight")
    plt.close()

    #plot pre-train loss vs pre-train epoch
    if pretrain_history is not None:
        out_dir = Path("artifacts/baseline/lossplots/pretrain")
        out_dir.mkdir(parents=True, exist_ok=True)

        x_pre = np.arange(1, len(pretrain_history["total"]) + 1)

        plt.figure(figsize=(8, 5))
        plt.semilogy(x_pre, pretrain_history["ic_phi"], label="IC phi")
        plt.semilogy(x_pre, pretrain_history["ic_c"], label="IC c")
        plt.semilogy(x_pre, pretrain_history["ic_mu"], label="IC mu")
        plt.semilogy(x_pre, pretrain_history["total"], label="IC total", linestyle="--")

        plt.xlabel("Pretraining epochs")
        plt.ylabel("IC loss")
        plt.title("Coupled AC-CH initial condition pretraining")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.savefig(out_dir / args.lossplot_name, dpi=200, bbox_inches="tight")
        plt.close()