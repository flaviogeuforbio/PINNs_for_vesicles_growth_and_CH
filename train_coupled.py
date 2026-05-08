from pathlib import Path
import torch
import time

from losses_coupled import pde_loss, bc_loss, ic_loss
from utils_coupled import generate_coll_points_and_ic

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
        ic_weight: float, #ic loss term weight
        bc_weight: float, #bc loss term weight 
        pde_weight: float #pde loss term weight (set to 0 in pre-training phase)
):
    model.train()

    #unpacking collocation dictionary to extract all collocation points
    x_pde, t_pde = collocation["x_pde"], collocation["t_pde"]
    x_bc, t_bc = collocation["x_bc"], collocation["t_bc"]
    x_ic, t_ic = collocation["x_ic"], collocation["t_ic"]

    optimizer.zero_grad() #zero-ing the gradients

    #calculating the total loss and backpropagating
    loss_pde, loss_pde_phi, loss_pde_c, loss_pde_mu = pde_loss(x_pde, t_pde, model, M_phi, M_c, eps_phi, eps_c, gamma)
    loss_bc, loss_bc_phi, loss_bc_c, loss_bc_mu = bc_loss(model, x_bc, t_bc)
    loss_ic, loss_ic_phi, loss_ic_c, loss_ic_mu = ic_loss(model, x_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true)

    loss = pde_weight * loss_pde + bc_weight * loss_bc + ic_weight * loss_ic
    loss.backward()

    optimizer.step() #updating the gradients

    #printing results 
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:05d} | Loss PDE (c): {loss_pde_c.item():.4e} | Loss PDE (mu): {loss_pde_mu.item():.4e} | Loss BC: {loss_bc.item():.4e} | Loss IC: {(loss_ic_c + loss_ic_mu).item():.4e}")

    return {
        "total": loss.item(),
        "pde": loss_pde.item(),
        "pde_phi": loss_pde_phi.item(),
        "pde_c": loss_pde_c.item(),
        "bc": loss_bc.item(),
        "bc_phi": loss_bc_phi.item(),
        "bc_c": loss_bc_c.item(),
        "bc_mu": loss_bc_mu.item(),
        "ic": loss_ic.item(),
        "ic_phi": loss_ic_phi.item(),
        "ic_c": loss_ic_c.item(),
        "ic_mu": loss_ic_mu.item()
    }



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
    n_epochs: int = 10000,
    pretrain_epochs: int = 1000,
    ic_weight: float = 100.0,
    bc_weight: float = 10.0,
    pde_weight: float = 1.0 
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

    # if pretrain_epochs > 0:
    #     #PRE-TRAINING PHASE (just IC training, setting pde loss term weight to 0)
    #     #-------------------------------------
    #     print("\n" + "="*40)
    #     print("FASE 1: PRE-TRAINING CONDIZIONE INIZIALE")
    #     print("="*40)

    #     #pre-train the model and save losses history on initial conditions 
    #     pretrain_losses = pretrain_model(
    #         model, 
    #         optimizer, 
    #         M, 
    #         epsilon, 
    #         pretrain_epochs
    #     )
    # else:
    #     pretrain_losses = None

    #TRAINING PHASE (w/ resampling)
    #-------------------------------------
    print("\n" + "="*40)
    print("FASE 2: TIME-MARCHING E SOLUZIONE PDE")
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
            pde_weight = pde_weight
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
    return train_losses


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
    max_iter=200,
    ic_weight=100.0,
    bc_weight=10.0,
    pde_weight=1.0,
):
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

    x_pde, t_pde = collocation["x_pde"], collocation["t_pde"]
    x_bc, t_bc = collocation["x_bc"], collocation["t_bc"]
    x_ic, t_ic = collocation["x_ic"], collocation["t_ic"]

    history = {
        "total": [],
        "pde": [],
        "bc": [],
        "ic": [],
    }

    def closure():
        optimizer.zero_grad()

        loss_pde, *_ = pde_loss(x_pde, t_pde, model, M_phi, M_c, eps_phi, eps_c, gamma)
        loss_bc, *_ = bc_loss(model, x_bc, t_bc)
        loss_ic, *_ = ic_loss(
            model, x_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true
        )

        loss = (
            pde_weight * loss_pde
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
    checkpoint_dir = Path(__file__).resolve().parents[0] / "artifacts" / "coupled"
    checkpoint_dir.mkdir(parents=True, exist_ok = True) #check if folder exists, if not create it
    checkpoint_path = checkpoint_dir / file_name

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModello salvato con successo in {checkpoint_path}")

if __name__ == "__main__":
    from torch.optim import Adam
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse

    from models import CoupledACCHPINN 

    #adding CLI parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--eps_phi", type=float, default=0.05, help = "Interface penalty term epsilon for phi")
    parser.add_argument("--eps_c", type=float, default=0.05, help = "Interface penalty term epsilon for c")
    parser.add_argument("--m_phi", type=float, default=1.0, help = "Mobility parameter for phi")
    parser.add_argument("--m_c", type=float, default=0.1, help = "Mobility parameter for c")
    parser.add_argument("--gamma", type=float, default=0.05, help = "Coupling parameter (int = gamma * c * phi)")
    parser.add_argument("--epochs", type=int, default=3000, help = "N. of training epochs")
    parser.add_argument("--lbfgs_iter", type=int, default=50, help = "N. of L-BFGS iterations")
    parser.add_argument("--n_pde", type=int, default=10000, help = "N. of PDE collocation points")
    parser.add_argument("--n_bc", type=int, default=2000, help = "N. of BC collocation points")
    parser.add_argument("--n_ic", type=int, default=2000, help = "N. of IC collocation points")
    parser.add_argument("--outpath", type=str, required=True, help = "model checkpoint name")
    args = parser.parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)   

    #physical parameters
    L = 1.0 #box lenght (1D)
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
    model = CoupledACCHPINN(hidden_layers = 4, hidden_dim = 128).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)

    n_epochs = args.epochs
    collocation, phi_ic_true, c_ic_true, mu_ic_true = generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max, eps_c, gamma, device) #generate training collocation pts

    start_time = time.time()
    #training the model for n_epochs
    train_losses = train_model(
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
        pretrain_epochs = 0,
        ic_weight = 100.0, 
        bc_weight = 10.0, 
        pde_weight = 1.0
    )

    #refining the training with lbfgs
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
        ic_weight=100.0,
        bc_weight=10.0,
        pde_weight=1.0,
    )
    print(f"Tempo di esecuzione: {time.time() - start_time:.2f}s")

    #save the model weights
    save_model(model, file_name = args.outpath)

    #plot train loss vs epoch
    x_epochs = np.arange(1, n_epochs + 1)

    plt.plot(x_epochs, train_losses["pde_c"], label = "pde (c)")
    plt.plot(x_epochs, train_losses["pde_mu"], label = "pde (mu)")
    plt.plot(x_epochs, train_losses["bc"], label = "bc")
    plt.plot(x_epochs, train_losses["ic"], label = "ic")

    plt.xlabel("Epochs")
    plt.ylabel("Train loss")
    
    plt.legend()
    plt.show()