from pathlib import Path
import torch
import time

from losses import pde_loss, bc_loss, ic_loss
from utils import generate_coll_points_and_ic

#function to train the model for just one iteration 
def train_one_epoch(
        model, 
        collocation, 
        c_ic_true, 
        mu_ic_true, 
        optimizer, 
        M, 
        epsilon, 
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
    loss_pde, loss_pde_c, loss_pde_mu = pde_loss(model, x_pde, t_pde, M, epsilon)
    loss_bc = bc_loss(model, x_bc, t_bc)
    loss_ic_c, loss_ic_mu = ic_loss(model, x_ic, t_ic, c_ic_true, mu_ic_true)

    loss = pde_weight * loss_pde + bc_weight * loss_bc + ic_weight * (loss_ic_c + loss_ic_mu)
    loss.backward()

    optimizer.step() #updating the gradients

    #printing results 
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:05d} | Loss PDE (c): {loss_pde_c.item():.4e} | Loss PDE (mu): {loss_pde_mu.item():.4e} | Loss BC: {loss_bc.item():.4e} | Loss IC: {(loss_ic_c + loss_ic_mu).item():.4e}")

    return loss, loss_pde, loss_pde_c, loss_pde_mu, loss_bc, loss_ic_c + loss_ic_mu

# #function to pre-train the model (to enforce the initial condition learning, setting to 0 the pde loss term)
# def pretrain_model(
#         model, 
#         optimizer, 
#         M, 
#         epsilon, 
#         pretrain_epochs
# ):
#     pretrain_losses = [] #initiating history of total losses in pre-train phase

#     coll_pre, c_ic_true_pre, m_ic_true_pre = generate_coll_points_and_ic(
#         N_pde = 10, #just to avoid any numerical error 
#         N_bc = 2, #in this case borders are just two points: (0,0), (L,0)
#         N_ic = N_ic, 
#         L = L, 
#         T_max = 0.0,
#         epsilon = epsilon
#     )
#     for epoch in range(1, pretrain_epochs + 1):
#         l_total, _, l_bc, l_ic = train_one_epoch(
#             model, 
#             coll_pre, c_ic_true_pre, 
#             optimizer, 
#             M, epsilon, 
#             epoch, 
#             ic_weight = 1.0,
#             pde_weight = 0.0 #!
#         )

#         pretrain_losses.append(l_total.item())

#     return pretrain_losses



#function to train the model for a fixed n. of iterations (and return the losses history)
def train_model(
    model, 
    optimizer, 
    M, 
    epsilon,
    collocation, #collocation points dictionary
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
        "pde_c": [],
        "pde_mu": [],
        "bc": [],
        "ic": []
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

        l_total, l_pde, l_pde_c, l_pde_mu, l_bc, l_ic = train_one_epoch(
            model, 
            collocation, 
            c_ic_true,
            mu_ic_true,  
            optimizer, 
            M, 
            epsilon, 
            epoch,
            ic_weight = ic_weight, 
            bc_weight = bc_weight,
            pde_weight = pde_weight
        ) #train one epoch and calculate losses terms

        #update losses history with current values
        train_losses["total"].append(l_total.item())
        train_losses["pde"].append(l_pde.item())
        train_losses["pde_c"].append(l_pde_c.item())
        train_losses["pde_mu"].append(l_pde_mu.item())
        train_losses["bc"].append(l_bc.item())
        train_losses["ic"].append(l_ic.item())

    # return pretrain_losses, train_losses
    return train_losses

#function to perform the training refinement with l-bfgs algorithm (post-Adam)
def train_lbfgs(
    model,
    collocation,
    c_ic_true,
    mu_ic_true,
    M,
    epsilon,
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

        loss_pde, loss_pde_c, loss_pde_mu = pde_loss(model, x_pde, t_pde, M, epsilon)
        loss_bc = bc_loss(model, x_bc, t_bc)
        loss_ic_c, loss_ic_mu = ic_loss(
            model, x_ic, t_ic, c_ic_true, mu_ic_true
        )

        loss_ic = loss_ic_c + loss_ic_mu
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
    checkpoint_dir = Path(__file__).resolve().parents[0] / "artifacts" 
    checkpoint_dir.mkdir(parents=True, exist_ok = True) #check if folder exists, if not create it
    checkpoint_path = checkpoint_dir / file_name

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModello salvato con successo in {checkpoint_path}")

if __name__ == "__main__":
    from torch.optim import Adam
    import matplotlib.pyplot as plt
    import numpy as np
    import argparse

    from models import CahnHilliardPINN 

    #adding CLI parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--epsilon", type=float, default=0.05, help = "Interface penalty term epsilon")
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
    M = 0.1 #mobility
    epsilon = args.epsilon #interface penalty term

    #number of collocation points 
    N_pde = args.n_pde #inside the domain 
    N_bc = args.n_bc #on the boundaries 
    N_ic = args.n_ic #for initial time

    #create model and set Adam optimizer
    model = CahnHilliardPINN(hidden_layers = 4, hidden_dim = 128).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)

    n_epochs = args.epochs
    collocation, c_ic_true, mu_ic_true = generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max, epsilon, device) #generate training collocation pts

    start_time = time.time()
    #training the model for n_epochs
    train_losses = train_model(
        model, 
        optimizer, 
        M, 
        epsilon, 
        collocation, 
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
        c_ic_true=c_ic_true,
        mu_ic_true=mu_ic_true,
        M=M,
        epsilon=epsilon,
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


