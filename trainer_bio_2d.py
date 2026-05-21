import time
import torch 
from torch.optim import Adam
from pathlib import Path

from models import BioACCHPINN2d
from losses_bio_2d import pde_loss, bc_loss, ic_loss
from utils_bio_2d import generate_coll_points_and_ic

#function to train the model for just one iteration 
def train_one_epoch(
        model, 
        collocation,
        phi_ic_true,  
        mu_ic_true, 
        psi_ic_true, 
        nu_ic_true,
        optimizer, 
        epoch,
        args
):
    model.train()

    #unpacking collocation dictionary to extract all collocation points
    x_pde, y_pde, t_pde = collocation["x_pde"], collocation["y_pde"], collocation["t_pde"]
    x_bc, y_bc, t_bc, normal = collocation["x_bc"], collocation["y_bc"], collocation["t_bc"], collocation["normal_bc"]
    x_ic, y_ic, t_ic = collocation["x_ic"], collocation["y_ic"], collocation["t_ic"]

    optimizer.zero_grad() #zero-ing the gradients

    #calculating the total loss and backpropagating
    l_pde, l_pde_phi, l_pde_mu, l_pde_psi, l_pde_nu = pde_loss(x_pde, y_pde, t_pde, model, args)
    l_bc, l_bc_phi, l_bc_mu, l_bc_psi, l_bc_nu = bc_loss(model, x_bc, y_bc, t_bc, normal)
    l_ic, l_ic_phi, l_ic_mu, l_ic_psi, l_ic_nu = ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true)

    #loss terms weight unpacking 
    pde_weight, bc_weight, ic_weight = args.pde_weight, args.bc_weight, args.ic_weight
    pde_phi_w, pde_mu_w, pde_psi_w, pde_nu_w = args.pde_phi_w, args.pde_mu_w, args.pde_psi_w, args.pde_nu_w

    loss = (
        pde_weight * (pde_phi_w * l_pde_phi + pde_mu_w * l_pde_mu + pde_psi_w * l_pde_psi + pde_nu_w * l_pde_nu) 
        + bc_weight * l_bc
        + ic_weight * l_ic
    )
    loss.backward()

    optimizer.step() #updating the gradients

    #printing results 
    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:05d} | "
            f"Loss PDE (phi): {l_pde_phi.item():.4e} | "
            f"Loss PDE (mu): {l_pde_mu.item():.4e} | "
            f"Loss PDE (psi): {l_pde_psi.item():.4e} | "
            f"Loss PDE (nu): {l_pde_nu.item():.4e} | "
            f"Loss BC: {l_bc.item():.4e} | "
            f"Loss IC: {l_ic.item():.4e}"
        )

    return {
        "total": loss.item(),
        "pde": l_pde.item(),
        "pde_phi": l_pde_phi.item(),
        "pde_mu": l_pde_mu.item(),
        "pde_psi": l_pde_psi.item(), 
        "pde_nu": l_pde_nu.item(),
        "bc": l_bc.item(),
        "bc_phi": l_bc_phi.item(),
        "bc_mu": l_bc_mu.item(),
        "bc_psi": l_bc_psi.item(), 
        "bc_nu": l_bc_nu.item(),
        "ic": l_ic.item(),
        "ic_phi": l_ic_phi.item(),
        "ic_mu": l_ic_mu.item(),
        "ic_psi": l_ic_psi.item(), 
        "ic_nu": l_ic_nu.item()
    }


#function to pre-train the model on IC only 
def pretrain_initial_condition(
        model, 
        collocation, 
        phi_ic_true, 
        mu_ic_true, 
        psi_ic_true, 
        nu_ic_true, 
        optimizer, 
        pretrain_epochs: int
):
    model.train()

    x_ic, y_ic, t_ic = collocation["x_ic"], collocation["y_ic"], collocation["t_ic"]

    #initiating the pre-train loss history
    history = {
        "total": [],
        "ic_phi": [],
        "ic_mu": [],
        "ic_psi": [], 
        "ic_nu": []
    }

    for epoch in range(1, pretrain_epochs + 1):
        optimizer.zero_grad()

        #calculating the loss, backprop & updating gradients
        l_ic, l_ic_phi, l_ic_mu, l_ic_psi, l_ic_nu  = ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true)

        l_ic.backward()
        optimizer.step()

        #updating history
        history["total"].append(l_ic.item())
        history["ic_phi"].append(l_ic_phi.item())
        history["ic_mu"].append(l_ic_mu.item())
        history["ic_psi"].append(l_ic_psi.item())
        history["ic_nu"].append(l_ic_nu.item())

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Pretrain epoch {epoch:05d} | "
                f"IC phi: {l_ic_phi.item():.4e} | "
                f"IC mu: {l_ic_mu.item():.4e} | "
                f"IC psi: {l_ic_psi.item():.4e} | "
                f"IC nu: {l_ic_nu.item():.4e} | "
                f"IC total: {l_ic.item():.4e}"
            )

    return history
    

#function to train the model for a fixed n. of iterations (and return the losses history)
def train_model(
    model, 
    optimizer, 
    collocation, #collocation points dictionary
    phi_ic_true, #true initial non-cons. field profile (phi(x, y, 0))
    mu_ic_true, #true initial potential profile (mu(x, y, 0))
    psi_ic_true, #true initial cons. field profile (psi(x, y, 0))
    nu_ic_true, #true initial potential profile (nu(x, y, 0))
    n_epochs: int,
    pretrain_epochs: int,
    args
):
    #initiate losses history
    train_losses = {
        "total": [],
        "pde": [],
        "pde_phi": [],
        "pde_mu": [],
        "pde_psi": [], 
        "pde_nu": [],
        "bc": [],
        "bc_phi": [],
        "bc_mu": [],
        "bc_psi": [], 
        "bc_nu": [],
        "ic": [],
        "ic_phi": [],
        "ic_mu": [],
        "ic_psi": [], 
        "ic_nu": []
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
            phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true,  
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
            mu_ic_true, 
            psi_ic_true, 
            nu_ic_true,  
            optimizer, 
            epoch,
            args
        ) #train one epoch and calculate losses terms

        #update losses history with current values
        train_losses["total"].append(epoch_losses["total"])
        train_losses["pde"].append(epoch_losses["pde"])
        train_losses["pde_phi"].append(epoch_losses["pde_phi"])
        train_losses["pde_mu"].append(epoch_losses["pde_mu"])
        train_losses["pde_psi"].append(epoch_losses["pde_psi"])
        train_losses["pde_nu"].append(epoch_losses["pde_nu"])
        train_losses["bc"].append(epoch_losses["bc"])
        train_losses["bc_phi"].append(epoch_losses["bc_phi"])
        train_losses["bc_mu"].append(epoch_losses["bc_mu"])
        train_losses["bc_psi"].append(epoch_losses["bc_psi"])
        train_losses["bc_nu"].append(epoch_losses["bc_nu"])
        train_losses["ic"].append(epoch_losses["ic"])
        train_losses["ic_phi"].append(epoch_losses["ic_phi"]) 
        train_losses["ic_mu"].append(epoch_losses["ic_mu"])  
        train_losses["ic_psi"].append(epoch_losses["ic_psi"])
        train_losses["ic_nu"].append(epoch_losses["ic_nu"])

    # return pretrain_losses, train_losses
    return train_losses, pretrain_losses


#function to perform the training refinement with l-bfgs algorithm (post-Adam)
def train_lbfgs(
    model,
    collocation,
    phi_ic_true,
    mu_ic_true,
    psi_ic_true, 
    nu_ic_true,
    args
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
        max_iter=args.lbfgs_iter,
        max_eval=args.lbfgs_iter * 2,
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

        l_pde, l_pde_phi, l_pde_mu, l_pde_psi, l_pde_nu = pde_loss(x_pde, y_pde, t_pde, model, args)
        l_bc, *_ = bc_loss(model, x_bc, y_bc, t_bc, normal)
        l_ic, *_ = ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true)

        #loss terms weight unpacking 
        pde_weight, bc_weight, ic_weight = args.pde_weight, args.bc_weight, args.ic_weight
        pde_phi_w, pde_mu_w, pde_psi_w, pde_nu_w = args.pde_phi_w, args.pde_mu_w, args.pde_psi_w, args.pde_nu_w

        loss = (
            pde_weight * (pde_phi_w * l_pde_phi + pde_mu_w * l_pde_mu + pde_psi_w * l_pde_psi + pde_nu_w * l_pde_nu) 
            + bc_weight * l_bc
            + ic_weight * l_ic
        )
        loss.backward()

        history["total"].append(loss.item())
        history["pde"].append(l_pde.item())
        history["bc"].append(l_bc.item())
        history["ic"].append(l_ic.item())

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


#function to train one model just on a specific time interval (time-windowing method with ensemble of networks)
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
    model = BioACCHPINN2d(
        hidden_layers = args.hidden_layers, 
        hidden_dim = args.hidden_dim
    ).to(device)

    optimizer = Adam(model.parameters(), lr = args.lr) 

    #generate collocation points + ic (true or from previous model)
    collocation, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = generate_coll_points_and_ic(
        args = args,
        device = device, 
        ic_fn = ic_fn
    )

    #start the training phase (main loop with Adam + refinement with L-BFGS algorithm)
    start_time = time.time()
    train_losses, pretrain_losses = train_model(
        model, 
        optimizer, 
        collocation,
        phi_ic_true, 
        mu_ic_true,
        psi_ic_true, 
        nu_ic_true, 
        n_epochs = args.epochs,
        pretrain_epochs = args.pretrain_epochs,
        args = args
    )

    #L-BFGS
    if args.lbfgs_iter > 0:
        lbfgs_losses = train_lbfgs(
            model=model,
            collocation=collocation,
            phi_ic_true=phi_ic_true,
            mu_ic_true=mu_ic_true,
            psi_ic_true=psi_ic_true, 
            nu_ic_true=nu_ic_true, 
            args=args
        )
    print(f"Execution time: {time.time() - start_time:.2f}s")

    return model, train_losses, pretrain_losses, lbfgs_losses