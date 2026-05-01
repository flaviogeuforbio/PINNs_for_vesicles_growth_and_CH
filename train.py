from pathlib import Path
import torch

from losses import pde_loss, bc_loss, ic_loss
from utils import generate_coll_points_and_ic, get_gradient_norm, add_confined_adaptive_points

#function to train the model for just one iteration 
def train_one_epoch(
        model, 
        collocation, 
        c_ic_true, 
        optimizer, 
        M, 
        epsilon, 
        epoch,
        ic_weight: float = 1.0, #ic loss term weight
        pde_weight: float = 1.0 #pde loss term weight (set to 0 in pre-training phase)
):
    model.train()

    #unpacking collocation dictionary to extract all collocation points
    x_pde, t_pde = collocation["x_pde"], collocation["t_pde"]
    x_bc, t_bc = collocation["x_bc"], collocation["t_bc"]
    x_ic, t_ic = collocation["x_ic"], collocation["t_ic"]

    optimizer.zero_grad() #zero-ing the gradients

    #calculating the total loss and backpropagating
    loss_pde = pde_loss(model, x_pde, t_pde, M, epsilon)
    loss_bc = bc_loss(model, x_bc, t_bc)
    loss_ic = ic_loss(model, x_ic, t_ic, c_ic_true)

    loss = pde_weight * loss_pde + loss_bc + ic_weight * loss_ic
    loss.backward()

    optimizer.step() #updating the gradients

    #printing results 
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:05d} | Loss PDE: {loss_pde.item():.4e} | Loss BC: {loss_bc.item():.4e} | Loss IC: {loss_ic.item():.4e}")

    return loss, loss_pde, loss_bc, loss_ic

#function to pre-train the model (to enforce the initial condition learning, setting to 0 the pde loss term)
def pretrain_model(
        model, 
        optimizer, 
        M, 
        epsilon, 
        pretrain_epochs
):
    pretrain_losses = [] #initiating history of total losses in pre-train phase

    coll_pre, c_ic_true_pre = generate_coll_points_and_ic(
        N_pde = 10, #just to avoid any numerical error 
        N_bc = 2, #in this case borders are just two points: (0,0), (L,0)
        N_ic = N_ic, 
        L = L, 
        T_max = 0.0
    )
    for epoch in range(1, pretrain_epochs + 1):
        l_total, _, l_bc, l_ic = train_one_epoch(
            model, 
            coll_pre, c_ic_true_pre, 
            optimizer, 
            M, epsilon, 
            epoch, 
            ic_weight = 1.0,
            pde_weight = 0.0 #!
        )

        pretrain_losses.append(l_total.item())

    return pretrain_losses



#function to train the model for a fixed n. of iterations (and return the losses history)
def train_model(
    model, 
    optimizer, 
    M, 
    epsilon, 
    n_epochs: int = 10000,
    pretrain_epochs: int = 1000,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0 
):
    #initiate losses history
    train_losses = {
        "total": [],
        "pde": [],
        "bc": [],
        "ic": []
    }

    if pretrain_epochs > 0:
        #PRE-TRAINING PHASE (just IC training, setting pde loss term weight to 0)
        #-------------------------------------
        print("\n" + "="*40)
        print("FASE 1: PRE-TRAINING CONDIZIONE INIZIALE")
        print("="*40)

        #pre-train the model and save losses history on initial conditions 
        pretrain_losses = pretrain_model(
            model, 
            optimizer, 
            M, 
            epsilon, 
            pretrain_epochs
        )
    else:
        pretrain_losses = None

    #TRAINING PHASE (w/ resampling)
    #-------------------------------------
    print("\n" + "="*40)
    print("FASE 2: TIME-MARCHING E SOLUZIONE PDE")
    print("="*40)

    curr_T_max = 0.1 * T_max
    checkpoint = n_epochs // 10

    for epoch in range(1, n_epochs + 1):
        #calculate n. of collocation points to generate in the current epoch to keep sampling density fixed in the space-time domain 
        curr_N_pde = int(N_pde * (curr_T_max / T_max))
        curr_N_bc = int(N_bc * (curr_T_max / T_max))

        collocation, c_ic_true = generate_coll_points_and_ic(curr_N_pde, curr_N_bc, N_ic, L, curr_T_max) #generate collocation pts in current time interval

        l_total, l_pde, l_bc, l_ic = train_one_epoch(
            model, 
            collocation, 
            c_ic_true, 
            optimizer, 
            M, 
            epsilon, 
            epoch,
            ic_weight, 
            pde_weight
        ) #train one epoch and calculate losses terms

        #update losses history with current values
        train_losses["total"].append(l_total.item())
        train_losses["pde"].append(l_pde.item())
        train_losses["bc"].append(l_bc.item())
        train_losses["ic"].append(l_ic.item())

        if epoch % checkpoint == 0 and curr_T_max < T_max:
            curr_T_max += 0.1 * T_max
            curr_T_max = round(curr_T_max, 1) #to avoid float precision errors
            print(f"\n---> TIME-MARCHING: Finestra espansa a T = {curr_T_max} <---")

    return pretrain_losses, train_losses



#------- NTK METHOD ---------------------------------------------------------------------------------------------------------------------

#function to train the model for one epoch using the dynamic weight adjustment given by NTK method
def train_one_epoch_ntk(
        model, 
        collocation, 
        c_ic_true, 
        optimizer, 
        M, 
        epsilon, 
        epoch, 
        w_bc_old: float, #BC loss dynamical weight of the previous epoch 
        w_ic_old: float, #IC loss """"
        alpha: float #parameter that regulates the dynamical weights update (e.g. 0.1 = 10% new, 90% old)
):
    
    model.train()

    #unpacking collocation dictionary to extract all collocation points
    x_pde, t_pde = collocation["x_pde"], collocation["t_pde"]
    x_bc, t_bc = collocation["x_bc"], collocation["t_bc"]
    x_ic, t_ic = collocation["x_ic"], collocation["t_ic"]

    optimizer.zero_grad() #zero-ing the gradients

    #calculating the total loss and backpropagating
    loss_pde = pde_loss(model, x_pde, t_pde, M, epsilon)
    loss_bc = bc_loss(model, x_bc, t_bc)
    loss_ic = ic_loss(model, x_ic, t_ic, c_ic_true)

    #extract the NTK traces estimation
    tr_pde = get_gradient_norm(loss_pde, model)
    tr_bc = get_gradient_norm(loss_bc, model)
    tr_ic = get_gradient_norm(loss_ic, model)

    #compute target dynamical weights (the idea is to fix w_pde = 1 and adjust w_bc, w_ic)
    w_bc_target = tr_pde / (tr_bc + 1e-8)
    w_ic_target = tr_pde / (tr_ic + 1e-8)

    #clamping to avoid explosion or vanishing 
    w_bc_target = torch.clamp(torch.tensor(w_bc_target), min = 1.0, max = 100.0).item()
    w_ic_target = torch.clamp(torch.tensor(w_ic_target), min = 1.0, max = 100.0).item()

    #compute new dynamical weights 'with cache'
    w_bc_new = (1 - alpha) * w_bc_old + alpha * w_bc_target
    w_ic_new = (1 - alpha) * w_ic_old + alpha * w_ic_target

    #calculate total loss and backpropagate
    loss = loss_pde + w_bc_new * loss_bc + w_ic_new * loss_ic
    loss.backward()

    optimizer.step() #updating the gradients

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:05d} | "
            f"L_PDE: {loss_pde.item():.2e} | L_BC: {loss_bc.item():.2e} (w={w_bc_new:.1f}) | "
            f"L_IC: {loss_ic.item():.2e} (w={w_ic_new:.1f})")

    return loss, loss_pde, loss_bc, loss_ic, w_bc_new, w_ic_new

#wrapper to train the model using NTK method for adaptive loss weights and collocation points resampling to address the interfaces
def train_model_ntk_adaptive(
    model, 
    optimizer, 
    M, 
    epsilon, 
    n_epochs: int = 10000,
    pretrain_epochs: int = 1000,
    refine_every: int = 500, #once every refine_every epochs new confined adaptive coll. points are generated and added to the training set
    alpha: float = 0.1 #parameter that regulates the NTK dynamical weights update (e.g. 0.1 = 10% new, 90% old) 
):
    #initiate losses history
    train_losses = {
        "total": [],
        "pde": [],
        "bc": [],
        "ic": [],
        "w_bc": [],
        "w_ic": []
    }

    if pretrain_epochs > 0:
        #PRE-TRAINING PHASE (just IC training, setting pde loss term weight to 0)
        #-------------------------------------
        print("\n" + "="*40)
        print("FASE 1: PRE-TRAINING CONDIZIONE INIZIALE")
        print("="*40)

        #pre-train the model and save losses history on initial conditions 
        pretrain_losses = pretrain_model(
            model, 
            optimizer, 
            M, 
            epsilon, 
            pretrain_epochs
        )
    else:
        pretrain_losses = None

    #TRAINING PHASE (w/ resampling)
    #-------------------------------------
    print("\n" + "="*40)
    print("FASE 2: TIME-MARCHING E SOLUZIONE PDE")
    print("="*40)

    curr_T_max = 0.1 * T_max
    checkpoint = n_epochs // 10

    #set reasonable initial values for adaptive weights 
    curr_w_bc = 1.0 
    curr_w_ic = 10.0 #just for test (NTK off for now calling the function with alpha = 0) !

    #creating the adaptive buffer for resampled points 
    adaptive_x_buffer = torch.empty((0, 1))
    adaptive_t_buffer = torch.empty((0, 1))

    for epoch in range(1, n_epochs + 1):
        #calculate n. of collocation points to generate in the current epoch to keep sampling density fixed in the space-time domain 
        curr_N_pde = int(N_pde * (curr_T_max / T_max))
        curr_N_bc = int(N_bc * (curr_T_max / T_max))

        collocation, c_ic_true = generate_coll_points_and_ic(curr_N_pde, curr_N_bc, N_ic, L, curr_T_max) #generate collocation pts in current time interval

        #concatenate collocation points with resampled points (if any)
        if adaptive_x_buffer.shape[0] > 0:
            x_pde_full = torch.cat([collocation["x_pde"], adaptive_x_buffer], dim = 0).detach().requires_grad_(True)
            t_pde_full = torch.cat([collocation["t_pde"], adaptive_t_buffer], dim = 0).detach().requires_grad_(True)

            collocation["x_pde"] = x_pde_full
            collocation["t_pde"] = t_pde_full

        l_total, l_pde, l_bc, l_ic, curr_w_bc, curr_w_ic = train_one_epoch_ntk(
            model, 
            collocation, 
            c_ic_true, 
            optimizer, 
            M, 
            epsilon, 
            epoch,
            curr_w_bc,
            curr_w_ic,
            alpha
        ) #train one epoch and calculate losses terms

        #update losses history with current values
        train_losses["total"].append(l_total.item())
        train_losses["pde"].append(l_pde.item())
        train_losses["bc"].append(l_bc.item())
        train_losses["ic"].append(l_ic.item())
        train_losses["w_bc"].append(curr_w_bc)
        train_losses["w_ic"].append(curr_w_ic)

        if epoch % refine_every == 0 and epoch > 100: #we turn on the resampling after the initial transient of the net
            new_x, new_t = add_confined_adaptive_points(
                model, 
                curr_T_max, 
                L, 
                M, 
                epsilon, 
                n_new_points = 1000, 
                n_candidates = 10000, 
                spread = 0.01
            )

            #updating the adaptive buffer with new resampled points 
            adaptive_x_buffer = torch.cat([adaptive_x_buffer, new_x])
            adaptive_t_buffer = torch.cat([adaptive_t_buffer, new_t])
            print(f"Nuova dimensione buffer adattivo: {adaptive_x_buffer.shape[0]} punti.")

        if epoch % checkpoint == 0 and curr_T_max < T_max:
            curr_T_max += 0.1 * T_max
            curr_T_max = round(curr_T_max, 1) #to avoid float precision errors
            print(f"\n---> TIME-MARCHING: Finestra espansa a T = {curr_T_max} <---")

    return pretrain_losses, train_losses


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

    from models import CahnHilliardPINN    

    #physical parameters
    L = 1.0 #box lenght (1D)
    T_max = 4.0 #simulation end time
    M = 0.1 #mobility
    epsilon = 0.05 #interface penalty term

    #number of collocation points 
    N_pde = 10000 #inside the domain 
    N_bc = 2000 #on the boundaries 
    N_ic = 2000 #for initial time

    #create model and set Adam optimizer
    model = CahnHilliardPINN(hidden_layers = 5, hidden_dim = 128)
    optimizer = Adam(model.parameters(), lr = 1e-3)

    n_epochs = 5000

    #training the model for n_epochs
    pretrain_losses, train_losses = train_model_ntk_adaptive(
        model, 
        optimizer, 
        M, 
        epsilon, 
        n_epochs,
        alpha = 0.0 #just for test (turn off NTK method) !
    )

    #save the model weights
    save_model(model, file_name = "ch_resamp_nontk_tmax4_pinn_005eps.pt")

    #plot train loss vs epoch
    x_epochs = np.arange(1, n_epochs + 1)

    plt.plot(x_epochs, train_losses["pde"], label = "pde")
    plt.plot(x_epochs, train_losses["bc"], label = "bc")
    plt.plot(x_epochs, train_losses["ic"], label = "ic")

    plt.xlabel("Epochs")
    plt.ylabel("Train loss")
    
    plt.legend()
    plt.show()



