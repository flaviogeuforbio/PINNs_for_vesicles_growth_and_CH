from pathlib import Path
import torch

from src.losses import pde_loss, bc_loss, ic_loss

#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max):
    #pde collocation points
    x_pde = torch.rand(size = (N_pde, 1)) * L
    t_pde = torch.rand(size = (N_pde, 1)) * T_max

    #bc collocation points 
    x_bc_right = torch.ones(size = (N_bc // 2, 1)) * L
    x_bc_left = torch.zeros(size = (N_bc // 2, 1))
    x_bc = torch.cat([x_bc_right, x_bc_left], dim = 0)
    t_bc = torch.rand(size = (N_bc, 1)) * T_max

    #ic collocation points
    x_ic = torch.rand(size = (N_ic, 1)) * L
    t_ic = torch.zeros(size = (N_ic, 1))

    #creating dictionary with all collocation points (pde, ic, bc)
    collocation = {
        "x_pde": x_pde,
        "t_pde": t_pde,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "x_ic": x_ic, 
        "t_ic": t_ic
    }

    #setting the true initial condition
    # noise = torch.randn_like(x_ic) * 0.1 #create gaussian noise with variance sigma = 0.1
    # c_ic_true = torch.zeros_like(x_ic) + noise #this way we are creating thermodynamically unstable domains (a thermal fluctuation)

    c_ic_true = 0.3 * torch.cos(torch.pi * x_ic)
    #this function satisfies the Neumann conditions 

    return collocation, c_ic_true

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
        N_pde = 10, #just to avoid any  
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
        "ic": [],
        "pretrain_tot": []
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

    from src.models import CahnHilliardPINN
    

    #physical parameters
    L = 1.0 #box lenght (1D)
    T_max = 10.0 #simulation end time
    M = 0.1 #mobility
    epsilon = 0.05 #interface penalty term

    #number of collocation points 
    N_pde = 10000 #inside the domain 
    N_bc = 2000 #on the boundaries 
    N_ic = 2000 #for initial time

    # collocation, c_ic_true = generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max) #generate collocation points and true I.C. for concentration

    #create model and set Adam optimizer
    model = CahnHilliardPINN(hidden_layers = 3, hidden_dim = 64)
    optimizer = Adam(model.parameters(), lr = 1e-3)

    n_epochs = 5000

    #training the model for n_epochs
    pretrain_losses, train_losses = train_model(
        model, 
        optimizer, 
        M, 
        epsilon, 
        n_epochs,
        ic_weight = 10.0, 
        pde_weight = 1.0 
    )

    #save the model weights
    save_model(model, file_name = "ch_pinn_005eps.pt")

    #plot train loss vs epoch
    x_epochs = np.arange(1, n_epochs + 1)

    plt.plot(x_epochs, train_losses["pde"], label = "pde")
    plt.plot(x_epochs, train_losses["bc"], label = "bc")
    plt.plot(x_epochs, train_losses["ic"], label = "ic")

    plt.xlabel("Epochs")
    plt.ylabel("Train loss")
    
    plt.legend()
    plt.show()



