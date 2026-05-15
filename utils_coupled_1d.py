from pathlib import Path
import torch

from models import CoupledACCHPINN1d

#initial condition function for phield phi
def initial_phi(x):
    # return 0.2 + 0.15 * torch.cos(torch.pi * x)
    return 0.2 + 0.12 * torch.cos(torch.pi * x) + 0.04 * torch.cos(2 * torch.pi * x)

#initial condition for concentration phield c
def initial_c(x):
    # return 0.1 + 0.1 * torch.cos(2.0 * torch.pi * x) #trying this new IC for conserved field!
    return 0.1 + 0.08 * torch.cos(2 * torch.pi * x) + 0.03 * torch.cos(4 * torch.pi * x)

#utility to compute second derivatives
def second_derivative(y, x):
    y_x = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]

    y_xx = torch.autograd.grad(
        y_x, x,
        grad_outputs=torch.ones_like(y_x),
        create_graph=True,
    )[0]

    return y_xx


#initial condition function for c, mu
def initial_fields(x, eps_c, gamma):
    x_clone = x.clone().detach().requires_grad_(True) #to avoid autograd errors

    #calculate initial c,phi
    c_0 = initial_c(x_clone) 
    phi_0 = initial_phi(x_clone)

    c0_xx = second_derivative(c_0, x_clone)
    mu_0 = c_0**3 - c_0 - eps_c**2 * c0_xx + gamma * phi_0 #initial mu_0 from c_0

    return phi_0.detach(), c_0.detach(), mu_0.detach()


#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max, eps_c, gamma, device, ic_fn = None):
    #pde collocation points
    x_pde = torch.rand(size = (N_pde, 1), device = device) * L
    t_pde = torch.rand(size = (N_pde, 1), device = device) * T_max

    #bc collocation points 
    x_bc_right = torch.ones(size = (N_bc // 2, 1), device = device) * L
    x_bc_left = torch.zeros(size = (N_bc // 2, 1), device = device)
    x_bc = torch.cat([x_bc_right, x_bc_left], dim = 0)
    t_bc = torch.rand(size = (N_bc, 1), device = device) * T_max

    #ic collocation points
    x_ic = torch.rand(size = (N_ic, 1), device = device) * L
    t_ic = torch.zeros(size = (N_ic, 1), device = device)

    #creating dictionary with all collocation points (pde, ic, bc)
    collocation = {
        "x_pde": x_pde,
        "t_pde": t_pde,
        "x_bc": x_bc,
        "t_bc": t_bc,
        "x_ic": x_ic, 
        "t_ic": t_ic
    }

    #setting the true initial condition or the last prediction of the previous segment
    if ic_fn is None:
        phi_ic_true, c_ic_true, mu_ic_true = initial_fields(x_ic, eps_c, gamma)
        phi_ic_true, c_ic_true, mu_ic_true = phi_ic_true.to(device), c_ic_true.to(device), mu_ic_true.to(device)

    else:
        phi_ic_true, c_ic_true, mu_ic_true = ic_fn(x_ic)

    return collocation, phi_ic_true, c_ic_true, mu_ic_true

#function to load the PINN best model
def load_model(model_checkpoint: str, hidden_layers: int, hidden_dim: int):
    model = CoupledACCHPINN1d(hidden_layers = hidden_layers, hidden_dim = hidden_dim)
    model.load_state_dict(torch.load(model_checkpoint))

    return model

#function to load the PINN models (time-windowing)
def load_models(check_dir: Path, hidden_layers: int, hidden_dim: int, device):
    models = []

    check_paths = sorted(
        check_dir.glob("segment_*.pt"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    print("Loading segment models in order:")
    for check_path in check_paths:
        print("  ", check_path.name)

        model = CoupledACCHPINN1d(hidden_layers, hidden_dim)
        model.load_state_dict(torch.load(check_path, map_location=device))
        model = model.to(device)
        model.eval()

        models.append(model)

    return models


#function to generate initial condition for intermediate sequence model (using last temporal prediction of previous segment model)
def make_ic_from_previous_model(previous_model, segment_length, device, args, recompute_mu: bool = False):
    previous_model.eval() #set in evaluation mode

    #build the function with the specific previous_model
    def ic_fn(x):
        x_eval = x.detach().to(device)
        t_eval = torch.full_like(x_eval, float(segment_length), device=device)

        if recompute_mu:
            x_eval = x_eval.requires_grad_(True)
            phi_ic_true, c_ic_true, _ = previous_model(x_eval, t_eval) #without torch.no_grad (to compute derivatives)

            c_ic_x = torch.autograd.grad(
                c_ic_true, 
                x_eval, 
                grad_outputs = torch.ones_like(c_ic_true),
                create_graph = True
            )[0]

            c_ic_xx = torch.autograd.grad(
                c_ic_x, 
                x_eval, 
                grad_outputs = torch.ones_like(c_ic_x),
                create_graph = True
            )[0]

            mu_ic_true = (c_ic_true**3) - c_ic_true - (args.eps_c**2)*c_ic_xx + args.gamma * phi_ic_true

        else:
            with torch.no_grad():
                phi_ic_true, c_ic_true, mu_ic_true = previous_model(x_eval, t_eval)

        return phi_ic_true.detach().to(device), c_ic_true.detach().to(device), mu_ic_true.detach().to(device)
    
    return ic_fn

#function to infer (phi, c, mu) from the networks ensemble (for time windowing)
def predict_windowed(models, x, t_global, segment_length, device):
    #compute the segment to use 
    segment_idx = int(min(
        t_global // segment_length, 
        len(models) - 1
    ))

    tau = t_global - segment_idx * segment_length #compute local tau
    t_local = torch.full_like(x, float(tau), device=device)

    model = models[segment_idx]
    model.eval()

    return model(x, t_local) #phi, c, mu


#function to compute total system energy at a given time
def compute_energy(x, phi, c, phi_x, c_x, eps_phi, eps_c, gamma):
    density = (
        0.5 * eps_phi**2 * phi_x**2
        + 0.25 * (phi**2 - 1.0) ** 2
        + 0.5 * eps_c**2 * c_x**2
        + 0.25 * (c**2 - 1.0) ** 2
        + gamma * phi * c
    )

    return torch.trapz(density.squeeze(), x.squeeze())

#utility to compute integrals -> to calculate mean phi and total mass then
def compute_integral(x, y):
    return torch.trapz(y.squeeze(), x.squeeze())