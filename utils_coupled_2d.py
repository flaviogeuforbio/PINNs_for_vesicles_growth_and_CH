import torch 

from models import CoupledACCHPINN2d

#function to compute first derivative
def grad(u, x):
    return torch.autograd.grad(
        u, x, 
        grad_outputs = torch.ones_like(u), 
        create_graph = True
    )[0]

#function to compute laplacian
def laplacian(u, x, y):
    #first derivatives
    u_x = grad(u, x)
    u_y = grad(u, y)

    #second derivatives
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    #summing them up
    return u_xx + u_yy


#initial condition function for phield phi
def initial_phi(x, y):
    return 0.2 + 0.15 * torch.cos(torch.pi * x) * torch.cos(torch.pi * y)

#initial condition for concentration phield c
def initial_c(x, y):
    return 0.1 + 0.1 * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)

#initial condition function for c, mu
def initial_fields(x, y, eps_c, gamma):
    x_clone = x.clone().detach().requires_grad_(True) #to avoid autograd error
    y_clone = y.clone().detach().requires_grad_(True)

    #calculate initial c,phi
    c_0 = initial_c(x_clone, y_clone) 
    phi_0 = initial_phi(x_clone, y_clone)

    lap_c0 = laplacian(c_0, x_clone, y_clone)
    mu_0 = c_0**3 - c_0 - eps_c**2 * lap_c0 + gamma * phi_0 #initial mu_0 from c_0

    return phi_0.detach(), c_0.detach(), mu_0.detach()


#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L_x, L_y, T_max, eps_c, gamma, device, ic_fn = None):
    #pde collocation points
    x_pde = torch.rand(size = (N_pde, 1), device = device) * L_x
    y_pde = torch.rand(size = (N_pde, 1), device = device) * L_y
    t_pde = torch.rand(size = (N_pde, 1), device = device) * T_max

    #ic collocation points
    x_ic = torch.rand(size = (N_ic, 1), device = device) * L_x
    y_ic = torch.rand(size = (N_ic, 1), device = device) * L_y
    t_ic = torch.zeros(size = (N_ic, 1), device = device)

    #bc collocation points 
    N_side = N_bc // 4

    #left, normal = (-1, 0)
    x_left = torch.zeros(size = (N_side, 1), device = device)
    y_left = torch.rand(size = (N_side, 1), device = device) * L_y
    t_left = torch.rand(size = (N_side, 1), device = device) * T_max
    n_left = torch.tensor([-1.0, 0.0], device = device).repeat(N_side, 1)

    #right, normal = (1, 0)
    x_right = torch.ones(size = (N_side, 1), device = device) * L_x
    y_right = torch.rand(size = (N_side, 1), device = device) * L_y
    t_right = torch.rand(size = (N_side, 1), device = device) * T_max
    n_right = torch.tensor([1.0, 0], device = device).repeat(N_side, 1)

    #down, normal = (0, -1)
    x_down = torch.rand(size = (N_side, 1), device = device) * L_x
    y_down = torch.zeros(size = (N_side, 1), device = device)
    t_down = torch.rand(size = (N_side, 1), device = device) * T_max
    n_down = torch.tensor([0.0, -1.0], device = device).repeat(N_side, 1)

    #up, normal = (0, 1)
    x_up = torch.rand(size = (N_side, 1), device = device) * L_x
    y_up = torch.ones(size = (N_side, 1), device = device) * L_y
    t_up = torch.rand(size = (N_side, 1), device = device) * T_max
    n_up = torch.tensor([0.0, 1.0], device = device).repeat(N_side, 1)

    #putting all together
    x_bc = torch.cat([x_left, x_right, x_down, x_up], dim = 0)
    y_bc = torch.cat([y_left, y_right, y_down, y_up], dim = 0)
    t_bc = torch.cat([t_left, t_right, t_down, t_up], dim = 0)
    normal_bc = torch.cat([n_left, n_right, n_down, n_up], dim = 0)

    #creating dictionary with all collocation points (pde, ic, bc)
    collocation = {
        "x_pde": x_pde,
        "y_pde": y_pde, 
        "t_pde": t_pde,
        "x_bc": x_bc,
        "y_bc": y_bc, 
        "t_bc": t_bc,
        "normal_bc": normal_bc,
        "x_ic": x_ic, 
        "y_ic": y_bc,
        "t_ic": t_ic
    }

    #setting the true initial condition or the last prediction of the previous segment
    if ic_fn is None:
        phi_ic_true, c_ic_true, mu_ic_true = initial_fields(x_ic, y_ic, eps_c, gamma)
        phi_ic_true, c_ic_true, mu_ic_true = phi_ic_true.to(device), c_ic_true.to(device), mu_ic_true.to(device)

    else:
        phi_ic_true, c_ic_true, mu_ic_true = ic_fn(x_ic, y_ic)

    return collocation, phi_ic_true, c_ic_true, mu_ic_true


#function to load the PINN best model
def load_model(model_checkpoint: str, hidden_layers: int, hidden_dim: int):
    model = CoupledACCHPINN2d(hidden_layers = hidden_layers, hidden_dim = hidden_dim)
    model.load_state_dict(torch.load(model_checkpoint))

    return model

#function to calculate 2d integrals (over a discrete 2-dimensional grid)
def integral_2d(field, x_lin, y_lin, n_grid):
    #we want to reshape the field, because the model outputs a vector with shape (n_grid*n_grid, 1) for each field phi, c, mu
    field = field.reshape(n_grid, n_grid)

    int_y = torch.trapz(field, y_lin, dim = 1) #integrating over y first (with trapezoidal method)
    int_xy = torch.trapz(int_y, x_lin, dim = 0)

    return int_xy


#function to compute total energy of the system 
def compute_energy(
    phi, c, 
    phi_x, phi_y, 
    c_x, c_y, 
    x_lin, y_lin, 
    n_grid, 
    eps_phi, eps_c, 
    gamma
):
    
    density = (
        0.5 * eps_phi**2 * (phi_x**2 + phi_y**2) 
        + 0.25 * (phi**2 - 1.0) ** 2
        + 0.5 * eps_c**2 * (c_x**2 + c_y**2)
        + 0.25 * (c**2 - 1.0) ** 2
        + gamma * c * phi
    )

    return integral_2d(density, x_lin, y_lin, n_grid)
