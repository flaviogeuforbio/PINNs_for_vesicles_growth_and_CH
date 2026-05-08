import torch

#initial condition function for phield phi
def initial_phi(x):
    return 0.2 + 0.15 * torch.cos(torch.pi * x)

#initial condition for concentration phield c
def initial_c(x):
    return 0.1 + 0.1 * torch.cos(2.0 * torch.pi * x) #trying this new IC for conserved field!

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

    c0_xx = second_derivative(c_0, x)
    mu_0 = c_0**3 - c_0 - eps_c**2 * c0_xx + gamma * phi_0 #initial mu_0 from c_0

    return phi_0.detach(), c_0.detach(), mu_0.detach()


#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max, eps_c, gamma, device):
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

    #setting the true initial condition
    phi_ic_true, c_ic_true, mu_ic_true = initial_fields(x_ic, eps_c, gamma)
    phi_ic_true, c_ic_true, mu_ic_true = phi_ic_true.to(device), c_ic_true.to(device), mu_ic_true.to(device)

    return collocation, phi_ic_true, c_ic_true, mu_ic_true

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