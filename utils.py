import torch

#initial condition function for phield phi
def initial_phi(x):
    return 0.3 * torch.cos(torch.pi * x)

#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max, epsilon):
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
    phi_ic_true = initial_phi(x_ic).detach()

    return collocation, phi_ic_true


#function to compute total system energy at a given time
def compute_energy(x, phi, phi_x, epsilon):
    density = 0.25 * (phi**2 - 1.0)**2 + 0.5 * epsilon**2 * phi_x**2 #free energy density
    return torch.trapz(density.squeeze(), x.squeeze()) 







