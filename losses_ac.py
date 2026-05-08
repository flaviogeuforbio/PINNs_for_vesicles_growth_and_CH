import torch

#governing equation term of the total loss
def pde_loss(model, x, t, M, epsilon):
    x.requires_grad_(True)
    t.requires_grad_(True)

    #calculate phield prediction
    phi = model(x, t)

    #calculate derivatives for c
    phi_t = torch.autograd.grad(
        phi, t, 
        grad_outputs = torch.ones_like(phi),
        create_graph = True
    )[0]

    phi_x = torch.autograd.grad(
        phi, x, 
        grad_outputs = torch.ones_like(phi),
        create_graph = True
    )[0]

    phi_xx = torch.autograd.grad(
        phi_x, x, 
        grad_outputs = torch.ones_like(phi_x),
        create_graph = True
    )[0]

    residual = phi_t - epsilon**2 * phi_xx + phi**3 - phi

    return torch.mean(residual ** 2)

#initial conditions term of the total loss
def ic_loss(model, x_ic, t_ic, phi_ic_true):
    phi_ic_pred = model(x_ic, t_ic) #calculate predicted concentration on IC
    residual = phi_ic_pred - phi_ic_true
    return torch.mean(residual ** 2)

#boundary conditions term of the total loss -> we choose to implement Neumann conditions (zero-flux, closed box)
def bc_loss(model, x_bc, t_bc):
    x_bc.requires_grad_(True)

    phi_bc = model(x_bc, t_bc) #calculate predicted concentration on boundaries

    phi_bc_x = torch.autograd.grad(
        phi_bc, x_bc, 
        grad_outputs = torch.ones_like(phi_bc), 
        create_graph = True
    )[0]

    residual = phi_bc_x

    return torch.mean(residual ** 2)

