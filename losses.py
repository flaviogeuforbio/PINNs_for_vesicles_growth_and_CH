import torch

#governing equation term of the total loss
def pde_loss(model, x, t, M, epsilon):
    x.requires_grad_(True)
    t.requires_grad_(True)

    #calculate output predictions 
    c, mu = model(x, t)

    #calculate derivatives for c
    dc_dt = torch.autograd.grad(
        c, t, 
        grad_outputs = torch.ones_like(c),
        create_graph = True
    )[0]

    dc_dx = torch.autograd.grad(
        c, x, 
        grad_outputs = torch.ones_like(c),
        create_graph = True
    )[0]
    dc2_dx2 = torch.autograd.grad(
        dc_dx, x,
        grad_outputs = torch.ones_like(dc_dx),
        create_graph = True
    )[0]

    #calculate derivatives for mu
    dmu_dx = torch.autograd.grad(
        mu, x, 
        grad_outputs = torch.ones_like(mu),
        create_graph = True
    )[0]
    dmu2_dx2 = torch.autograd.grad(
        dmu_dx, x, 
        grad_outputs = torch.ones_like(dmu_dx),
        create_graph = True
    )[0]

    #physical residues
    res_c = dc_dt - M * dmu2_dx2 #mass conservation
    res_mu = mu - (c**3 - c) + (epsilon**2)*dc2_dx2 #chemical potential equation 

    #mean squared error calculation 
    mse_pde_c = torch.mean(res_c**2)
    mse_pde_mu = torch.mean(res_mu**2)

    return mse_pde_c + mse_pde_mu, mse_pde_c, mse_pde_mu

#initial conditions term of the total loss
def ic_loss(model, x_ic, t_ic, c_ic_true, mu_ic_true):
    c_ic_pred, mu_ic_pred = model(x_ic, t_ic) #calculate predicted concentration on IC

    mse_c = torch.mean((c_ic_pred - c_ic_true)**2)
    mse_mu = torch.mean((mu_ic_pred - mu_ic_true)**2)

    return mse_c, mse_mu

#boundary conditions term of the total loss -> we choose to implement Neumann conditions (zero-flux, closed box)
def bc_loss(model, x_bc, t_bc):
    x_bc.requires_grad_(True)

    c_bc, mu_bc = model(x_bc, t_bc) #calculate predicted concentration on boundaries

    dc_dx_bc = torch.autograd.grad(
        c_bc, x_bc, 
        grad_outputs = torch.ones_like(c_bc),
        create_graph = True
    )[0]

    dmu_dx_bc = torch.autograd.grad(
        mu_bc, x_bc, 
        grad_outputs = torch.ones_like(mu_bc),
        create_graph = True
    )[0]

    mse_c_bc = torch.mean(dc_dx_bc**2)
    mse_mu_bc = torch.mean(dmu_dx_bc**2)

    return mse_c_bc + mse_mu_bc

