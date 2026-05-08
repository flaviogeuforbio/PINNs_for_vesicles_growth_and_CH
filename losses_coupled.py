import torch

#governing equation term of the total loss (1D AC + CH + interaction term gamma*c*phi)
def pde_loss(
        x, t,
        model, 
        M_phi, M_c, 
        eps_phi, eps_c, 
        gamma #coupling parameter
):
    x.requires_grad_(True)
    t.requires_grad_(True)

    phi, c, mu = model(x, t) #calculating phields prediction

    #computing derivatives
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

    c_t = torch.autograd.grad(
        c, t, 
        grad_outputs = torch.ones_like(c),
        create_graph = True
    )[0]

    c_x = torch.autograd.grad(
        c, x, 
        grad_outputs = torch.ones_like(c), 
        create_graph = True
    )[0]

    c_xx = torch.autograd.grad(
        c_x, x, 
        grad_outputs = torch.ones_like(c_x), 
        create_graph = True
    )[0]

    mu_x = torch.autograd.grad(
        mu, x, 
        grad_outputs = torch.ones_like(mu), 
        create_graph = True
    )[0]

    mu_xx = torch.autograd.grad(
        mu_x, x, 
        grad_outputs = torch.ones_like(mu_x),
        create_graph = True
    )[0]

    #calculating residuals
    res_phi = phi_t + M_phi * (phi**3 - phi - eps_phi**2 * phi_xx + gamma * c)
    res_c = c_t - M_c * mu_xx
    res_mu = mu - (c**3 - c - eps_c**2 * c_xx + gamma * phi)

    #calculating loss terms
    mse_phi = torch.mean(res_phi**2)
    mse_c = torch.mean(res_c**2)
    mse_mu = torch.mean(res_mu**2)

    return mse_phi + mse_c + mse_mu, mse_phi, mse_c, mse_mu


#initial conditions term of the total loss
def ic_loss(model, x_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true):
    phi_ic_pred, c_ic_pred, mu_ic_pred = model(x_ic, t_ic) #calculate predicted concentration on IC
    
    #residual -> loss terms
    mse_phi = torch.mean((phi_ic_pred - phi_ic_true)**2)
    mse_c = torch.mean((c_ic_pred - c_ic_true)**2)
    mse_mu = torch.mean((mu_ic_pred - mu_ic_true)**2)

    return mse_phi + mse_c + mse_mu, mse_phi, mse_c, mse_mu


#boundary conditions term of the total loss -> we choose to implement Neumann conditions (zero-flux, closed box)
def bc_loss(model, x_bc, t_bc):
    x_bc.requires_grad_(True)

    phi_bc, c_bc, mu_bc = model(x_bc, t_bc) #calculate predicted concentration on boundaries

    #computing derivatives
    phi_bc_x = torch.autograd.grad(
        phi_bc, x_bc, 
        grad_outputs = torch.ones_like(phi_bc), 
        create_graph = True
    )[0]

    c_bc_x = torch.autograd.grad(
        c_bc, x_bc, 
        grad_outputs = torch.ones_like(c_bc), 
        create_graph = True
    )[0]

    mu_bc_x = torch.autograd.grad(
        mu_bc, x_bc, 
        grad_outputs = torch.ones_like(mu_bc),
        create_graph = True
    )[0]

    mse_phi = torch.mean(phi_bc_x**2)
    mse_c = torch.mean(c_bc_x**2)
    mse_mu = torch.mean(mu_bc_x**2)

    return mse_phi + mse_c + mse_mu, mse_phi, mse_c, mse_mu