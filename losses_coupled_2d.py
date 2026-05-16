import torch 
from utils_coupled_2d import grad, laplacian

#governing equation term of the total loss (1D AC + CH + interaction term gamma*c*phi)
def pde_loss(
        x, y, t, 
        model, 
        M_phi, M_c, 
        eps_phi, eps_c, 
        gamma
):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    #calculate predictions
    phi, c, mu = model(x, y, t)

    #first derivatives
    phi_t = grad(phi, t)
    c_t = grad(c, t)

    #laplacians 
    lap_phi = laplacian(phi, x, y)
    lap_c = laplacian(c, x, y)
    lap_mu = laplacian(mu, x, y)

    #calculating residuals for phi, c, mu
    r_phi = phi_t + M_phi * (phi**3 - phi - eps_phi**2 * lap_phi + gamma * c)
    r_c = c_t - M_c * lap_mu
    r_mu = mu - (c**3 - c - eps_c**2 * lap_c + gamma * phi)

    pde_phi_loss = torch.mean(r_phi**2)
    pde_c_loss = torch.mean(r_c**2)
    pde_mu_loss = torch.mean(r_mu**2)

    pde_loss = pde_phi_loss + pde_c_loss + pde_mu_loss
    return pde_loss, pde_phi_loss, pde_c_loss, pde_mu_loss


#initial conditions term of the total loss
def ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, c_ic_true, mu_ic_true): 
    phi_ic_pred, c_ic_pred, mu_ic_pred = model(x_ic, y_ic, t_ic) #calculate model predictions on ic

    ic_phi_loss = torch.mean((phi_ic_pred - phi_ic_true) ** 2)
    ic_c_loss = torch.mean((c_ic_pred - c_ic_true) ** 2)
    ic_mu_loss = torch.mean((mu_ic_pred - mu_ic_true) ** 2)

    ic_loss = ic_phi_loss + ic_c_loss + ic_mu_loss
    return ic_loss, ic_phi_loss, ic_c_loss, ic_mu_loss


#boundary conditions term of the total loss -> we choose to implement Neumann conditions (zero-flux, closed box)
def bc_loss(model, x_bc, y_bc, t_bc, normal): 
    x_bc.requires_grad_(True)
    y_bc.requires_grad_(True)

    phi_b, c_b, mu_b = model(x_bc, y_bc, t_bc) #calculating model predictions on boundaries

    #calculating gradients for each field
    phi_b_x = grad(phi_b, x_bc)
    phi_b_y = grad(phi_b, y_bc)

    c_b_x = grad(c_b, x_bc)
    c_b_y = grad(c_b, y_bc)
    
    mu_b_x = grad(mu_b, x_bc)
    mu_b_y = grad(mu_b, y_bc)

    #calculating normal to the boundary (towards outside)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]

    #calculating directional derivative on normal direction of each field
    phi_n = phi_b_x * nx + phi_b_y * ny
    c_n = c_b_x * nx + c_b_y * ny
    mu_n = mu_b_x * nx + mu_b_y * ny

    #calculating losses
    bc_phi_loss = torch.mean(phi_n ** 2)
    bc_c_loss = torch.mean(c_n ** 2)
    bc_mu_loss = torch.mean(mu_n ** 2)

    bc_loss = bc_phi_loss + bc_c_loss + bc_mu_loss
    return bc_loss, bc_phi_loss, bc_c_loss, bc_mu_loss
