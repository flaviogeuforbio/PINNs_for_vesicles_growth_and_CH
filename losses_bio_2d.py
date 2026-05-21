import torch 
import math

from utils_bio_2d import grad, laplacian, f_in, f_out, g_der, p_interp, p_interp_der

k = 3 * math.sqrt(2) / 4 #paper constant

#governing equation term of the total loss (2D AC + CH + interaction term gamma*c*phi)
def pde_loss(
        x, y, t, 
        model, 
        args
):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    phi, mu, psi, nu = model(x, y, t) #model predictions of the fields

    #COMPUTING DERIVATIVES AND USEFUL QUANTITIES
    phi_t = grad(phi, t)
    psi_t = grad(psi, t)

    nu_x = grad(nu, x)
    nu_y = grad(nu, y)

    lap_phi = laplacian(phi, x, y)
 
    #p(phi), g(phi) and derivatives
    p_phi = p_interp(phi)
    p_phi_der = p_interp_der(phi)
    g_phi_der = g_der(phi)

    f_in_values, f_out_values = f_in(psi, args.psi_in_eq, args.lambda_in, args.beta_in), f_out(phi, args.psi_out_eq, args.lambda_out, args.beta_out) #f_in(psi), f_out(psi)
    m_psi = 1 - args.m0 * ((phi**2 - 1) ** 2) #psi mobility (depends on phi)

    #psi current
    psi_curr_x = - m_psi * grad(nu, x)
    psi_curr_y = - m_psi * grad(nu, y)

    #RESIDUALS
    res_phi = phi_t + args.m_phi * mu
    res_mu = mu - (args.lambda_surf * k * ((1.0 / args.eps) * g_phi_der - args.eps * lap_phi) + (1.0/2.0) * p_phi_der * (f_in_values - f_out_values))
    res_psi = psi_t + grad(psi_curr_x, x) + grad(psi_curr_y, y)
    res_nu = nu - ((1 + p_phi)/2.0 * args.lambda_in * (psi - args.psi_in_eq) + (1 - p_phi)/2.0 * args.lambda_out * (psi - args.psi_out_eq))

    loss_pde_phi = torch.mean(res_phi ** 2)
    loss_pde_mu = torch.mean(res_mu ** 2)
    loss_pde_psi = torch.mean(res_psi ** 2)
    loss_pde_nu = torch.mean(res_nu ** 2)

    loss_pde = loss_pde_phi + loss_pde_mu + loss_pde_psi + loss_pde_nu 
    return loss_pde, loss_pde_phi, loss_pde_mu, loss_pde_psi, loss_pde_nu


#initial conditions term of the total loss
def ic_loss(model, x_ic, y_ic, t_ic, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true): 
    phi_ic_pred, mu_ic_pred, psi_ic_pred, nu_ic_pred = model(x_ic, y_ic, t_ic) #calculate model predictions on ic

    ic_phi_loss = torch.mean((phi_ic_pred - phi_ic_true) ** 2)
    ic_mu_loss = torch.mean((mu_ic_pred - mu_ic_true) ** 2)
    ic_psi_loss = torch.mean((psi_ic_pred - psi_ic_true) ** 2)
    ic_nu_loss = torch.mean((nu_ic_pred - nu_ic_true) ** 2)

    ic_loss = ic_phi_loss + ic_mu_loss + ic_psi_loss + ic_nu_loss
    return ic_loss, ic_phi_loss, ic_mu_loss, ic_psi_loss, ic_nu_loss


#boundary conditions term of the total loss -> we choose to implement Neumann conditions (zero-flux, closed box)
def bc_loss(model, x_bc, y_bc, t_bc, normal): 
    x_bc.requires_grad_(True)
    y_bc.requires_grad_(True)

    phi_b, mu_b, psi_b, nu_b = model(x_bc, y_bc, t_bc) #calculating model predictions on boundaries

    #calculating gradients for each field
    phi_b_x = grad(phi_b, x_bc)
    phi_b_y = grad(phi_b, y_bc)
    
    mu_b_x = grad(mu_b, x_bc)
    mu_b_y = grad(mu_b, y_bc)

    psi_b_x = grad(psi_b, x_bc)
    psi_b_y = grad(psi_b, y_bc)

    nu_b_x = grad(nu_b, x_bc)
    nu_b_y = grad(nu_b, y_bc)

    #calculating normal to the boundary (towards outside)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]

    #calculating directional derivative on normal direction of each field
    phi_b_n = phi_b_x * nx + phi_b_y * ny
    mu_b_n = mu_b_x * nx + mu_b_y * ny
    psi_b_n = psi_b_x * nx + psi_b_y * ny
    nu_b_n = nu_b_x * nx + nu_b_y * ny

    #calculating losses
    bc_phi_loss = torch.mean(phi_b_n ** 2)
    bc_mu_loss = torch.mean(mu_b_n ** 2)
    bc_psi_loss = torch.mean(psi_b_n ** 2)
    bc_nu_loss = torch.mean(nu_b_n ** 2)


    bc_loss = bc_phi_loss + bc_mu_loss + bc_psi_loss + bc_nu_loss
    return bc_loss, bc_phi_loss, bc_mu_loss, bc_psi_loss, bc_nu_loss
