import torch 
import math 

# from utils_bio_2d import (
#     grad, 
#     laplacian, 
#     p_interp, 
#     p_interp_der, 
#     g_der, 
#     f_in, 
#     f_out, 
#     k
# )

k = 3 * math.sqrt(2) / 4 #paper constant

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

def p_interp(phi):
    return -0.5 * phi**3 + 1.5 * phi

#derivative of interpolating function p(phi)
def p_interp_der(phi):
    return 1.5 * (1.0 - phi ** 2)

def g(phi):
    return 0.25 * (phi**2 - 1.0) ** 2

def g_der(phi):
    return phi**3 - phi

#function to compute free energy density inside the vesicle
def f_in(psi, psi_eq: float, lambda_in: float, beta_in: float):
    return (lambda_in/2.0) * (psi - psi_eq)**2 + beta_in

#function to compute free energy density outside the vesicle
def f_out(psi, psi_eq: float, lambda_out: float, beta_out: float):
    return (lambda_out/2.0) * (psi - psi_eq)**2 + beta_out


#function to calculate the exact profile of manufactured solutions phi_ex, psi_ex
def exact_phi_psi(x, y, t):
    #we choose cosines so that normal derivatives vanish at borders (for L_x, L_y = 1)
    spatial = torch.cos(math.pi * x) * torch.cos(math.pi * y)

    phi = 0.2 + 0.3 * spatial * torch.exp(-1.0 * t)
    psi = 0.5 + 0.2 * spatial * torch.exp(-0.7 * t)

    return phi, psi

#function to compute the target potentials mu_ex, nu_ex for manufactured solutions phi_ex, psi_ex (same code of initial_phiels in utils)
def compute_mu_nu_targets(phi, psi, x, y, args):
    lap_phi = laplacian(phi, x, y)
 
    #p(phi), g(phi) and derivatives
    p_phi = p_interp(phi)
    p_phi_der = p_interp_der(phi)
    g_phi_der = g_der(phi)

    f_in_values, f_out_values = f_in(psi, args.psi_in_eq, args.lambda_in, args.beta_in), f_out(psi, args.psi_out_eq, args.lambda_out, args.beta_out) #f_in(psi), f_out(psi)

    mu_target = (
        args.lambda_surf
        * k *((1.0 / args.eps) * g_phi_der - args.eps * lap_phi)
        + 0.5 * p_phi_der * (f_in_values - f_out_values)
    )

    nu_target = (
        0.5 * (1.0 + p_phi) * args.lambda_in * (psi - args.psi_in_eq)
        + 0.5 * (1.0 - p_phi) * args.lambda_out * (psi - args.psi_out_eq)
    )

    return mu_target, nu_target


#function to compute the exact fields and auxiliary fields phi_ex, psi_ex, mu_ex, nu_ex for manufactured solution
def exact_fields_ms(x, y, t, args):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    phi_ex, psi_ex = exact_phi_psi(x, y, t)

    mu_ex, nu_ex = compute_mu_nu_targets(
        phi_ex, 
        psi_ex, 
        x, y, args
    )

    return phi_ex, mu_ex, psi_ex, nu_ex

#same function as before BUT used in IC target computation for IC loss (because here we need to decouple graphs to avoid autograd errors)
def exact_fields_ms_target(x, y, t, args):
    xt = x.detach().clone().requires_grad_(True)
    yt = y.detach().clone().requires_grad_(True)
    tt = t.detach().clone().requires_grad_(True)

    phi_ex, mu_ex, psi_ex, nu_ex = exact_fields_ms(xt, yt, tt, args)

    return (
        phi_ex.detach(),
        mu_ex.detach(),
        psi_ex.detach(),
        nu_ex.detach(),
    )


#function that returns the artificial source terms S_phi and S_psi such that exact_field are solution of the forced PDE
#phi_t + m_phi * mu - S_phi = 0
#psi_t + div(-m_psi(phi)*grad(nu)) - S_psi = 0
def manufactured_sources(x, y, t, args):

    #independent coordinates only for source construction
    xs = x.detach().clone().requires_grad_(True)
    ys = y.detach().clone().requires_grad_(True)
    ts = t.detach().clone().requires_grad_(True)
    
    #calculate exact fields
    phi_ex, mu_ex, psi_ex, nu_ex = exact_fields_ms(xs, ys, ts, args)

    #computing derivatives and useful quantities
    phi_t = grad(phi_ex, ts)
    psi_t = grad(psi_ex, ts)

    nu_x = grad(nu_ex, xs)
    nu_y = grad(nu_ex, ys)

    m_psi = 1.0 - args.m0 * ((phi_ex**2 - 1.0)**2)

    psi_curr_x = -m_psi * nu_x
    psi_curr_y = -m_psi * nu_y

    div_psi_curr = grad(psi_curr_x, xs) + grad(psi_curr_y, ys)

    #calculating source terms
    S_phi = phi_t + args.m_phi * mu_ex
    S_psi = psi_t + div_psi_curr

    return S_phi.detach(), S_psi.detach()