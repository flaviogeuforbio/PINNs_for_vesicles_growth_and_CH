import torch
from pathlib import Path
import math

from models import BioACCHPINN2d
from manufactured_bio_2d import exact_fields_ms_target

k = 3 * math.sqrt(2) / 4 #paper constant

#function to compute first derivative
def grad(u, x):
    return torch.autograd.grad(
        u, x, 
        grad_outputs = torch.ones_like(u), 
        create_graph = True,
        retain_graph = True
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

#function to calculate 2d integrals (over a discrete 2-dimensional grid)
def integral_2d(field, x_lin, y_lin, n_grid):
    #we want to reshape the field, because the model outputs a vector with shape (n_grid*n_grid, 1) for each field phi, c, mu
    field = field.reshape(n_grid, n_grid)

    int_y = torch.trapz(field, y_lin, dim = 1) #integrating over y first (with trapezoidal method)
    int_xy = torch.trapz(int_y, x_lin, dim = 0)

    return int_xy


#AUXILIARY FUNCTIONS ----------------------------------------------------
#interpolating function p(phi)
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

#function to compute surface free energy density (related to non-conserved field phi) 
def f_surf_density(phi, phi_x, phi_y, eps): 
    return k * (
        (1.0/eps) * g(phi)
        + 0.5 * eps * (phi_x**2 + phi_y**2)
    )

#function to compute osmotic free energy density (related to conserved concentration psi)
def f_osm_density(
    phi, 
    psi, 
    psi_in_eq: float, 
    psi_out_eq: float, 
    lambda_in: float, 
    lambda_out: float, 
    beta_in: float, 
    beta_out: float
):
    p_phi = p_interp(phi)

    f_in_val = f_in(
        psi, 
        psi_in_eq, 
        lambda_in, 
        beta_in
    )

    f_out_val = f_out(
        psi, 
        psi_out_eq, 
        lambda_out, 
        beta_out
    )

    return 0.5 * (1 + p_phi) * f_in_val + 0.5 * (1 - p_phi) * f_out_val

#function to compute total energy of the system (by now we consider F_tot = F_surf + F_osm + F_area, and we neglect F_bend for simplicity)
def compute_energy(
    phi, 
    psi, 
    phi_x, 
    phi_y, 
    x_lin, 
    y_lin, 
    n_grid, 
    args, 
    area_target = None
):
    #compute surface free energy density 
    surf_density = args.lambda_surf * f_surf_density(
        phi = phi, 
        phi_x = phi_x, 
        phi_y = phi_y, 
        eps = args.eps
    )

    #compute osmotic free energy density
    osm_density = f_osm_density(
        phi = phi, 
        psi = psi, 
        psi_in_eq = args.psi_in_eq, 
        psi_out_eq = args.psi_out_eq, 
        lambda_in = args.lambda_in, 
        lambda_out = args.lambda_out, 
        beta_in = args.beta_in, 
        beta_out = args.beta_out
    )

    surf_energy = integral_2d(surf_density, x_lin, y_lin, n_grid)
    osm_energy = integral_2d(osm_density, x_lin, y_lin, n_grid)

    total_energy = surf_energy + osm_energy

    area_energy = None
    if area_target is not None and hasattr(args, "lambda_area"):

        diffuse_area = integral_2d(
            f_surf_density(phi, phi_x, phi_y, args.eps),
            x_lin, 
            y_lin, 
            n_grid
        )

        area_energy = 0.5 * args.lambda_area * (diffuse_area - area_target) ** 2
        total_energy += area_energy

    return total_energy, surf_energy, osm_energy, area_energy




#simple IC for phi: diffused disk -> tanh
def initial_phi(x, y, eps, radius=0.28, x0=0.5, y0=0.5, pert_a=0.00, pert_mode=4): #no perturbation (perfect circle) by default
    # circular shape with angular perturbation (we don't want to start from a shape with maximized volume/surface ratio)
    dx = x - x0
    dy = y - y0
    
    r = torch.sqrt(dx**2 + dy**2)
    theta = torch.atan2(dy, dx)

    R_theta = radius * (1 + pert_a * torch.cos(pert_mode * theta))
    return torch.tanh((R_theta - r) / (math.sqrt(2.0) * eps))

#simple IC for psi: almost 'step-wise' uniform distribution with different values inside and outside
def initial_psi(phi, psi_in_0, psi_out_0):
    p_phi = p_interp(phi)
    chi_in = 0.5 * (1 + p_phi) #this factor selects inside/outside the vesicle

    return chi_in * psi_in_0 + (1.0 - chi_in) * psi_out_0

#function to compute initial condition for phi, mu, psi, nu
def initial_fields(x, y, args):
    x_clone = x.clone().detach().requires_grad_(True)
    y_clone = y.clone().detach().requires_grad_(True)

    #initial phi, initial psi
    phi_0 = initial_phi(x_clone, y_clone, args.eps)
    psi_0 = initial_psi(phi_0, args.psi_in_0, args.psi_out_0)

    #computing useful quantities and derivatives for initial mu, initial nu
    lap_phi = laplacian(
        phi_0, x_clone, y_clone
    )

    p_phi = p_interp(phi_0)
    p_phi_der = p_interp_der(phi_0)

    g_phi_der = g_der(phi_0)

    f_in_values = f_in(
        psi_0, 
        psi_eq = args.psi_in_eq, 
        lambda_in = args.lambda_in, 
        beta_in = args.beta_in
    )
    f_out_values = f_out(
        psi_0, 
        psi_eq = args.psi_out_eq, 
        lambda_out = args.lambda_out, 
        beta_out = args.beta_out
    )

    mu_0 = args.lambda_surf * k * ((1 / args.eps) * g_phi_der - args.eps * lap_phi) + 0.5 * p_phi_der * (f_in_values - f_out_values)
    nu_0 = 0.5 * (1 + p_phi) * args.lambda_in * (psi_0 - args.psi_in_eq) + 0.5 * (1 - p_phi) * args.lambda_out * (psi_0 - args.psi_out_eq)

    return phi_0.detach(), mu_0.detach(), psi_0.detach(), nu_0.detach()


#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(args, device, ic_fn = None):
    N_pde, N_bc, N_ic = args.n_pde, args.n_bc, args.n_ic
    T_max = args.segment_length
    L_x, L_y = args.lx, args.ly

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
        "y_ic": y_ic,
        "t_ic": t_ic
    }

    #setting the true initial condition or the last prediction of the previous segment
    if ic_fn is None:
        phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = (
            exact_fields_ms_target(x_ic, y_ic, t_ic, args) if args.manufactured else initial_fields(x_ic, y_ic, args)
        )
        phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = phi_ic_true.to(device), mu_ic_true.to(device), psi_ic_true.to(device), nu_ic_true.to(device)

    else:
        phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = ic_fn(x_ic, y_ic)

    return collocation, phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true


#function to implement adaptive resampling after a first (warmup) training phase (two-stage adaptive resampling)
def adaptive_resample_pde_points(
    model, 
    args, 
    device, 
):
    model.eval() #we are stopping the training to resample collocation points

    n_adapt = int(args.adaptive_frac * args.n_pde)
    n_uniform = args.n_pde - n_adapt
    n_candidates = args.n_candidates_resamp

    #candidate points
    x_cand = torch.rand(n_candidates, 1, device = device) * args.lx
    y_cand = torch.rand(n_candidates, 1, device = device) * args.ly
    t_cand = torch.rand(n_candidates, 1, device = device) * args.segment_length

    x_cand.requires_grad_(True)
    y_cand.requires_grad_(True)
    t_cand.requires_grad_(True)

    #compute fields prediction
    phi, mu, psi, nu = model(x_cand, y_cand, t_cand)

    #compute residuals 
    res_phi, res_mu, res_psi, res_nu = pde_residuals(
        x_cand, y_cand, t_cand, 
        phi, 
        psi, 
        mu,
        nu,
        args
    )

    #compute the residual score to detect the 'hardest' regions for the net
    score = (
        torch.abs(res_psi) #we weight res_psi more because we observed the net struggling in this specific term (pde psi loss term) 
        + 0.25 * torch.abs(res_mu)
        + 0.25 * torch.abs(res_phi)
        + 0.25 * torch.abs(res_nu)
    )
    score = score.detach().flatten() #1d tensor out of any comp. graph

    #select highest-score points 
    topk_idx = torch.topk(score, k = n_adapt, largest = True).indices

    x_adapt = x_cand.detach()[topk_idx]
    y_adapt = y_cand.detach()[topk_idx]
    t_adapt = t_cand.detach()[topk_idx]

    #uniform points for global coverage
    x_uni = torch.rand(n_uniform, 1, device = device) * args.lx
    y_uni = torch.rand(n_uniform, 1, device = device) * args.ly
    t_uni = torch.rand(n_uniform, 1, device = device) * args.segment_length

    #merging uniform points and adaptive points
    x_pde = torch.cat([x_adapt, x_uni], dim = 0).detach()
    y_pde = torch.cat([y_adapt, y_uni], dim = 0).detach()
    t_pde = torch.cat([t_adapt, t_uni], dim = 0).detach()

    print(
        f"Adaptive PDE resampling done | "
        f"N_candidates={n_candidates} | "
        f"N_adapt={n_adapt} | "
        f"N_uniform={n_uniform} | "
        f"score_max={score.max().item():.4e} | "
        f"score_mean={score.mean().item():.4e}"
    )

    model.train() #continuing the training loop

    return x_pde, y_pde, t_pde


#function to load the PINN best model
def load_model(model_checkpoint: str, hidden_layers: int, hidden_dim: int):
    model = BioACCHPINN2d(hidden_layers = hidden_layers, hidden_dim = hidden_dim)
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

        model = BioACCHPINN2d(hidden_layers, hidden_dim)
        model.load_state_dict(torch.load(check_path, map_location=device))
        model = model.to(device)
        model.eval()

        models.append(model)

    return models




#function to generate initial condition for intermediate sequence model (using last temporal prediction of previous segment model)
def make_ic_from_previous_model(previous_model, segment_length, device, args, recompute_potentials: bool = False):
    previous_model.eval() #set in evaluation mode

    #build the function with the specific previous_model
    def ic_fn(x, y):
        x_eval = x.detach().to(device)
        y_eval = y.detach().to(device)
        t_eval = torch.full_like(x_eval, float(segment_length), device=device)

        if recompute_potentials:
            x_eval = x_eval.requires_grad_(True)
            y_eval = y_eval.requires_grad_(True)

            phi_ic_true, _, psi_ic_true, _ = previous_model(x_eval, y_eval, t_eval)

             #computing useful quantities and derivatives for initial mu, initial nu
            lap_phi = laplacian(
                phi_ic_true, x_eval, y_eval
            )

            p_phi = p_interp(phi_ic_true)
            p_phi_der = p_interp_der(phi_ic_true)

            g_phi_der = g_der(phi_ic_true)

            f_in_values = f_in(
                psi_ic_true, 
                psi_eq = args.psi_in_eq, 
                lambda_in = args.lambda_in, 
                beta_in = args.beta_in
            )
            f_out_values = f_out(
                psi_ic_true, 
                psi_eq = args.psi_out_eq, 
                lambda_out = args.lambda_out, 
                beta_out = args.beta_out
            )

            mu_ic_true = args.lambda_surf * k * ((1 / args.eps) * g_phi_der - args.eps * lap_phi) + 0.5 * p_phi_der * (f_in_values - f_out_values)
            nu_ic_true = 0.5 * (1 + p_phi) * args.lambda_in * (psi_ic_true - args.psi_in_eq) + 0.5 * (1 - p_phi) * args.lambda_out * (psi_ic_true - args.psi_out_eq)

        else:
            with torch.no_grad():
                phi_ic_true, mu_ic_true, psi_ic_true, nu_ic_true = previous_model(x_eval, y_eval, t_eval)

        return phi_ic_true.detach().to(device), mu_ic_true.detach().to(device), psi_ic_true.detach().to(device), nu_ic_true.detach().to(device)
    
    return ic_fn

#function to infer (phi, mu, psi, nu) from the networks ensemble (for time windowing)
def predict_windowed(models, x, y, t_global, segment_length, device):
    #compute the segment to use 
    segment_idx = int(min(
        t_global // segment_length, 
        len(models) - 1
    ))

    tau = t_global - segment_idx * segment_length #compute local tau
    t_local = torch.full_like(x, float(tau), device=device)

    model = models[segment_idx]
    model.eval()

    return model(x, y, t_local) #phi, mu, psi, nu


#helper to compute pde residuals
def pde_residuals(
        x, y, t,
        phi, 
        psi, 
        mu, 
        nu,
        args
):
    #computing derivatives and useful quantities
    phi_t = grad(phi, t)
    psi_t = grad(psi, t)

    nu_x = grad(nu, x)
    nu_y = grad(nu, y)

    lap_phi = laplacian(phi, x, y)
 
    #p(phi), g(phi) and derivatives
    p_phi = p_interp(phi)
    p_phi_der = p_interp_der(phi)
    g_phi_der = g_der(phi)

    f_in_values, f_out_values = f_in(psi, args.psi_in_eq, args.lambda_in, args.beta_in), f_out(psi, args.psi_out_eq, args.lambda_out, args.beta_out) #f_in(psi), f_out(psi)
    m_psi = 1 - args.m0 * ((phi**2 - 1) ** 2) #psi mobility (depends on phi)

    #psi current
    psi_curr_x = - m_psi * nu_x
    psi_curr_y = - m_psi * nu_y

    #residuals
    res_phi = phi_t + args.m_phi * mu
    res_mu = mu - (args.lambda_surf * k * ((1.0 / args.eps) * g_phi_der - args.eps * lap_phi) + (1.0/2.0) * p_phi_der * (f_in_values - f_out_values))
    res_psi = psi_t + grad(psi_curr_x, x) + grad(psi_curr_y, y)
    res_nu = nu - ((1 + p_phi)/2.0 * args.lambda_in * (psi - args.psi_in_eq) + (1 - p_phi)/2.0 * args.lambda_out * (psi - args.psi_out_eq))
    
    return res_phi, res_mu, res_psi, res_nu