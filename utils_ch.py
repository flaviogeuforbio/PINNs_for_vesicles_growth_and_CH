import torch

#initial condition function for concentration c
def initial_c(x):
    return 0.2 + 0.15 * torch.cos(torch.pi * x) #trying a new 'asymmetric' IC

#initial condition function for c, mu
def initial_c_mu(x, epsilon):
    x_clone = x.clone().detach().requires_grad_(True) #to avoid autograd errors

    c_0 = initial_c(x_clone) #calculate initial c
    c0_x = torch.autograd.grad(c_0, x_clone, grad_outputs = torch.ones_like(c_0), create_graph = True, retain_graph = True)[0]
    c0_xx = torch.autograd.grad(c0_x, x_clone, grad_outputs = torch.ones_like(c0_x), create_graph = True, retain_graph = True)[0]
    
    mu_0 = c_0**3 - c_0 - epsilon**2 * c0_xx

    return c_0.detach(), mu_0.detach()

#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max, epsilon, device):
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
    c_ic_true, mu_ic_true = initial_c_mu(x_ic, epsilon)
    c_ic_true, mu_ic_true = c_ic_true.to(device), mu_ic_true.to(device)

    return collocation, c_ic_true, mu_ic_true


#function to compute total system mass at a given time
def compute_mass(x, c):
    return torch.trapz(c.squeeze(), x.squeeze()) #trapezoidal rule

#function to compute total system energy at a given time
def compute_energy(x, c, c_x, epsilon):
    density = 0.25 * (c**2 - 1.0)**2 + 0.5 * epsilon**2 * c_x**2 #free energy density
    return torch.trapz(density.squeeze(), x.squeeze()) 


# #function to compute the estimated (inverse) weight of a specific loss term (gradient norm)
# def get_gradient_norm(loss, model):

#     #calculate gradients with respect to model parameters and retain graph (it will be used in the training backward pass)
#     grads = torch.autograd.grad(
#         outputs = loss, 
#         inputs = model.parameters(),
#         grad_outputs = torch.ones_like(loss),
#         retain_graph = True, 
#         create_graph = False, 
#         allow_unused = True #if some parameters are not used in the model
#     )

#     norm_sq = 0.0 #accumulating squared gradients norm
#     for g in grads:
#         if g is not None:
#             norm_sq += torch.sum(g ** 2)

#     return norm_sq.item()


# def add_confined_adaptive_points(model, curr_T_max, L, M, epsilon, n_new_points = 2000, n_candidates = 10000, spread = 0.02):
#     model.eval()

#     #generate candidates 
#     x_cand = torch.rand(size = (n_candidates, 1)) * L
#     t_cand = torch.rand(size = (n_candidates, 1)) * curr_T_max
#     x_cand.requires_grad = True
#     t_cand.requires_grad = True

#     pred_c, pred_mu = model(x_cand, t_cand) #calculating predictions

#     #calculating residues
#     c_t = torch.autograd.grad(pred_c, t_cand, grad_outputs=torch.ones_like(pred_c), create_graph=True)[0]
#     mu_x = torch.autograd.grad(pred_mu, x_cand, grad_outputs=torch.ones_like(pred_mu), create_graph=True)[0]
#     mu_xx = torch.autograd.grad(mu_x, x_cand, grad_outputs=torch.ones_like(mu_x), create_graph=True)[0]
#     c_x = torch.autograd.grad(pred_c, x_cand, grad_outputs=torch.ones_like(pred_c), create_graph=True)[0]
#     c_xx = torch.autograd.grad(c_x, x_cand, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]

#     res_c = torch.abs(c_t - M * mu_xx)
#     mu_true = (pred_c**3 - pred_c) - (epsilon**2)*c_xx
#     res_mu = torch.abs(pred_mu - mu_true)

#     total_residual = res_c + res_mu 

#     #selecting top k points by tot res value
#     n_centers = n_new_points // 10
#     _, topk_indices = torch.topk(total_residual.flatten(), n_centers) #-> these top k points will be the centers of the gaussian generated points
#     x_centers = x_cand[topk_indices].detach()
#     t_centers = t_cand[topk_indices].detach() 

#     #gaussian generation of new points around top k residual points (the idea is to add a confined gaussian noise)
#     random_center_indices = torch.randint(0, n_centers, (n_new_points, ))

#     x_new_base = x_centers[random_center_indices]
#     t_new_base = t_centers[random_center_indices]

#     x_new = x_new_base + torch.randn(size = (n_new_points, 1)) * spread
#     t_new = t_new_base + torch.randn(size = (n_new_points, 1)) * spread

#     #clamping new points to make sure they belong to the domain
#     x_new = torch.clamp(x_new, min = 0.0, max = L)
#     t_new = torch.clamp(t_new, min = 0.0, max = curr_T_max)

#     model.train()

#     return x_new, t_new






