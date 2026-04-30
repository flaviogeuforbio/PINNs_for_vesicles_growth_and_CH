import torch

#function that generates a dict of all collocation points (pde, bc, ic) randomly generated to train the network
def generate_coll_points_and_ic(N_pde, N_bc, N_ic, L, T_max):
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
    # noise = torch.randn_like(x_ic) * 0.1 #create gaussian noise with variance sigma = 0.1
    # c_ic_true = torch.zeros_like(x_ic) + noise #this way we are creating thermodynamically unstable domains (a thermal fluctuation)

    c_ic_true = 0.3 * torch.cos(torch.pi * x_ic)
    #this function satisfies the Neumann conditions 

    return collocation, c_ic_true


#function to compute the estimated (inverse) weight of a specific loss term (gradient norm)
def get_gradient_norm(loss, model):

    #calculate gradients with respect to model parameters and retain graph (it will be used in the training backward pass)
    grads = torch.autograd.grad(
        outputs = loss, 
        inputs = model.parameters(),
        grad_outputs = torch.ones_like(loss),
        retain_graph = True, 
        create_graph = False, 
        allow_unused = True #if some parameters are not used in the model
    )

    norm_sq = 0.0 #accumulating squared gradients norm
    for g in grads:
        if g is not None:
            norm_sq += torch.sum(g ** 2)

    return norm_sq.item()





