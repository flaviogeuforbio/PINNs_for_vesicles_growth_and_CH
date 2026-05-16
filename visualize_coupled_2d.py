import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils_coupled_2d import load_model

#function to parse data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required = False, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    parser.add_argument("--checkpoint_name", type=str, required=True, help = "Model checkpoint name")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    # parser.add_argument("--segment_length", type=float, default=0.5, help = "Time segment dimension of each network in the ensemble (time windowing)")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
    parser.add_argument("--gif_name", type=str, required = True, help = "File name of output gif")
    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of PINN hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons for each PINN hidden layers")

    args = parser.parse_args()

    return args

#function to create model, compute all points (c, phi) needed to generate the animation
def compute_pts_to_visualize(
    model_checkpoint, 
    L_x, L_y, 
    T_max, 
    hidden_layers, 
    hidden_dim, 
    device, 
    N_x_plot: int = 128, 
    N_y_plot: int = 128, 
    N_t_plot: int = 80
):
    #loading the model  
    model = load_model(
        model_checkpoint, 
        hidden_layers, 
        hidden_dim
    )
    model.eval()

    #creating the grid 
    x_grid = torch.linspace(0.0, L_x, N_x_plot)
    y_grid = torch.linspace(0.0, L_y, N_y_plot)
    t_grid = torch.linspace(0.0, T_max, N_t_plot)

    X, Y = torch.meshgrid(x_grid, y_grid, indexing = "xy")

    #flatten the grid to pass it to the network
    x_flat = X.reshape(-1, 1).to(device)
    y_flat = Y.reshape(-1, 1).to(device)

    #instantiate arrays with fields values on the discrete grid
    phi_results = np.zeros((N_t_plot, N_y_plot, N_x_plot))
    c_results = np.zeros((N_t_plot, N_y_plot, N_x_plot))

    with torch.no_grad():
        for i, t_val in enumerate(t_grid):
            t_tensor = torch.full_like(x_flat, float(t_val), device = device)

            #calculate model predictions for t = t_val
            phi, c, _ = model(x_flat, y_flat, t_tensor)

            #update results array
            phi_results[i, :, :] = phi.detach().cpu().numpy().reshape(N_y_plot, N_x_plot)
            c_results[i, :, :] = c.detach().cpu().numpy().reshape(N_y_plot, N_x_plot)

    return phi_results, c_results, x_grid, y_grid, t_grid

#function to create the 2d animation gif
def create_animation_2d(
    save_path,
    Lx,
    Ly,
    phi_results,
    c_results,
    x_grid,
    y_grid,
    t_grid,
):
    N_t_plot = phi_results.shape[0]

    fig, (ax_phi, ax_c) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    phi_min = np.min(phi_results)
    phi_max = np.max(phi_results)
    c_min = np.min(c_results)
    c_max = np.max(c_results)

    extent = [0.0, Lx, 0.0, Ly]

    im_phi = ax_phi.imshow(
        phi_results[0],
        origin="lower",
        extent=extent,
        vmin=phi_min,
        vmax=phi_max,
        aspect="equal",
        interpolation="bilinear",
    )

    im_c = ax_c.imshow(
        c_results[0],
        origin="lower",
        extent=extent,
        vmin=c_min,
        vmax=c_max,
        aspect="equal",
        interpolation="bilinear",
    )

    ax_phi.set_title(r"Allen-Cahn field $\phi(x,y,t)$")
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel("y")

    ax_c.set_title(r"Cahn-Hilliard field $c(x,y,t)$")
    ax_c.set_xlabel("x")
    ax_c.set_ylabel("y")

    cbar_phi = fig.colorbar(im_phi, ax=ax_phi, fraction=0.046, pad=0.04)
    cbar_phi.set_label(r"$\phi$")

    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label(r"$c$")

    fig.suptitle(
        f"Coupled Allen-Cahn + Cahn-Hilliard 2D | t = {t_grid[0]:.3f}",
        fontsize=14,
    )

    def animate(i):
        im_phi.set_data(phi_results[i])
        im_c.set_data(c_results[i])

        fig.suptitle(
            f"Coupled Allen-Cahn + Cahn-Hilliard 2D | t = {t_grid[i]:.3f}",
            fontsize=14,
        )

        return im_phi, im_c

    anim = FuncAnimation(
        fig,
        animate,
        frames=N_t_plot,
        interval=80,
        blit=False,
    )

    anim.save(save_path, fps=10)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path 

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #model weights path
    checkpoint_path = Path("artifacts/baseline/weights") / args.checkpoint_name

    phi_results, c_results, x_grid, y_grid, t_grid = compute_pts_to_visualize(
        model_checkpoint=checkpoint_path,
        L_x=args.lx,
        L_y=args.ly,
        T_max=args.tmax,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        device=device
    )

    save_dir = Path("artifacts/baseline/animations")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / args.gif_name

    create_animation_2d(
        save_path=save_path,
        Lx=args.lx,
        Ly=args.ly,
        phi_results=phi_results,
        c_results=c_results,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
    )

    print(f"Animation saved to: {save_path}")