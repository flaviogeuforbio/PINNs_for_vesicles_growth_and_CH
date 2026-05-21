import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils_bio_2d import load_models, predict_windowed

#function to parse data from CLI
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required = True, help = "Name of the current run (specify parameters/hyperparameters, e.g. gamma005_tmax1_epochs2000)")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--segment_length", type=float, default=0.5, help = "Time segment dimension of each network in the ensemble (time windowing)")
    parser.add_argument("--lx", type=float, default=1.0, help = "Box length on x direction")
    parser.add_argument("--ly", type=float, default=1.0, help = "Box length on y direction")
    parser.add_argument("--gif_name", type=str, default="bio_ACCH_2d.gif", help = "File name of output gif")
    parser.add_argument("--hidden_layers", type=int, default=4, help = "N. of PINN hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "N. of neurons for each PINN hidden layers")
    parser.add_argument("--n_x_plot", type=int, default=128, help = "N. of x points in discrete grid used to compute data to visualize")
    parser.add_argument("--n_y_plot", type=int, default=128, help = "N. of y points in discrete grid used to compute data to visualize")
    parser.add_argument("--n_t_plot", type=int, default=80, help = "N. of t point in discrete grid used to compute data to visualize")

    args = parser.parse_args()

    return args

#function to compute all points (phi, psi) needed to generate the animation in time-windowing framework (ensemble of networks)
def compute_pts_to_visualize_tw(
        check_dir, 
        L_x, L_y, 
        T_max, 
        segment_length,
        hidden_layers,
        hidden_dim, 
        device, 
        N_x_plot: int, 
        N_y_plot: int, 
        N_t_plot: int
):
    #load segment models 
    models = load_models(check_dir, hidden_layers, hidden_dim, device)

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
    psi_results = np.zeros((N_t_plot, N_y_plot, N_x_plot))

    #inference
    with torch.no_grad(): # Disattiviamo i gradienti per velocizzare
        for i, t_val in enumerate(t_grid):

            phi_pred, _, psi_pred, _ = predict_windowed(models, x_flat, y_flat, float(t_val), segment_length, device)

            phi_results[i, :, :] = phi_pred.detach().cpu().numpy().reshape(N_y_plot, N_x_plot)            
            psi_results[i, :, :] =psi_pred.detach().cpu().numpy().reshape(N_y_plot, N_x_plot) #saving results

    return phi_results, psi_results, x_grid, y_grid, t_grid

#function to create the 2d animation gif
def create_animation_2d(
    save_path,
    Lx,
    Ly,
    phi_results,
    psi_results,
    x_grid,
    y_grid,
    t_grid,
):
    N_t_plot = phi_results.shape[0]

    fig, (ax_phi, ax_psi) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    phi_min = np.min(phi_results)
    phi_max = np.max(phi_results)
    psi_min = np.min(psi_results)
    psi_max = np.max(psi_results)

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

    im_psi = ax_psi.imshow(
        psi_results[0],
        origin="lower",
        extent=extent,
        vmin=psi_min,
        vmax=psi_max,
        aspect="equal",
        interpolation="bilinear",
    )

    ax_phi.set_title(r"Allen-Cahn field $\phi(x,y,t)$")
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel("y")

    ax_psi.set_title(r"Ionic concentration field $\psi(x,y,t)$")
    ax_psi.set_xlabel("x")
    ax_psi.set_ylabel("y")

    cbar_phi = fig.colorbar(im_phi, ax=ax_phi, fraction=0.046, pad=0.04)
    cbar_phi.set_label(r"$\phi$")

    cbar_psi = fig.colorbar(im_psi, ax=ax_psi, fraction=0.046, pad=0.04)
    cbar_psi.set_label(r"$\psi$")

    fig.suptitle(
        f"Coupled Allen-Cahn + Cahn-Hilliard 2D | t = {t_grid[0]:.3f}",
        fontsize=14,
    )

    def animate(i):
        im_phi.set_data(phi_results[i])
        im_psi.set_data(psi_results[i])

        fig.suptitle(
            f"Coupled Allen-Cahn + Cahn-Hilliard 2D | t = {t_grid[i]:.3f}",
            fontsize=14,
        )

        return im_phi, im_psi

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

    #compute all phi, psi points on the 2d discrete grid (for each time-step)
    check_dir = Path("artifacts/bio-minimal") / args.run_name / "weights"

    if not check_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")

    phi_results, psi_results, x_grid, y_grid, t_grid = compute_pts_to_visualize_tw(
        check_dir = check_dir,
        L_x = args.lx, L_y = args.ly, 
        T_max = args.tmax,
        segment_length = args.segment_length,
        hidden_layers = args.hidden_layers, 
        hidden_dim = args.hidden_dim, 
        device = device,
        N_x_plot = args.n_x_plot,
        N_y_plot = args.n_y_plot, 
        N_t_plot = args.n_t_plot
    )

    save_dir = Path("artifacts/bio-minimal") / args.run_name / "animations"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / args.gif_name

    create_animation_2d(
        save_path=save_path,
        Lx=args.lx,
        Ly=args.ly,
        phi_results=phi_results,
        psi_results=psi_results,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
    )

    print(f"Animation saved to: {save_path}")