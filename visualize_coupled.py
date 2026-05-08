import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models import CoupledACCHPINN

#function to create model, compute all points (c, phi) needed to generate the animation
def compute_pts_to_visualize(model_checkpoint, L, T_max, device, N_x_plot: int = 200, N_t_plot: int = 100):

    #initialize and load best checkpoint for PINN model
    model = CoupledACCHPINN(hidden_layers=4, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval() 
    
    #creating the spacetime grid
    x_grid = np.linspace(0, L, N_x_plot)
    t_grid = np.linspace(0, T_max, N_t_plot)
    
    #initiating 2D concentration values arrays
    phi_results = np.zeros((N_t_plot, N_x_plot))
    c_results = np.zeros((N_t_plot, N_x_plot))

    #inference
    with torch.no_grad(): # Disattiviamo i gradienti per velocizzare
        for i, t_val in enumerate(t_grid):
            x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)
            t_tensor = torch.ones_like(x_tensor) * t_val #same t repeated for N_x_plot times
            t_tensor = t_tensor.to(device)

            phi_pred, c_pred, _ = model(x_tensor, t_tensor) #prediction

            phi_results[i, :] = phi_pred.cpu().numpy().flatten()            
            c_results[i, :] = c_pred.cpu().numpy().flatten() #saving results

    return phi_results, c_results, x_grid, t_grid


def create_animation(save_path: str, L: float, phi_results: np.array, c_results: np.array, x_grid: np.array, t_grid: np.array):
    N_t_plot = c_results.shape[0]

    fig, (ax_phi, ax_c) = plt.subplots(
        2, 1,
        figsize=(8, 7),
        sharex=True
    )

    # Dynamic y-limits with small margin
    phi_min, phi_max = np.min(phi_results), np.max(phi_results)
    c_min, c_max = np.min(c_results), np.max(c_results)

    phi_margin = 0.1 * max(1e-8, phi_max - phi_min)
    c_margin = 0.1 * max(1e-8, c_max - c_min)

    ax_phi.set_xlim(0, L)
    ax_phi.set_ylim(phi_min - phi_margin, phi_max + phi_margin)
    ax_phi.set_ylabel(r"$\phi(x,t)$")
    ax_phi.set_title("Allen-Cahn field")
    ax_phi.grid(True, linestyle="--", alpha=0.6)

    ax_c.set_xlim(0, L)
    ax_c.set_ylim(c_min - c_margin, c_max + c_margin)
    ax_c.set_xlabel("Spazio (x)")
    ax_c.set_ylabel(r"$c(x,t)$")
    ax_c.set_title("Cahn-Hilliard field")
    ax_c.grid(True, linestyle="--", alpha=0.6)

    # Initial conditions
    ax_phi.plot(
        x_grid,
        phi_results[0, :],
        lw=1,
        linestyle="--",
        color="gray",
        label=r"$\phi(x,0)$"
    )

    ax_c.plot(
        x_grid,
        c_results[0, :],
        lw=1,
        linestyle="--",
        color="gray",
        label=r"$c(x,0)$"
    )

    # Animated lines
    line_phi, = ax_phi.plot(
        [],
        [],
        lw=2,
        label=r"$\phi(x,t)$"
    )

    line_c, = ax_c.plot(
        [],
        [],
        lw=2,
        label=r"$c(x,t)$"
    )

    ax_phi.legend()
    ax_c.legend()

    fig.suptitle("Evoluzione coupled Allen-Cahn + Cahn-Hilliard", fontsize=13)

    def animate(i):
        line_phi.set_data(x_grid, phi_results[i, :])
        line_c.set_data(x_grid, c_results[i, :])

        fig.suptitle(
            f"Evoluzione coupled Allen-Cahn + Cahn-Hilliard | "
            f"t = {t_grid[i]:.3f}",
            fontsize=13
        )

        return line_phi, line_c

    anim = FuncAnimation(
        fig,
        animate,
        frames=N_t_plot,
        interval=50,
        blit=False
    )

    anim.save(save_path, fps=10)
    plt.show()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_checkpoint", type=str, required = True, help = "File name of model checkpoint used")
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--outpath", type=str, required = True, help = "File name of output gif")

    args = parser.parse_args()

    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    #compute all c(x, t) points on the discrete grid (200*100)
    model_checkpoint = "/".join(["artifacts/coupled", args.model_checkpoint])
    phi_results, c_results, x_grid, t_grid = compute_pts_to_visualize(
        model_checkpoint = model_checkpoint,
        L = 1.0, 
        T_max = args.tmax,
        device = device
    )

    #create the time animation of the 1D plot
    save_path = "/".join(["artifacts", args.outpath])
    create_animation(
        save_path = save_path,
        L = 1.0, #L must be the same as in the previous function!
        phi_results = phi_results, 
        c_results = c_results,
        x_grid = x_grid, 
        t_grid = t_grid
    )

