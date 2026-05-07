import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models import AllenCahnPINN

#function to create model, compute all points (c, t) needed to generate the animation
def compute_pts_to_visualize(model_checkpoint, L, T_max, device, N_x_plot: int = 200, N_t_plot: int = 100):

    #initialize and load best checkpoint for PINN model
    model = AllenCahnPINN(hidden_layers=4, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval() 
    
    #creating the spacetime grid
    x_grid = np.linspace(0, L, N_x_plot)
    t_grid = np.linspace(0, T_max, N_t_plot)
    
    #initiating 2D concentration values array
    phi_results = np.zeros((N_t_plot, N_x_plot))

    #inference
    with torch.no_grad(): # Disattiviamo i gradienti per velocizzare
        for i, t_val in enumerate(t_grid):
            x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1).to(device)
            t_tensor = torch.ones_like(x_tensor) * t_val #same t repeated for N_x_plot times
            t_tensor = t_tensor.to(device)

            phi_pred = model(x_tensor, t_tensor) #prediction
            
            phi_results[i, :] = phi_pred.cpu().numpy().flatten() #saving results

    return phi_results, x_grid, t_grid


def create_animation(save_path: str, L: float, phi_results: np.array, x_grid: np.array, t_grid: np.array):
    fig, ax = plt.subplots(figsize=(8, 5))

    N_t_plot = phi_results.shape[0]
    
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, L)
    ax.set_xlabel('Spazio (x)')
    ax.set_ylabel('Campo (phi)')
    ax.set_title('Evoluzione Allen-Cahn (PINN)')
    ax.grid(True, linestyle='--', alpha=0.6)

    line, = ax.plot([], [], lw=2, color='blue') #line to update
    
    #initial condition
    ax.plot(x_grid, phi_results[0, :], lw=1, color='gray', linestyle='--', label='t=0 (IC)')
    ax.legend()

    #function to animate the curve (scroll through time)
    def animate(i):
        line.set_data(x_grid, phi_results[i, :])
        ax.set_title(f'Evoluzione Allen-Cahn - Tempo: {t_grid[i]:.2f}')
        return line,

    anim = FuncAnimation(fig, animate, frames=N_t_plot, interval=50, blit=False) #compiling the animation
    anim.save(save_path, fps=10)

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_checkpoint", type=str, required = True)
    parser.add_argument("--tmax", type=float, default=1.0, help = "End time of the simulation")
    parser.add_argument("--outpath", type=str, required = True)

    args = parser.parse_args()

    #setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)  

    #compute all phi(x, t) points on the discrete grid (200*100)
    model_checkpoint = "/".join(["artifacts", args.model_checkpoint])
    phi_results, x_grid, t_grid = compute_pts_to_visualize(
        model_checkpoint = "artifacts/ac_baseline_lbfgs_tmax1_asymmIC.pt",
        L = 1.0, 
        T_max = args.tmax,
        device = device
    )

    #create the time animation of the 1D plot
    save_path = "/".join(["artifacts", args.outpath])
    create_animation(
        save_path = "artifacts/ac-animation_baseline_lbfgs_tmax1_asymmIC.gif",
        L = 1.0, #L must be the same as in the previous function!
        phi_results = phi_results,
        x_grid = x_grid, 
        t_grid = t_grid
    )

