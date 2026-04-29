import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.models import CahnHilliardPINN

#function to create model, compute all points (c, t) needed to generate the animation
def compute_pts_to_visualize(model_checkpoint: str, L: float, T_max: float, N_x_plot: int = 200, N_t_plot: int = 100):

    #initialize and load best checkpoint for PINN model
    model = CahnHilliardPINN(hidden_layers=3, hidden_dim=64)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval() 
    
    #creating the spacetime grid
    x_grid = np.linspace(0, L, N_x_plot)
    t_grid = np.linspace(0, T_max, N_t_plot)
    
    #initiating 2D concentration values array
    c_results = np.zeros((N_t_plot, N_x_plot))

    #inference
    with torch.no_grad(): # Disattiviamo i gradienti per velocizzare
        for i, t_val in enumerate(t_grid):
            x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
            t_tensor = torch.ones_like(x_tensor) * t_val #same t repeated for N_x_plot times
            
            c_pred, _ = model(x_tensor, t_tensor) #prediction
            
            c_results[i, :] = c_pred.numpy().flatten() #saving results

    return c_results, x_grid, t_grid


def create_animation(save_path: str, L: float, c_results: np.array, x_grid: np.array, t_grid: np.array):
    fig, ax = plt.subplots(figsize=(8, 5))

    N_t_plot = c_results.shape[0]
    
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, L)
    ax.set_xlabel('Spazio (x)')
    ax.set_ylabel('Concentrazione (c)')
    ax.set_title('Evoluzione Cahn-Hilliard (PINN)')
    ax.grid(True, linestyle='--', alpha=0.6)

    line, = ax.plot([], [], lw=2, color='blue') #line to update
    
    #initial condition
    ax.plot(x_grid, c_results[0, :], lw=1, color='gray', linestyle='--', label='t=0 (IC)')
    ax.legend()

    #function to animate the curve (scroll through time)
    def animate(i):
        line.set_data(x_grid, c_results[i, :])
        ax.set_title(f'Evoluzione Cahn-Hilliard - Tempo: {t_grid[i]:.2f}')
        return line,

    anim = FuncAnimation(fig, animate, frames=N_t_plot, interval=50, blit=False) #compiling the animation
    anim.save(save_path, fps=10)

    plt.show()

if __name__ == "__main__":
    #compute all c(x, t) points on the discrete grid (200*100)
    c_results, x_grid, t_grid = compute_pts_to_visualize(
        model_checkpoint = "artifacts/ch_pinn_005eps.pt",
        L = 1.0, 
        T_max = 10.0
    )

    #create the time animation of the 1D plot
    create_animation(
        save_path = "artifacts/ch-animation_005eps.gif",
        L = 1.0, #L must be the same as in the previous function!
        c_results = c_results,
        x_grid = x_grid, 
        t_grid = t_grid
    )

