"""
generate_ac1d_data.py

Generate source-free 1D Allen-Cahn data for a blind inverse benchmark.

The generated dataset is used to infer m_phi from observations of phi(x,t).
No manufactured source term is used. The true mobility m_phi_true is used
only to generate the numerical trajectory and is saved only for final
evaluation/comparison.

Model:
    phi_t = - m_phi * mu

    mu = k * [ (1/eps) * (phi^3 - phi) - eps * phi_xx ]

Boundary conditions:
    phi_x(0,t) = phi_x(1,t) = 0

Initial condition:
    phi(x,0) = ic_a0 + ic_b0 * cos(pi*x)
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PARAMETERS
# =============================================================================

# physical parameters
M_PHI_TRUE = 0.5
K = 1.0
EPS = 0.08

# numerical setup
T_FINAL = 0.5
N_X = 160
N_SNAP = 101

# initial condition: phi(x,0) = IC_A0 + IC_B0*cos(pi*x)
IC_A0 = 0.2
IC_B0 = 0.3

# BDF solver tolerances
RTOL = 1e-9
ATOL = 1e-11

# output files
OUT_NPZ = "artifacts/ac1d_blind_data.npz"
OUT_PNG = "artifacts/ac1d_diagnostic.png"


# =============================================================================
# GRID AND DISCRETE OPERATORS
# =============================================================================

def make_grid(n_x):
    # uniform grid on [0,1]
    x = np.linspace(0.0, 1.0, n_x)
    dx = x[1] - x[0]

    return x, dx


def trapezoid_weights(n_x, dx):
    # trapezoidal weights for spatial integration
    w = np.full(n_x, dx)
    w[0] *= 0.5
    w[-1] *= 0.5

    return w


def laplacian_matrix(n_x, dx):
    """
    Build the 1D second-derivative matrix with homogeneous Neumann BCs.

    Interior nodes:
        phi_xx[i] = (phi[i-1] - 2*phi[i] + phi[i+1]) / dx^2

    Boundary nodes use mirrored ghost points:
        phi[-1] = phi[1]
        phi[n]  = phi[n-2]
    """

    L = np.zeros((n_x, n_x))

    # interior rows
    for i in range(1, n_x - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0

    # left boundary: ghost reflection
    L[0, 0] = -2.0
    L[0, 1] = 2.0

    # right boundary: ghost reflection
    L[-1, -1] = -2.0
    L[-1, -2] = 2.0

    return L / dx**2


def mu_of_phi(phi, L, k=K, eps=EPS):
    # chemical potential for the Allen-Cahn gradient flow
    phi_xx = L @ phi

    mu = k * (
        (1.0 / eps) * (phi**3 - phi)
        - eps * phi_xx
    )

    return mu


# =============================================================================
# METHOD OF LINES: ODE SYSTEM AND JACOBIAN
# =============================================================================

def rhs(t, phi, L, m_phi=M_PHI_TRUE):
    # right-hand side of the semi-discrete ODE system
    return -m_phi * mu_of_phi(phi, L)


def jac(t, phi, L, m_phi=M_PHI_TRUE, k=K, eps=EPS):
    """
    Jacobian of the ODE right-hand side.

    Since:
        phi_t = -m_phi * mu(phi)

    and:
        dmu/dphi = k * [ (1/eps)*(3*phi^2 - 1)*I - eps*L ]

    then:
        d(phi_t)/dphi = -m_phi * dmu/dphi
    """

    diag = k * (1.0 / eps) * (3.0 * phi**2 - 1.0)
    J_mu = np.diag(diag) - k * eps * L

    return -m_phi * J_mu


# =============================================================================
# ENERGY AND LEAST-SQUARES VALIDATION
# =============================================================================

def energy(phi, dx, w, k=K, eps=EPS):
    """
    Discrete Allen-Cahn energy:

        E = k * int [ eps/2 * phi_x^2 + (1/eps)*W(phi) ] dx

    with:
        W(phi) = 1/4 * (phi^2 - 1)^2
    """

    phi_x_edges = (phi[1:] - phi[:-1]) / dx
    grad_term = 0.5 * eps * np.sum(phi_x_edges**2) * dx

    W = 0.25 * (phi**2 - 1.0)**2
    well_term = (1.0 / eps) * np.sum(w * W)

    return k * (grad_term + well_term)


def weak_ls_estimate(phi_snaps, t_snaps, L, w):
    """
    Weak-form least-squares estimate of m_phi from clean solver data.

    From:
        phi_t = -m_phi * mu

    integrate between two consecutive snapshots:
        phi(t_b) - phi(t_a) = -m_phi * int_a^b mu dt

    Therefore:
        B = -(phi(t_b) - phi(t_a))
        A = int_a^b mu dt

    and:
        m_phi = sum(w*A*B) / sum(w*A*A)
    """

    A_all = []
    B_all = []
    w_all = []

    for n in range(len(t_snaps) - 1):
        dt = t_snaps[n + 1] - t_snaps[n]

        mu_n = mu_of_phi(phi_snaps[n], L)
        mu_np1 = mu_of_phi(phi_snaps[n + 1], L)

        A = 0.5 * dt * (mu_n + mu_np1)
        B = -(phi_snaps[n + 1] - phi_snaps[n])

        A_all.append(A)
        B_all.append(B)
        w_all.append(w)

    A = np.concatenate(A_all)
    B = np.concatenate(B_all)
    W = np.concatenate(w_all)

    return float(np.sum(W * A * B) / np.sum(W * A * A))


def strong_ls_estimate(phi_snaps, t_snaps, L, w):
    """
    Strong-form least-squares estimate of m_phi from clean solver data.

    From:
        -phi_t = m_phi * mu

    Therefore:
        m_phi = sum(w*mu*(-phi_t)) / sum(w*mu^2)
    """

    A_all = []
    B_all = []
    w_all = []

    for n in range(1, len(t_snaps) - 1):
        dt_centered = t_snaps[n + 1] - t_snaps[n - 1]

        phi_t = (phi_snaps[n + 1] - phi_snaps[n - 1]) / dt_centered
        mu_n = mu_of_phi(phi_snaps[n], L)

        A_all.append(mu_n)
        B_all.append(-phi_t)
        w_all.append(w)

    A = np.concatenate(A_all)
    B = np.concatenate(B_all)
    W = np.concatenate(w_all)

    return float(np.sum(W * A * B) / np.sum(W * A * A))


# =============================================================================
# SOLVER
# =============================================================================

def run_once(n_x, t_final=T_FINAL, n_snap=N_SNAP):
    # solve the semi-discrete Allen-Cahn system on a grid with n_x nodes

    x, dx = make_grid(n_x)
    w = trapezoid_weights(n_x, dx)
    L = laplacian_matrix(n_x, dx)

    phi0 = IC_A0 + IC_B0 * np.cos(np.pi * x)
    t_eval = np.linspace(0.0, t_final, n_snap)

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, t_final),
        y0=phi0,
        method="BDF",
        t_eval=t_eval,
        jac=jac,
        args=(L,),
        rtol=RTOL,
        atol=ATOL,
    )

    if not sol.success:
        raise RuntimeError(
            f"solve_ivp failed for n_x={n_x}: {sol.message}"
        )

    phi_snaps = sol.y.T

    return x, t_eval, phi_snaps, dx, L, w


def l2_rel(a, b):
    # relative L2 error between two arrays
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =============================================================================
# PLOTTING
# =============================================================================

def make_figure(x, t, phi, E):
    # create diagnostic figure: profiles, space-time map, energy

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # selected profiles
    idx = [
        0,
        len(t) // 4,
        len(t) // 2,
        3 * len(t) // 4,
        len(t) - 1,
    ]

    for i in idx:
        ax[0].plot(x, phi[i], label=f"t={t[i]:.2f}")

    ax[0].set_title("phi(x,t) profiles")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("phi")
    ax[0].axhline(1.0, ls=":", c="gray", lw=0.8)
    ax[0].axhline(-1.0, ls=":", c="gray", lw=0.8)
    ax[0].legend(fontsize=8)

    # space-time map
    im = ax[1].imshow(
        phi,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax[1].set_title("phi(x,t)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("t")

    fig.colorbar(im, ax=ax[1], fraction=0.046)

    # energy decay
    ax[2].plot(t, E, "-o", ms=3)
    ax[2].set_title("Energy E(t)")
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("E")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("AC1D SOURCE-FREE DATA GENERATION")
    print("=" * 80)
    print(f"m_phi_true = {M_PHI_TRUE}")
    print(f"k          = {K}")
    print(f"eps        = {EPS}")
    print(f"T_final    = {T_FINAL}")
    print(f"N_x        = {N_X}")
    print(f"N_snap     = {N_SNAP}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # main simulation
    # -------------------------------------------------------------------------

    x, t, phi, dx, L, w = run_once(N_X)

    print("Main simulation completed.")
    print(f"phi shape  : {phi.shape}")
    print(f"phi(t=0)  : min={phi[0].min():+.6f} | max={phi[0].max():+.6f}")
    print(f"phi(t=T)  : min={phi[-1].min():+.6f} | max={phi[-1].max():+.6f}")

    # -------------------------------------------------------------------------
    # validation A: grid refinement
    # -------------------------------------------------------------------------

    print("-" * 80)
    print("(A) Grid refinement check")

    n0 = N_X

    _, _, phi_c, *_ = run_once(n0)
    _, _, phi_m, *_ = run_once(2 * n0 - 1)
    _, _, phi_f, *_ = run_once(4 * n0 - 3)

    phi_c_T = phi_c[-1]
    phi_m_T = phi_m[-1]
    phi_f_T = phi_f[-1]

    err_coarse = l2_rel(phi_c_T, phi_f_T[::4])
    err_medium = l2_rel(phi_m_T, phi_f_T[::2])

    ratio = err_coarse / err_medium if err_medium > 0 else np.inf
    conv_ok = err_coarse < 1e-3

    print(f"relative L2 error | coarse vs fine : {err_coarse:.6e}")
    print(f"relative L2 error | medium vs fine : {err_medium:.6e}")
    print(f"error ratio       | expected ~4    : {ratio:.3f}")
    print(f"grid check passed                  : {conv_ok}")

    # -------------------------------------------------------------------------
    # validation B: energy monotonicity
    # -------------------------------------------------------------------------

    print("-" * 80)
    print("(B) Energy monotonicity check")

    E = np.array([energy(phi[n], dx, w) for n in range(len(t))])
    dE = np.diff(E)

    max_increase = float(dE.max())
    energy_ok = max_increase <= 1e-8

    print(f"E(0)                         : {E[0]:.8e}")
    print(f"E(T)                         : {E[-1]:.8e}")
    print(f"max energy increment          : {max_increase:.8e}")
    print(f"energy monotonicity passed    : {energy_ok}")

    # -------------------------------------------------------------------------
    # validation C: direct m_phi recovery on clean solver data
    # -------------------------------------------------------------------------

    print("-" * 80)
    print("(C) Least-squares recovery check")

    m_weak = weak_ls_estimate(phi, t, L, w)
    m_strong = strong_ls_estimate(phi, t, L, w)

    err_weak = abs(m_weak - M_PHI_TRUE) / abs(M_PHI_TRUE)
    err_strong = abs(m_strong - M_PHI_TRUE) / abs(M_PHI_TRUE)

    recover_ok = (err_weak < 1e-2) and (err_strong < 1e-2)

    print(f"m_phi_true       : {M_PHI_TRUE:.8e}")
    print(f"m_phi_weak_ls    : {m_weak:.8e} | rel.err={err_weak:.8e}")
    print(f"m_phi_strong_ls  : {m_strong:.8e} | rel.err={err_strong:.8e}")
    print(f"recovery passed  : {recover_ok}")

    # -------------------------------------------------------------------------
    # final summary
    # -------------------------------------------------------------------------

    print("=" * 80)

    all_ok = conv_ok and energy_ok and recover_ok

    print("VALIDATION SUMMARY")
    print(f"grid refinement : {conv_ok}")
    print(f"energy decay    : {energy_ok}")
    print(f"LS recovery     : {recover_ok}")
    print(f"all checks      : {all_ok}")

    print("=" * 80)

    # -------------------------------------------------------------------------
    # save data and diagnostic figure
    # -------------------------------------------------------------------------

    Path(OUT_NPZ).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUT_NPZ,
        x=x,
        t=t,
        phi=phi,
        m_phi_true=M_PHI_TRUE,
        k=K,
        eps=EPS,
        dx=dx,
        ic_a0=IC_A0,
        ic_b0=IC_B0,
        energy=E,
    )

    make_figure(x, t, phi, E)

    print(f"Saved data       : {OUT_NPZ}")
    print(f"Saved diagnostic : {OUT_PNG}")

    return all_ok


if __name__ == "__main__":
    main()