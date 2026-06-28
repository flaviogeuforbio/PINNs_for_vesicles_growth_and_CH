"""
generate_ac1d_data.py
======================================================================
Genera i dati di riferimento ("ground truth") per un benchmark inverso
CIECO basato sull'equazione di Allen-Cahn 1D SENZA termine sorgente.

Perche' questo file e' il cuore del progetto
--------------------------------------------
Nel benchmark "manufactured" (MMS) la sorgente S_phi era costruita
ANALITICAMENTE a partire dal parametro vero m_phi_true. Quindi il valore
da ritrovare era gia' nascosto dentro i dati: il problema NON era cieco.

Qui facciamo l'opposto. Risolviamo numericamente la PDE vera
        phi_t = - m_phi_true * mu(phi)
SENZA sorgente. Il parametro m_phi_true entra SOLO nella generazione dei
dati (regola la velocita' della dinamica), e NON compare in nessun
termine che il problema inverso usera' come bersaglio. Di conseguenza
ritrovare m_phi a partire da phi(x,t) e dalla forma della PDE e' un
problema inverso GENUINAMENTE cieco.

Modello (settore Allen-Cahn / phase-field, campo NON conservato)
----------------------------------------------------------------
    phi_t = - m_phi * mu
    mu    =   k * [ (1/eps) (phi^3 - phi)  -  eps * phi_xx ]

con:
    phi(x,t)  campo di fase, x in [0,1], condizioni di Neumann ai bordi
    mu        potenziale chimico = derivata variazionale dell'energia
    m_phi     mobilita' (IL parametro da inferire)  -> qui = m_phi_true
    k         costante di scala dell'energia (FISSA E NOTA)
    eps       spessore d'interfaccia

Energia dissipata dalla dinamica (flusso di gradiente L^2):
    E[phi] = k * Integrale[ (eps/2) phi_x^2  +  (1/eps) W(phi) ] dx
    W(phi) = (1/4)(phi^2 - 1)^2     (doppio pozzo, minimi in phi = +-1)

Cosa produce lo script
----------------------
1. Risolve la PDE con il "metodo delle linee" + integratore implicito
   per ODE stiff (scipy.solve_ivp, metodo BDF).
2. Esegue TRE validazioni:
   (A) convergenza sotto raffinamento di griglia (verifica lo schema
       spaziale ed il fatto che eps sia risolto);
   (B) monotonia dell'energia nel tempo (verifica la struttura di
       flusso di gradiente: l'energia non puo' aumentare);
   (C) stima least-squares di m_phi sui campi ESATTI (checkpoint 1.2):
       deve restituire ~ m_phi_true. Se passa, il problema inverso e'
       ben posto e cieco, e la matematica e' giusta. Cosi' ogni
       degrado successivo sara' colpa della rete, non del setup.
3. Salva i dati (solo phi come osservabile) in formato .npz e produce
   una figura diagnostica.

NB: questo script genera dati PULITI. Il rumore di osservazione verra'
aggiunto nello script di training/inferenza (nella data loss), non qui.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")  # backend senza finestra grafica (gira in container)
import matplotlib.pyplot as plt


# =====================================================================
# 1. PARAMETRI  (tutti in un solo posto, facili da cambiare)
# =====================================================================

# --- fisica del modello ---
M_PHI_TRUE = 0.5    # mobilita' VERA: il parametro che il problema inverso dovra' ritrovare
K          = 1.0    # costante di scala dell'energia: FISSA e NOTA (risolve la degenerazione m_phi*k)
EPS        = 0.08   # spessore d'interfaccia (regola quanto e' "spessa" la transizione +1/-1)

# --- dominio temporale e discretizzazione ---
T_FINAL = 0.5       # tempo finale della simulazione
N_X     = 160       # numero di nodi spaziali sulla griglia [0,1]
N_SNAP  = 101       # numero di istantanee temporali salvate (incluso t=0)

# --- condizione iniziale:  phi(x,0) = IC_A0 + IC_B0 * cos(pi x) ---
IC_A0 = 0.2
IC_B0 = 0.3

# --- tolleranze dell'integratore ODE (piu' piccole = dati piu' puliti) ---
RTOL = 1e-9
ATOL = 1e-11

# --- output ---
OUT_NPZ = "artifacts/ac1d_blind_data.npz"
OUT_PNG = "artifacts/ac1d_diagnostic.png"


# =====================================================================
# 2. GRIGLIA E OPERATORI DISCRETI
# =====================================================================

def make_grid(n_x):
    """Griglia uniforme di n_x nodi su [0,1]. Restituisce (x, dx)."""
    x = np.linspace(0.0, 1.0, n_x)
    dx = x[1] - x[0]
    return x, dx


def trapezoid_weights(n_x, dx):
    """
    Pesi per integrare nello spazio con la regola del trapezio.
    I nodi di bordo pesano dx/2 perche' rappresentano "mezze celle".
    Questi stessi pesi rendono l'energia discreta esattamente monotona.
    """
    w = np.full(n_x, dx)
    w[0]  *= 0.5
    w[-1] *= 0.5
    return w


def laplacian_matrix(n_x, dx):
    """
    Matrice (n_x x n_x) della derivata seconda d^2/dx^2 con condizioni
    di Neumann (derivata normale nulla ai bordi), schema del 2o ordine.

    Righe interne:  [ ... 1  -2  1 ... ] / dx^2   (differenza centrata)
    Riga di bordo:  [ -2  2  0 ... ]    / dx^2   (nodo fantasma riflettente:
                                                  phi_{-1} = phi_{1})
    La costruiamo UNA volta sola perche' e' costante (non dipende da phi).
    """
    L = np.zeros((n_x, n_x))
    for i in range(1, n_x - 1):
        L[i, i - 1] = 1.0
        L[i, i]     = -2.0
        L[i, i + 1] = 1.0
    # bordo sinistro (Neumann): phi_fantasma = phi[1]
    L[0, 0] = -2.0
    L[0, 1] = 2.0
    # bordo destro (Neumann): phi_fantasma = phi[-2]
    L[-1, -1] = -2.0
    L[-1, -2] = 2.0
    return L / dx**2


def mu_of_phi(phi, L, k=K, eps=EPS):
    """
    Potenziale chimico  mu = k[ (1/eps)(phi^3 - phi) - eps * phi_xx ].
    phi_xx e' calcolato con la matrice del Laplaciano (L @ phi).
    """
    return k * ((1.0 / eps) * (phi**3 - phi) - eps * (L @ phi))


# =====================================================================
# 3. SISTEMA DI ODE (metodo delle linee) E SUO JACOBIANO
# =====================================================================
# Discretizzando solo lo spazio, la PDE diventa un sistema di n_x ODE
# accoppiate:   d phi_i / dt = F_i(phi).  Lo consegniamo a solve_ivp.

def rhs(t, phi, L, m_phi=M_PHI_TRUE):
    """Lato destro del sistema di ODE:  phi_t = - m_phi * mu(phi)."""
    return -m_phi * mu_of_phi(phi, L)


def jac(t, phi, L, m_phi=M_PHI_TRUE, k=K, eps=EPS):
    """
    Jacobiano d(phi_t)/d(phi). Serve all'integratore implicito per
    fare passi grandi e stabili sul sistema stiff.

      d(mu)/d(phi)  = k[ (1/eps)(3 phi^2 - 1) I  -  eps L ]
      phi_t = -m_phi mu   =>   J = -m_phi * d(mu)/d(phi)
    """
    diag = k * (1.0 / eps) * (3.0 * phi**2 - 1.0)   # parte non lineare (diagonale)
    J_mu = np.diag(diag) - k * eps * L              # + parte di diffusione (costante)
    return -m_phi * J_mu


# =====================================================================
# 4. ENERGIA E STIMATORI LEAST-SQUARES (per le validazioni)
# =====================================================================

def energy(phi, dx, w, k=K, eps=EPS):
    """
    Energia discreta E[phi] = k[ (eps/2) Int phi_x^2 + (1/eps) Int W(phi) ].
    - gradiente: calcolato sugli "spigoli" tra nodi (differenze in avanti);
    - doppio pozzo: integrato coi pesi del trapezio (mezze celle ai bordi).
    Questa e' l'energia che la dinamica deve dissipare monotonamente.
    """
    g = (phi[1:] - phi[:-1]) / dx              # gradiente sugli spigoli (n_x-1 valori)
    grad_term = 0.5 * eps * np.sum(g**2) * dx
    W = 0.25 * (phi**2 - 1.0)**2
    well_term = (1.0 / eps) * np.sum(w * W)
    return k * (grad_term + well_term)


def weak_ls_estimate(phi_snaps, t_snaps, L, w):
    """
    Stima INTEGRALE (weak-form) di m_phi a partire da phi(x,t).

    Da  phi_t = -m_phi mu, integrando tra due istanti consecutivi:
        phi(t_{n+1}) - phi(t_n) = -m_phi * Integrale_{t_n}^{t_{n+1}} mu dt
    Ponendo
        A = Integrale mu dt  (regola del trapezio tra i due snapshot)
        B = -(phi(t_{n+1}) - phi(t_n))
    si ha  B ~ m_phi * A  per ogni punto x e ogni coppia di tempi.
    Risolviamo ai minimi quadrati (proiezione L^2 pesata sullo spazio):
        m_phi = sum(w * A * B) / sum(w * A * A)

    Punto chiave: NON usa la derivata temporale puntuale, solo differenze
    tra istantanee e integrali. E' la stima robusta alle imprecisioni di phi_t.
    """
    A_all, B_all, w_all = [], [], []
    for n in range(len(t_snaps) - 1):
        dt = t_snaps[n + 1] - t_snaps[n]
        mu_n   = mu_of_phi(phi_snaps[n],     L)
        mu_np1 = mu_of_phi(phi_snaps[n + 1], L)
        A = 0.5 * dt * (mu_n + mu_np1)            # Int mu dt  (trapezio in tempo)
        B = -(phi_snaps[n + 1] - phi_snaps[n])
        A_all.append(A); B_all.append(B); w_all.append(w)
    A = np.concatenate(A_all); B = np.concatenate(B_all); W = np.concatenate(w_all)
    return float(np.sum(W * A * B) / np.sum(W * A * A))


def strong_ls_estimate(phi_snaps, t_snaps, L, w):
    """
    Stima FORTE (strong-form) di m_phi, per confronto.

    Approssima phi_t con differenze CENTRATE nel tempo, poi:
        phi_t = -m_phi mu  =>  -phi_t ~ m_phi * mu
        m_phi = sum(w * mu * (-phi_t)) / sum(w * mu * mu)
    Sui dati PULITI del solver anche questa funziona (le differenze
    centrate sono accurate). Sui campi della RETE, invece, phi_t e'
    impreciso: e' li' che strong e weak divergeranno.
    """
    A_all, B_all, w_all = [], [], []
    for n in range(1, len(t_snaps) - 1):
        dt_c  = t_snaps[n + 1] - t_snaps[n - 1]
        phi_t = (phi_snaps[n + 1] - phi_snaps[n - 1]) / dt_c   # differenza centrata
        mu_n  = mu_of_phi(phi_snaps[n], L)
        A_all.append(mu_n); B_all.append(-phi_t); w_all.append(w)
    A = np.concatenate(A_all); B = np.concatenate(B_all); W = np.concatenate(w_all)
    return float(np.sum(W * A * B) / np.sum(W * A * A))


# =====================================================================
# 5. SIMULAZIONE
# =====================================================================

def run_once(n_x, t_final=T_FINAL, n_snap=N_SNAP):
    """Risolve la PDE su una griglia di n_x nodi. Restituisce (x, t, phi, dx, L, w)."""
    x, dx = make_grid(n_x)
    w = trapezoid_weights(n_x, dx)
    L = laplacian_matrix(n_x, dx)
    phi0 = IC_A0 + IC_B0 * np.cos(np.pi * x)        # condizione iniziale
    t_eval = np.linspace(0.0, t_final, n_snap)      # istanti in cui salvare
    sol = solve_ivp(
        fun=rhs,                 # lato destro del sistema di ODE
        t_span=(0.0, t_final),
        y0=phi0,
        method="BDF",            # integratore implicito per sistemi stiff
        t_eval=t_eval,
        jac=jac,                 # Jacobiano analitico (passi grandi e stabili)
        args=(L,),               # argomento extra passato a rhs e jac
        rtol=RTOL, atol=ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp ha fallito a n_x={n_x}: {sol.message}")
    phi_snaps = sol.y.T          # forma [n_snap, n_x]
    return x, t_eval, phi_snaps, dx, L, w


def l2_rel(a, b):
    """Errore relativo L2 tra due vettori (per la convergenza di griglia)."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =====================================================================
# 6. MAIN: simula, valida, salva, traccia
# =====================================================================

def main():
    print("=" * 64)
    print(" GENERAZIONE DATI - Allen-Cahn 1D source-free (benchmark cieco)")
    print("=" * 64)
    print(f" m_phi_true = {M_PHI_TRUE} | k = {K} | eps = {EPS}")
    print(f" T = {T_FINAL} | N_x = {N_X} | snapshot = {N_SNAP}")
    print("-" * 64)

    # --- simulazione principale alla risoluzione scelta ---
    x, t, phi, dx, L, w = run_once(N_X)
    print(f" Simulazione completata: phi ha forma {phi.shape} (tempo x spazio)")
    print(f" phi(t=0):   min={phi[0].min():+.3f}  max={phi[0].max():+.3f}")
    print(f" phi(t=T):   min={phi[-1].min():+.3f}  max={phi[-1].max():+.3f}")

    # --- VALIDAZIONE (A): convergenza di griglia (ordine ~2) ---
    print("-" * 64)
    print(" (A) Convergenza sotto raffinamento di griglia")
    N0 = N_X
    _, _, phi_c, *_ = run_once(N0)            # griglia "coarse"
    _, _, phi_m, *_ = run_once(2 * N0 - 1)    # griglia "media" (nidificata)
    _, _, phi_f, *_ = run_once(4 * N0 - 3)    # griglia "fine"   (riferimento)
    fc, fm, ff = phi_c[-1], phi_m[-1], phi_f[-1]   # campi al tempo finale
    err_coarse = l2_rel(fc, ff[::4])   # coarse vs fine (sottocampionata sui nodi coarse)
    err_medium = l2_rel(fm, ff[::2])   # media  vs fine (sottocampionata sui nodi medi)
    ratio = err_coarse / err_medium if err_medium > 0 else np.inf
    print(f"     errore L2 (coarse  N={N0})  vs fine : {err_coarse:.3e}")
    print(f"     errore L2 (media   N={2*N0-1}) vs fine : {err_medium:.3e}")
    print(f"     rapporto errori (atteso ~4 per 2o ordine): {ratio:.2f}")
    conv_ok = err_coarse < 1e-3
    print(f"     -> risoluzione adeguata: {conv_ok}  (soglia 1e-3)")

    # --- VALIDAZIONE (B): monotonia dell'energia ---
    print("-" * 64)
    print(" (B) Monotonia dell'energia (flusso di gradiente)")
    E = np.array([energy(phi[n], dx, w) for n in range(len(t))])
    dE = np.diff(E)
    max_increase = float(dE.max())
    energy_ok = max_increase <= 1e-8
    print(f"     E(0) = {E[0]:.6f}  ->  E(T) = {E[-1]:.6f}  (deve calare)")
    print(f"     massimo incremento tra snapshot: {max_increase:.2e}")
    print(f"     -> energia monotona non crescente: {energy_ok}  (tolleranza 1e-8)")

    # --- VALIDAZIONE (C): recupero di m_phi sui campi ESATTI (checkpoint 1.2) ---
    print("-" * 64)
    print(" (C) Recupero di m_phi sui campi esatti del solver  [CHECKPOINT 1.2]")
    m_weak   = weak_ls_estimate(phi, t, L, w)
    m_strong = strong_ls_estimate(phi, t, L, w)
    err_weak   = abs(m_weak   - M_PHI_TRUE) / M_PHI_TRUE * 100
    err_strong = abs(m_strong - M_PHI_TRUE) / M_PHI_TRUE * 100
    print(f"     m_phi_true                 = {M_PHI_TRUE:.6f}")
    print(f"     stima WEAK   (integrale)   = {m_weak:.6f}   (errore {err_weak:.3f}%)")
    print(f"     stima STRONG (derivata)    = {m_strong:.6f}   (errore {err_strong:.3f}%)")
    recover_ok = (err_weak < 1.0) and (err_strong < 1.0)
    print(f"     -> problema ben posto e cieco: {recover_ok}  (soglia 1% su entrambe)")

    # --- riepilogo ---
    print("=" * 64)
    all_ok = conv_ok and energy_ok and recover_ok
    print(f" ESITO VALIDAZIONI:  convergenza={conv_ok}  energia={energy_ok}  "
          f"recupero={recover_ok}")
    print(f" >>> TUTTO OK: {all_ok}")
    print("=" * 64)

    # --- salvataggio (SOLO phi come osservabile: il benchmark e' cieco) ---
    np.savez(
        OUT_NPZ,
        x=x, t=t, phi=phi,            # dati osservabili + griglie
        m_phi_true=M_PHI_TRUE,        # "risposta" (solo per il confronto finale, NON per il training)
        k=K, eps=EPS, dx=dx,          # costanti note del modello
        ic_a0=IC_A0, ic_b0=IC_B0,
        energy=E,
    )
    print(f" Dati salvati in: {OUT_NPZ}")

    # --- figura diagnostica ---
    make_figure(x, t, phi, E)
    print(f" Figura salvata in: {OUT_PNG}")

    return all_ok


def make_figure(x, t, phi, E):
    """Tre pannelli: profili phi(x) a vari tempi, mappa spazio-tempo, energia(t)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) profili a tempi selezionati
    idx = [0, len(t)//4, len(t)//2, 3*len(t)//4, len(t)-1]
    for i in idx:
        ax[0].plot(x, phi[i], label=f"t={t[i]:.2f}")
    ax[0].set_title("phi(x, t) profile")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("phi")
    ax[0].axhline(1.0, ls=":", c="gray", lw=0.8)
    ax[0].axhline(-1.0, ls=":", c="gray", lw=0.8)
    ax[0].legend(fontsize=8)

    # (b) mappa spazio-tempo
    im = ax[1].imshow(phi, aspect="auto", origin="lower",
                      extent=[x[0], x[-1], t[0], t[-1]], cmap="coolwarm",
                      vmin=-1, vmax=1)
    ax[1].set_title("phi(x,t)")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("t")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    # (c) energia
    ax[2].plot(t, E, "-o", ms=3)
    ax[2].set_title("Energy  E(t)")
    ax[2].set_xlabel("t"); ax[2].set_ylabel("E")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
