import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
from tqdm import tqdm
from utils.visualization import find_figsize

# Parameters for Morse
D = 10.0
a = 1.0
x0 = 0.0

# Grid
N = 1000
x = np.linspace(-1, 2, N)
dx = x[1] - x[0]

def morse_potential(x, D, a, x0):
    """
    Morse potential function.
    """
    return D * (1 - np.exp(-a * (x - x0)))**2 - D

V_matrix = np.diag(morse_potential(x, D, a, x0))
T_dvr = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            T_dvr[i, j] = np.pi**2 / (6 * dx**2)
        else:
            T_dvr[i, j] = ((-1)**(i - j)) / (dx**2 * (i - j)**2)
T_true = np.zeros((N, N))
T_diag = 1 / dx**2 
T_off_diag = -1 / (2 * dx**2)
for i in range(N):
    for j in range(N):
        if i == j:
            T_true[i, j] = T_diag
        elif abs(i - j) == 1:
            T_true[i, j] = T_off_diag
        else:
            T_true[i, j] = 0


H_true = T_true + V_matrix
H_dvr = T_dvr + V_matrix
# Diagonalize the Hamiltonian
E_dvr, psi_dvr = np.linalg.eigh(H_dvr)
E_true, psi_true = np.linalg.eigh(H_true)
olap = False
if olap:
    matplotlib.style.use('seaborn-v0_8-deep')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    # custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_seaborn', [b, g, r], N=256,)
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue_white_red', [b, (1,1,1), r], N=256
    )
    # blue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_map", ['white', colors[0]], N=256)
    # Calculate the overlap between the DVR and true eigenstates
    n_levels = 7 # Number of states to compare
    S = np.zeros((n_levels, n_levels))
    for i in range(n_levels):
        for j in range(n_levels):
            # Calculate the overlap integral
            overlap = np.sum(psi_dvr[:, i].conj() * psi_true[:, j]) 
            S[i, j] = np.abs(overlap) ** 2
    # Plot the overlap matrix
    fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
    im = ax.imshow(S, cmap=custom_cmap, vmin=0, vmax=1, extent=[-0.5, n_levels - 0.5, n_levels - 0.5, -0.5], origin='upper')
    # ax.grid(False)
    major_ticks = np.arange(-0.5, n_levels)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.grid(which='major', color='white', linewidth=1.0)
    

    minor_ticks = np.arange(n_levels)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    # ax.set_yticks(np.arange(n_levels) + 0.5, minor=True)
    # ax.set_xticks(np.arange(n_levels) + 0.5, minor=True,)
    # ax.grid(True, which='minor', color='white', linewidth=0.5)
    ax.set_xticklabels([f"$\psi_{{{j}}}^{{\\mathrm{{true}}}}$" for j in range(n_levels)], minor=True)
    ax.set_yticklabels([f"$\psi_{{{i}}}^{{\\mathrm{{DVR}}}}$" for i in range(n_levels)], minor=True)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(which='major', length=0, labelbottom=False, labelleft=False)
    ax.tick_params(which='minor', length=0) 
    ax.set_xlabel("Numerically Exact Eigenstates (FD)")
    ax.set_ylabel("DVR Eigenstates")
    ax.set_title("Overlap Matrix $|\\langle \\psi_n^{\\text{DVR}} | \\psi_m^{\\text{true}} \\rangle|^2$")
    fig.colorbar(im, ax=ax, label="Overlap")


    fig.subplots_adjust(left=-0.7, right=1.0, bottom=0.2, top=0.9)
    # plt.savefig('../doc/figs/dvr_validation_overlap.pdf')
    plt.show()
    exit()


spectrum = True
if spectrum:
    # Visualize the first 5 energy eigenlevels
    matplotlib.style.use('seaborn-v0_8-deep')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    n_levels = 16 # Number of levels to plot
    xpos = 0.0    # x-location for energy ladders
    width = 0.25   # half-width of ladder line
    # Find deviance between DVR and true eigenvalues per level
    dev = np.zeros(n_levels)
    for n in range(n_levels):
        E_dvr_n = E_dvr[n]
        E_true_n = E_true[n]
        dev[n] = np.abs(E_dvr_n - E_true_n)
    levels = np.arange(n_levels)
    breakpoint()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.4, 0.4))
    # Plot the deviance
    ax[1].semilogy(np.arange(n_levels), dev, label='Deviance', marker='x', )
    ax[1].set_xlabel('Energy level')
    ax[1].set_ylabel(r'Absolute deviance $|E_{DVR} - E_{exact}|$')
    ax[1].set_title('Spectrum deviance')
    ax[1].set_xticks(levels[::3])

    # Plot each energy level
    for i in range(n_levels):
        E_t = E_true[i]
        E_d = E_dvr[i]
        
        # True energy: solid line
        ax[0].hlines(E_t, xpos - width, xpos + width, color=b, linewidth=2, label='True' if i == 0 else "")
        
        # DVR energy: dashed line
        ax[0].hlines(E_d, xpos - width, xpos + width, color=r, linewidth=2, linestyle='--', label='DVR' if i == 0 else "")


    ax[0].set_xlim(xpos - width -0.1 , xpos + width + 0.1)
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylabel('Energy')
    ax[0].set_title('Energy spectrum')
    ax[0].legend(bbox_to_anchor=(1.05, 1), )

    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.3)

    plt.tight_layout()
    # plt.savefig('../doc/figs/dvr_validation.pdf')
    plt.show()
    exit()