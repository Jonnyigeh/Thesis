import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
from tqdm import tqdm
from scipy.linalg import svdvals
from scipy.optimize import minimize, differential_evolution, dual_annealing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Local imports
from utils.sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS
from utils.potential import MorsePotentialDW
from utils.qd_system import ODMorse
from utils.visualization import find_figsize

def linear_lamba_t(t):
    lmbda = np.linspace(0, 2, len(t))

    return lmbda

def _find_VN_entropies(rho):
    """Find entropy from reduced density matrix"""
    eigs = np.linalg.eigvalsh(rho)
    return -np.sum(eigs * np.log2(eigs + 1e-15))    
def _make_density_matrix(C):
    # self._rho = np.zeros((self.num_func ** 2, self.num_func ** 2), dtype=np.complex128)
    # for n in range(self.n_particles):
    #     self._rho += np.outer(C[n], np.conj(C[n]).T)
    _rho = np.outer(C, np.conj(C))

    return _rho
def set_system(params, c_l=None, c_r=None, initial=True):
    grid_length = 200
    num_grid_points = 400
    a = 0.1
    alpha = 1.0
    l = 25
    num_l = 4
    num_r = 4
    potential = MorsePotentialDW(
        *params,
    )
    basis = ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        _a=a,
        alpha=alpha,
        potential=potential,
        dvr=True,
    )
    # H = basis.H
    # E, C = np.linalg.eigh(H)
    h_l = basis._h_l
    h_r = basis._h_r
    u_lr = basis._ulr
    u_lr4 = basis._ulr4d
    if initial:
        bhs = sinc_BHS(h_l, h_r, u_lr, num_l, num_r)
        bhs.solve()
        eps_l = bhs.eps_l
        eps_r = bhs.eps_r
        c_l = bhs.c_l
        c_r = bhs.c_r
    h_l = c_l.conj().T @ h_l @ c_l
    h_r = c_r.conj().T @ h_r @ c_r
    M = np.einsum('ia, ij, ic -> acj', c_l.conj(), u_lr, c_l, optimize=True)
    u_lr = np.einsum('acj, jb, jd -> abcd', M, c_r.conj(), c_r, optimize=True)

    H_ = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r) 
    U = u_lr.reshape(*H_.shape)
    H = H_ + U
    eps, C = np.linalg.eigh(H)

    S = np.zeros(len(eps))
    for i in range(len(eps)):
        _rho = _make_density_matrix(C[:,i]) # Each column is the energy eigenstates
        rho = np.trace(_rho.reshape(h_l.shape[0], h_l.shape[0], h_l.shape[0], h_l.shape[0]), axis1=0, axis2=2)
        # and then entropy
        S[i] = _find_VN_entropies(rho)
    if initial:
        return c_l, c_r
    return eps, C, S
def update_params(lmbda):
    config_I = [62.17088395, 60.73364357 ,19.89474221 ,21.81940414, 15.        ]
    config_II = [62.97325982, 64.11742637 ,13.22714092 ,13.09781006 ,14.95744294]
    """Update the parameters of the system."""
    params = (1 - lmbda) * np.array(config_I) + lmbda * np.array(config_II)
    return params



def plot_entanglement_peak(lmbda, S_mat):
    matplotlib.style.use('seaborn-v0_8-deep')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    l = colors[3]
    fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
    ax.plot(lmbda, S_mat[:, 0], label=r'S$[\psi_1]$', color=b, lw=2)
    ax.plot(lmbda, S_mat[:, 1], label=r'S$[\psi_2]$', color=g, lw=2)
    ax.plot(lmbda, S_mat[:, 2], label=r'S$[\psi_3]$', color=r, lw=2, linestyle='--')
    ax.plot(lmbda, S_mat[:, 3], label=r'S$[\psi_4]$', color=l, lw=2, linestyle='--')
    ax.set_xlabel(r'$\lambda$'), 
    ax.set_ylabel(r'Entanglement entropy $S$ ')
    ax.set_title(r'Entanglement as function of $\lambda$')
    ax.vlines(x=1.0, ymin=-0.5, ymax=1.0, color='k', linestyle='--', lw=1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 2.0)
    axins = inset_axes(ax,
                       width="40%",   # percent of parent_bbox width
                       height="50%",  # percent of parent_bbox height
                       loc='upper right',
                       borderpad=1)
    # plot same curves on inset
    axins.plot(lmbda, S_mat[:, 0], color=b, lw=2)
    axins.plot(lmbda, S_mat[:, 2], color=r, lw=2)
    axins.plot(lmbda, S_mat[:, 1], color=g, lw=2, ls='--')
    axins.plot(lmbda, S_mat[:, 3], color=l, lw=2, ls='--')
    axins.vlines(1.0, 0, 1, color='k', ls='--', lw=0.5)

    # restrict the view to the peak region
    axins.set_xlim(0.98, 1.02)
    axins.set_ylim(0.0, 1.05)
    axins.set_xticks([0.98, 1.00, 1.02])
    axins.set_yticks([0.0, 0.5, 1.0])

    # draw a box and connecting lines
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.legend(loc='upper left' )
    plt.tight_layout(rect=[0,0, 0.95, 1])
    plt.savefig('../doc/figs/entanglement_peak.pdf')
    plt.show()

def plot_energy_curves(lmbda, eps_mat):
    matplotlib.style.use('seaborn-v0_8-deep')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    l = colors[3]
    fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
    ax.plot(lmbda, eps_mat[:, 0], label=r'$E[\psi_0]$', color=b, lw=2)
    ax.plot(lmbda, eps_mat[:, 1], label=r'$E[\psi_1]$', color=g, lw=2)
    ax.plot(lmbda, eps_mat[:, 2], label=r'$E[\psi_2]$', color=r, lw=2)
    ax.plot(lmbda, eps_mat[:, 3], label=r'$E[\psi_3]$', color=l, lw=2)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Energy [a.u]')
    ax.set_title(r'Energy as function of $\lambda$')
    ax.vlines(x=1.0, ymin=-0.5, ymax=20.0, color='k', linestyle='--', lw=1.0)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(2.0, 14.0)
    axins = inset_axes(ax,
                       width="40%", height="50%",
                       loc='upper right', borderpad=1.2)
    for idx, col in enumerate((b, g, r, l)):
        axins.plot(lmbda, eps_mat[:,idx], color=col, lw=1)
    axins.vlines(1.0,
                 eps_mat[(lmbda>0.995)&(lmbda<1.005),1:].min(),
                 eps_mat[(lmbda>0.995)&(lmbda<1.005),1:].max(),
                 color='k', linestyle='--', linewidth=0.8)

    # Zoom limits
    axins.set_xlim(0.99975, 1.00025)
    ylo = eps_mat[(lmbda>0.99975)&(lmbda<1.00025),1:3].min() - 0.001
    yhi = eps_mat[(lmbda>0.99975)&(lmbda<1.00025),1:3].max() + 0.001
    axins.set_ylim(ylo, yhi)

    axins.set_xticks([0.99975, 1.000, 1.00025])
    # axins.set_yticks([])  # hide yticks if preferred

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.legend(loc='upper left')
    plt.tight_layout(rect=[0,0, 0.95, 1])
    plt.savefig('../doc/figs/energy_curves_avoided_crossing.pdf')
    plt.show()

if __name__ == "__main__":
    run_program = False
    if run_program:
        config_I = [62.17088395, 60.73364357 ,19.89474221 ,21.81940414, 15.        ]
        config_II = [62.97325982, 64.11742637 ,13.22714092 ,13.09781006 ,14.95744294]
        # Set initial HS transformation
        c_l, c_r = set_system(config_I, initial=True)

        lmbda = linear_lamba_t(np.linspace(0, 1, 1001))
        S_mat = np.zeros((len(lmbda), 4))  # Assuming 4 energy levels for the system
        eps_mat = np.zeros((len(lmbda), 4))  # Assuming 4
        for i, l in tqdm(enumerate(lmbda)):
            params = update_params(l)
            eps, C, S = set_system(params, c_l, c_r, initial=False)
            S_mat[i] = S[:4]  # Store the first 4 entropies
            eps_mat[i] = eps[:4]  # Store the first 4 energies

        data = {
            "lmbda": lmbda,
            "S_mat": S_mat,
            "eps_mat": eps_mat,
        }
    import pickle
    with open('data/entanglement_peak_data.pkl', 'rb') as f:
        data = pickle.load(f)
    lmbda = data["lmbda"]
    S_mat = data["S_mat"]
    eps_mat = data["eps_mat"]
    # plot_entanglement_peak(lmbda, S_mat)
    plot_energy_curves(lmbda, eps_mat)
    
