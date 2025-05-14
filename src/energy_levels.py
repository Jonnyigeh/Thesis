import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
from scipy.optimize import minimize
from scipy.special import erf
from tqdm import tqdm

# Local imports
import quantum_systems as qs
from sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS
from utils.visualization import find_figsize


def build_quantum_system(
    params,
    l,
    grid_length,
    num_grid_points,
    a,
    alpha,
    num_lr,
):
    """
    Builds and returns a quantum system Hamiltonian using a double-well Morse potential and bipartite Hartree calculation.

    Parameters:
    - params: List of Morse potential parameters.
    - l, grid_length, num_grid_points: Grid and basis parameters.
    - a, alpha: Morse basis parameters.
    - num_l, num_r: Number of states for left and right subsystems.
    - qs: Quantum simulation module containing potential and basis classes.
    - sinc_BHS: Function to solve bipartite Hartree system.

    Returns:
    - H: Full Hamiltonian matrix (with interaction).
    - E: Eigenenergies.
    - C: Eigenvectors.
    - eps_l, eps_r: Single-particle energies for left and right.
    - h_l, h_r: Effective left and right Hamiltonians (in Hartree basis).
    """

    potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(*params)
    num_l = num_lr
    num_r = num_lr
    basis = qs.ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        _a=a,
        alpha=alpha,
        potential=potential,
        dvr=True,
    )

    h_l = basis._h_l
    h_r = basis._h_r
    u_lr = basis._ulr

    bhs = sinc_BHS(h_l, h_r, u_lr, num_l, num_r)
    bhs.solve()

    eps_l = bhs.eps_l
    eps_r = bhs.eps_r
    c_l = bhs.c_l
    c_r = bhs.c_r

    # Transform Hamiltonians to Hartree basis
    # h_l = c_l.conj().T @ h_l @ c_l
    # h_r = c_r.conj().T @ h_r @ c_r

    # # Transform interaction tensor to Hartree basis
    # u_lr_transformed = np.einsum(
    #     'ai, bj, ab, ak, bl -> ijkl',
    #     c_l.conj(), c_r.conj(), u_lr, c_l, c_r
    # )

    # # Construct full Hamiltonian
    # H_0 = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)
    # U = u_lr_transformed.reshape(*H_0.shape)
    # H = H_0 + U

    # # E, C = np.linalg.eigh(H)

    return eps_l, eps_r
# def plot_energy_levels(
#     trans_energies1,
#     trans_energies2,
#     num_levels=3,
# ):
#     matplotlib.style.use('seaborn-v0_8')
#     colors = sns.color_palette()
#     b = colors[0]
#     g = colors[1]
#     r = colors[2]
#     fig, ax = plt.subplots(nrows=1, ncols=2,figsize=find_figsize(1.2, 0.4))

    

#     plt.tight_layout()
#     plt.show()


def plot_energy_levels(
    trans_energies1,
    trans_energies2,
    num_levels=3,
):
    matplotlib.style.use('seaborn-v0_8')
    colors = sns.color_palette()

    def plot_config(ax, energies, title, config_label):
        base_x = 0
        width = 0.6
        spacing = 1.2
        labels = {
            'w_l1': r'$|10\rangle$',
            'w_r1': r'$|01\rangle$',
            'w_L1 + w_R1': r'$|11\rangle$',
            'w_l2': r'$|20\rangle$',
            'w_r2': r'$|02\rangle$',
        }

        group_colors = {
            'w_l1': colors[0],
            'w_r1': colors[0],
            'w_l2': colors[1],
            'w_r2': colors[1],
            'w_L1 + w_R1': colors[2],
        }

        for i, (key, energy) in enumerate(energies.items()):
            color = group_colors.get(key, 'black')

            # Vertical stem
            ax.vlines(i, 0, energy, color='gray', linewidth=0.8, linestyle='--')

            # Energy platform
            ax.hlines(energy, i - width / 2, i + width / 2, color=color, linewidth=2.5)

            # Annotation
            ax.text(i, energy + 0.1, labels[key], ha='center', va='bottom', fontsize=10)

        ax.set_title(title)
        ax.set_xlim(-1, len(energies) * spacing)

    # Create broken axes
    fig = plt.figure(figsize=(6, 4))
    bax = brokenaxes(ylims=((0, 1.5), (2.3, 2.6)), hspace=0.05)
    
    # Plot on broken axes
    plot_config(bax, trans_energies1, 'Configuration I', 'config1')
    plot_config(bax, trans_energies2, 'Configuration II', 'config2')

    plt.tight_layout()
    plt.show()








if __name__ == "__main__":
    # Parmeters
    params1 = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
    params2 = [73.44693037, 71.99175625 ,29.16144963 ,29.16767609, 42.79831711]
    alpha = 1.0
    num_lr = 3
    el1, er1, = build_quantum_system(
        params=params1,
        l=25,
        num_lr=num_lr,
        num_grid_points=400,
        grid_length=200,
        a=0.25,
        alpha=1.0,
    )
    el2, er2, = build_quantum_system(
        params=params2,
        l=25,
        num_lr=num_lr,
        num_grid_points=400,
        grid_length=200,
        a=0.25,
        alpha=1.0,
    )
    el1 -= el1[0]
    el2 -= el2[0]
    er1 -= er1[0]
    er2 -= er2[0]
    trans_energies1 = {
    'w_l1': float(round(el1[1] / (2 * np.pi), 5)),
    'w_r1': float(round(er1[1] / (2 * np.pi), 5)),
    'w_L1 + w_R1': float(round((el1[1] + er1[1]) / (2 * np.pi), 5)),
    'w_l2': float(round(el1[2] / (2 * np.pi), 5)),
    'w_r2': float(round(er1[2] / (2 * np.pi), 5)),
    }
    trans_energies2 = {
    'w_l1': float(round(el2[1] / (2 * np.pi), 5)),
    'w_r1': float(round(er2[1] / (2 * np.pi), 5)),
    'w_L1 + w_R1': float(round((el2[1] + er2[1]) / (2 * np.pi), 5)),
    'w_l2': float(round(el2[2] / (2 * np.pi), 5)),
    'w_r2': float(round(er2[2] / (2 * np.pi), 5)),
    }
    plot_energy_levels(
        trans_energies1=trans_energies1,
        trans_energies2=trans_energies2
    )
        
