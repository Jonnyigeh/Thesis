import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import scipy.special
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy

# Local imports
import quantum_systems as qs        # Quantum systems library, will be swapped out with the new library at a later stage
# from bipartite_hartree import BipartiteHartreeSolver as BHS   
from sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS  
from utils.visualization import find_figsize



def entropy_analysis(params,
                    l,
                    num_func,
                    grid_length,
                    num_grid_points,
                    alpha=1.0,
                    a=0.25,
                    potential=None,
                    verbose=False,
                ):
    """Analyse the entropy of the system as a function of separation.
    
    args:
        d (float): The separation between the well minimas.
    
    returns:
        entropy (np.ndarray): The Von Neumann entropy of the system.
    """
    
    # System parameters
    if potential is None:
        potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
                   *params,
                )
    basis = qs.ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        alpha=alpha,
        _a=a,
        potential=potential,
        dvr=True,
    )
    bhs = sinc_BHS(
        h_l=basis.h_l,
        h_r=basis.h_r,
        u_lr=basis._ulr,
        num_basis_l=num_func,
        num_basis_r=num_func,
    )
    bhs.solve()
    eps_l = bhs.eps_l
    eps_r = bhs.eps_r
    c_l = bhs.c_l
    c_r = bhs.c_r
    h_l = c_l.conj().T @ basis.h_l @ c_l
    h_r = c_r.conj().T @ basis.h_r @ c_r
    u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', c_l.conj(), c_r.conj(), basis._ulr, c_l, c_r)
    # Compute energies
    h = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)
    u = u_lr.reshape(*h.shape)
    eps, C = np.linalg.eigh(h + u)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
    if verbose:
        print(f"Separation: {params[-1]}")
    entropies = []
    if verbose:
        print(f"Energies {eps}")
    for i in range(num_func**2):
        # 1) pick out the i-th TWO-PARTICLE eigenvector correctly:
        psi = C[:, i]   # shape (N,)

        # 2) build the pure‐state density and reshape
        rho_full = np.outer(psi, psi.conj()).reshape(
            num_func, num_func,   # left, right
            num_func, num_func    # left′, right′
        )

        # 3) partial trace over the RIGHT subsystem (axes 1 and 3)
        rho_L = np.trace(rho_full, axis1=1, axis2=3)  # shape (num_func, num_func)

        # 4) Von Neumann entropy
        eigs = np.linalg.eigvalsh(rho_L)
        S = -np.sum(eigs * np.log2(eigs + 1e-15))
        entropies.append(S)

        if verbose:
            probs = np.abs(psi)**2
            print(f"State {i} populations:", probs)
            print(f"Entropy of state {i}: {S:.6f}")

    return eps, C, entropies

def _visualize(C_, num_func, entropy, tol=1e-4):
    from matplotlib.colors import TwoSlopeNorm
    import seaborn as sns
    matplotlib.style.use('seaborn-v0_8')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    # sns.set_theme()
    # C = np.abs(C_)**2
    C = np.real(C_)
    C[np.abs(C)**2 < tol] = np.nan
    C_ = C[:(num_func)**2, :(num_func)**2]
    fig, axs = plt.subplots(1, num_func, figsize=find_figsize(1.2,0.4), constrained_layout=True, sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1) # colorbar range
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_seaborn', [b, g, r], N=256)
    for i, ax in enumerate(axs):
        im = ax.imshow(C_[:, i].reshape(num_func,num_func), cmap=custom_cmap, norm=norm, origin="lower", interpolation="none", aspect='equal')
        ax.set_xticks(np.arange(num_func))
        ax.set_yticks(np.arange(num_func))
        # ax.set_xlabel("R_j")
        # ax.set_ylabel("L_i")
        ax.set_title(fr"$\Psi_{i}$ | S: {entropy[i]:.3f}")
        # Set minor ticks
        ax.set_xticks(np.arange(num_func))
        ax.set_yticks(np.arange(num_func))
        ax.set_xticks(np.arange(-0.5, num_func, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, num_func, 1.0), minor=True)
        # Gridlines based on minor ticks, and major ticks
        ax.grid(False, which='major')
        ax.grid(which='minor' , linestyle='-', linewidth=1.0)
    
    axs[0].set_ylabel(r"Left: $\phi_i$")
    axs[1].set_xlabel(r"Right: $\phi_j$")
    fig.suptitle("Energyeigenstate population in configuration I")
    fig.colorbar(im, ax=axs, orientation="horizontal",
                 fraction=0.65, pad=0.1, )
    # plt.savefig('../doc/figs/state_populations_II.pdf')
    plt.show()
    # plt.show(block=False)
    # plt.pause(4.0)
    # plt.close()

def visualize(C_, num_func, entropy, tol=1e-7):
    from matplotlib.colors import TwoSlopeNorm
    import seaborn as sns
    matplotlib.style.use('seaborn-v0_8')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]

    C = np.real(C_)
    C[np.abs(C)**2 < tol] = np.nan

    # 2 rows × 3 cols, now showing states 0–5
    fig, axs = plt.subplots(
        2, 3,
        figsize=find_figsize(1.2, 0.8),   # doubled height for 2 rows
        constrained_layout=True,
        sharex=True, sharey=True
    )
    axs = axs.flatten()

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'custom_seaborn', [b, g, r], N=256
    )

    for i in range(6):
        ax = axs[i]
        im = ax.imshow(
            C[:, i].reshape(num_func, num_func),
            cmap=custom_cmap,
            norm=norm,
            origin="lower",
            interpolation="none",
            aspect='equal'
        )
        
        # remove all interior major gridlines
        ax.grid(False, which='major')
        # set major ticks
        ax.set_xticks(np.arange(num_func))
        ax.set_yticks(np.arange(num_func))
        ax.set_title(fr"$\Psi_{i}$ | S: {entropy[i]:.3f}")
        ax.grid(which='minor', linestyle='-', linewidth=1.0)
        # Set minor ticks
        ax.set_xticks(np.arange(-0.5, num_func, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, num_func, 1.0), minor=True)


    # turn off the extra 6th axis if num_func<6 (not needed here)
    # axs[5].axis('off')  # now used by Ψ₅

    # shared labels
    axs[3].set_ylabel(r"Left: $\phi_i$")
    axs[4].set_xlabel(r"Right: $\phi_j$")  # bottom‐middle

    fig.suptitle("Energyeigenstate population in configuration I")
    fig.colorbar(
        im, ax=axs, orientation="horizontal",
        fraction=0.5, pad=0.03
    )
    # plt.savefig('../doc/figs/state_populations_I.pdf')
    plt.show()
if __name__ == "__main__":
    separations = [25, 50, 75, 100, 200]
    # separations = [50]
    grid_length = 200
    num_grid_points = 400
    l = 25
    num_func = 4
    alpha = 1.0
    a = 0.01
    params_II = [73.44552758, 71.99208131, 29.16163181, 29.16725463, 42.80228627]
    params_I = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
    params = params_I
    ep, C, ent = entropy_analysis(params, l, num_func, grid_length, num_grid_points, alpha, a, verbose=False)
    visualize(C, num_func, entropy=ent)
    exit()
    
    # en = []
    # Cs = []
    # eps = []
    # DL = [40, 60, 80]
    # DR = [40, 60, 80]
    # kL = [20, 40, 60]
    # kR = [20, 40, 60]
    # d = 50
    # import json
    # save_data = []
    # for d in separations:
    #     for dl in tqdm.tqdm(DL):
    #         for dr in tqdm.tqdm(DR, leave=False):
    #             for kl in tqdm.tqdm(kL, leave=False):
    #                 for kr in tqdm.tqdm(kR, leave=False):
    #                     potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
    #                         D_a=dl,
    #                         D_b=dr,
    #                         k_a=kl,
    #                         k_b=kr,
    #                         d=d,
    #                     )
    #                     ep, C, ent = entropy_analysis(d, l, num_func, grid_length, num_grid_points, alpha, a, potential=potential, verbose=False)
    #                     save_data.append({
    #                         "separation": d,
    #                         "DL": dl,
    #                         "DR": dr,
    #                         "kL": kl,
    #                         "kR": kr,
    #                         "entropies": ent,
    #                         "energies": ep,
    #                         "C": C,
    #                     })
    #                     # print(f"Separation: {d}, DL: {dl}, DR: {dr}, kL: {kl}, kR: {kr}")
    #                     # print(f"Entropies: {ent}")
    #                     # print('\n')
    #                     # en.append(ent)
    #                     # Cs.append(C)
    #                     # eps.append(ep)
    #                     # # print('\n')
    #                     # visualize(C, num_func, entropy=ent)
    # # try:
    # #     with open("data/entropies_various_params_150125.json", "w") as f:
    # #         json.dump(save_data, f, indent=4)
    # # except:
    # #     breakpoint()
    # exit()    
    # for d in separations:
    #     # print(f"Separation: {d}")
    #     ep, C, ent = entropy_analysis(d, l, num_func, grid_length, num_grid_points, alpha, a, verbose=False)
    #     en.append(ent)
    #     Cs.append(C)
    #     eps.append(ep)
    #     # print('\n')
    
    #     visualize(C, num_func, entropy=ent)
    # exit()
