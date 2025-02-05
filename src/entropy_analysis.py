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
from bipartite_hartree import BipartiteHartreeSolver as BHS   
    


def entropy_analysis(d, l, num_func, grid_length, num_grid_points, alpha=1.0, a=0.25, potential = None, verbose=False):
    """Analyse the entropy of the system as a function of separation.
    
    args:
        d (float): The separation between the well minimas.
    
    returns:
        entropy (np.ndarray): The Von Neumann entropy of the system.
    """
    # System parameters
    if potential is None:
        potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
                    D_a=50.0,
                    D_b=39.0,
                    k_a=10.0,
                    k_b=15.0,
                    d=d,
                )
    basis = qs.ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        alpha=alpha,
        _a=a,
        potential=potential,
    )
    bhs_solver = BHS(
        h_l=basis.h_l,
        h_r=basis.h_r,
        u_lr=basis._ulr,
        num_basis_l=num_func,
        num_basis_r=num_func,
    )
    e_l, c_l, e_r, c_r = bhs_solver.solve()
    h_l = c_l.conj().T @ basis.h_l @ c_l
    h_r = c_r.conj().T @ basis.h_r @ c_r
    u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), basis._ulr, c_l, c_r)
    # Compute energies
    h = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)
    u = u_lr.reshape(*h.shape)
    eps, C = np.linalg.eigh(h + u)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
    if verbose:
        print(f"Separation: {d}")
    entropies = []
    # print(f"Energies {eps}")
    for i in range(num_func):
    # for i in range(C.shape[0]):
        # Find density matrices
        rho = np.einsum('p, q -> pq', C[i], C[i].conj().T).reshape(num_func, num_func, num_func, num_func)
        # Trace out the subsystems
        rho_l = np.trace(rho, axis1=0, axis2=2)
        breakpoint()
        # rho_r = np.trace(rho, axis1=1, axis2=3)
        # Compute entropies
        eigs_l = np.linalg.eigvalsh(rho_l)
        # eigs_r = np.linalg.eigvalsh(rho_r)
        entropy_l = -np.sum(eigs_l * np.log(eigs_l + 1e-15))
        # entropy_r = -np.sum(eigs_r * np.log(eigs_r + 1e-15))
        entropies.append(entropy_l)
        if verbose:
            print(f"State {i}: {np.real(C[i])**2}")
            print(f"Entropy in the {i}-th state: {entropy_l:6f}")
        # print(f"Deviation between subsystem L and R: {np.abs(entropy_l - entropy_r):6e}")
    
    return eps, C, entropies

def visualize(C_, num_func, entropy, tol=1e-8):
    from matplotlib.colors import TwoSlopeNorm
    import seaborn as sns
    # sns.set_theme()
    # C = np.abs(C_)**2
    C = np.real(C_)
    C[np.abs(C)**2 < tol] = np.nan
    fig, axs = plt.subplots(1, num_func, figsize=(10, 10))#, constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1) # colorbar range
    for i, ax in enumerate(axs):
        im = ax.imshow(C[:, i].reshape(num_func,num_func), cmap="RdBu", norm=norm, origin="lower", interpolation="none")
        ax.set_xlabel("Right Well Index")
        ax.set_ylabel("Left Well Index")
        ax.set_title(f"State {i} | entropy: {entropy[i]:.6f}")
        # Set minor ticks
        ax.set_xticks(np.arange(-0.5, num_func, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, num_func, 1.0), minor=True)
        ax.tick_params(which='minor', bottom=False, left=False) # Hide minor ticks
        # Major ticks
        ax.set_xticks(np.arange(0, num_func, 1.0))
        ax.set_yticks(np.arange(0, num_func, 1.0))
        # Gridlines based on minor ticks, and major ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1.0)
        # ax.grid(which='major', color='k', linestyle='-', linewidth=2.0)
        # ax.spines['top'].set_visible(False)

    fig.colorbar(im, ax=axs, orientation="horizontal", label="Probability")
    plt.show()
    # plt.show(block=False)
    # plt.pause(4.0)
    # plt.close()

if __name__ == "__main__":
    separations = [25, 50, 75, 100, 200]
    # separations = [50]
    grid_length = 400
    num_grid_points = 4_001
    l = 15
    num_func = 3
    alpha = 1.0
    a = 0.01
    en = []
    Cs = []
    eps = []
    DL = [40, 60, 80]
    DR = [40, 60, 80]
    kL = [20, 40, 60]
    kR = [20, 40, 60]
    d = 50
    ep, C, ent = entropy_analysis(d, l, num_func, grid_length, num_grid_points, alpha, a, verbose=False)
    breakpoint()
    import json
    save_data = []
    for d in separations:
        for dl in tqdm.tqdm(DL):
            for dr in tqdm.tqdm(DR, leave=False):
                for kl in tqdm.tqdm(kL, leave=False):
                    for kr in tqdm.tqdm(kR, leave=False):
                        potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
                            D_a=dl,
                            D_b=dr,
                            k_a=kl,
                            k_b=kr,
                            d=d,
                        )
                        ep, C, ent = entropy_analysis(d, l, num_func, grid_length, num_grid_points, alpha, a, potential=potential, verbose=False)
                        save_data.append({
                            "separation": d,
                            "DL": dl,
                            "DR": dr,
                            "kL": kl,
                            "kR": kr,
                            "entropies": ent,
                            "energies": ep,
                            "C": C,
                        })
                        # print(f"Separation: {d}, DL: {dl}, DR: {dr}, kL: {kl}, kR: {kr}")
                        # print(f"Entropies: {ent}")
                        # print('\n')
                        # en.append(ent)
                        # Cs.append(C)
                        # eps.append(ep)
                        # # print('\n')
                        # visualize(C, num_func, entropy=ent)
    # try:
    #     with open("data/entropies_various_params_150125.json", "w") as f:
    #         json.dump(save_data, f, indent=4)
    # except:
    #     breakpoint()
    exit()    
    for d in separations:
        # print(f"Separation: {d}")
        ep, C, ent = entropy_analysis(d, l, num_func, grid_length, num_grid_points, alpha, a, verbose=False)
        en.append(ent)
        Cs.append(C)
        eps.append(ep)
        # print('\n')
    
        visualize(C, num_func, entropy=ent)
    exit()
