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
from utils.potential import MorsePotentialDW
from utils.qd_system import ODMorse
from utils.sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS  
from utils.visualization import find_figsize


def TIHF(h, u, num_func, n_particles, max_iters=10_000, epsilon=1e-10, verbose=True):
        """Solve the time-independent Hartree-Fock equations.
        
        Procedure:
        Set an initial guess for the eigenvectors. Here we've used the identity matrix, so the initial guess is the standard computational basis. [1, 0, 0, 0, ...] etc.
        Loop over the following steps:
        - Compute the Fock matrix using the initial guess, where a density matrix is created from the eigenvectors, and used to solve the one- and two-body matrix elements.
        - Diagonalize the Fock matrix to find the new eigenvectors.
        - Check for convergence, and if the convergence criterion is not met, repeat the procedure with the new eigenvectors. 

        args:
            t0          (float): Initial time.
            max_iters   (float): Maximum number of iterations.
            epsilon     (float): Convergence tolerance.

        returns: 
            energy (np.nparray): The energy of the system, i.e the eigenvalues of the system Hamiltonian.
            C (np.ndarray): The eigenvectors of the system Hamiltonian.
        """
        # Fock matrix
        def fill_fock_matrix(C):
            """Fill the Fock matrix.
            
            Computes the fock-operator matrix by usage of the one- and two-body matrix elements stored in quantum_system.GeneralOrbitalSystem,
            with the density matrix containing the eigenvectors found in the "C"-matrix.

            args:
                C  (np.ndarray): Matrix of (current) eigenvectors
            
            returns:
                fock (np.ndarray): Fock-operator matrix.
            """
            fock = np.zeros(h.shape, dtype=np.complex128)
            density_matrix = np.zeros((h.shape[0],h.shape[0]), dtype=np.complex128)
            for i in range(n_particles):
                density_matrix += np.outer(C[:, i], np.conj(C[:, i]))
            fock = np.einsum('ij, aibj->ab', density_matrix, u, dtype=np.complex128)        # Compute the two-body operator potential
            fock += h                                                                       # Add the one-body operator hamiltonian

            return fock, density_matrix

        # energy, C = scipy.linalg.eigh(np.eye(h.shape[0]), subset_by_index=[0, num_func - 1])
        energy, C = scipy.linalg.eigh(h, subset_by_index=[0, num_func - 1])
        fock, density_matrix = fill_fock_matrix(C)
        converged=False
        delta_E = 0.0
        e_list = []
        with tqdm.tqdm(total=max_iters,
                desc=rf"[Minimization progress, $\Delta E$ = {delta_E:.10f}]",
                position=0,
                colour="green",
                leave=True) as pbar:
            for i in range(max_iters):
                energy_new, C_ = scipy.linalg.eigh(fock, subset_by_index=[0, num_func - 1])
                e_list.append(energy_new[0])
                delta_E = np.linalg.norm(energy_new - energy) / len(energy)
                pbar.set_description(
                    rf"[Optimization progress, $\Delta E$ = {delta_E:.10f}]"
                )
                pbar.update(1)
                if delta_E < epsilon:
                    if verbose:
                        print(f"Converged in {i} iterations.")
                    converged=True
                    break
                C = C_
                energy = energy_new
                fock, _ = fill_fock_matrix(C)
                # Update with damping
                # _, density_matrix_new = fill_fock_matrix(C)
                # density_matrix = alpha * density_matrix_new + (1 - alpha) * density_matrix
                # fock = np.einsum('ij, aibj-> ab', density_matrix, u, dtype=np.complex128) + h

        if not converged:
            if verbose:
                raise RuntimeError(f"The solver failed to converged after maximum number (iters={max_iters}) of iterations was reached.")
            
        return energy, C


def construct_slater_determinants(spf_l, spf_r):
    """Construct the slater determinants from the single-particle functions."""
    num_spf_l = spf_l.shape[1]
    num_spf_r = spf_r.shape[1]
    num_spf = num_spf_l + num_spf_r
    num_sd = num_spf_l * num_spf_r
    sd = np.zeros((num_sd, num_spf), dtype=np.complex128)
    idx = 0
    # for i in range(num_spf_l):
    #     for j in range(num_spf_r):
    #         sd[i * num_spf_r + j, :] = np.kron(spf_l[:, i], spf_r[:, j]) - np.kron(spf_r[:, j], spf_l[:, i])
    for i in range(num_spf_l):
        for j in range(num_spf_r):
            sd[idx, :] = spf_l[:, i] * spf_r[:, j] - spf_r[:, j] * spf_l[:, i]
    return sd

def make_comparison(potential, alpha, a, num_grid_points, grid_length, num_func, l, n_particles, num_grid_points_sinc=400):
    # # Sinc basis Hartree-product WF
    num_grid_points_sinc = 800
    basis = ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points_sinc,
        _a=a,
        alpha=alpha,
        potential=potential,
        dvr=True,
    )
    h_l = basis._h_l
    h_r = basis._h_r
    u_lr = basis._ulr
    bhs = sinc_BHS(h_l, h_r, u_lr, num_func, num_func)
    bhs.solve()
    dinsting_eps_l = bhs.eps_l
    disting_eps_r = bhs.eps_r
    c_l = bhs.c_l
    c_r = bhs.c_r
    h_l = c_l.conj().T @ h_l @ c_l
    h_r = c_r.conj().T @ h_r @ c_r
    # New attempt nr.2, using two-step einsum to avoid memory issues. Possible due to sparsity in u matrix (i!=j, k!=l is zero by construction) 
    M = np.einsum('ia, ij, ic -> acj', bhs.c_l.conj(), u_lr, bhs.c_l, optimize=True)
    u_lr = np.einsum('acj, jb, jd -> abcd', M, bhs.c_r.conj(), bhs.c_r, optimize=True)
    H_ = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r) 
    U = u_lr.reshape(*H_.shape)
    H = H_ + U
    M = h_l.shape[0]               # number of one-particle functions
    pairs = [(i, j) for i in range(M) for j in range(i)]

    P = np.zeros((M*M, len(pairs)), dtype=complex)
    for idx, (i, j) in enumerate(pairs):
        e_i = np.zeros(M); e_i[i] = 1
        e_j = np.zeros(M); e_j[j] = 1
        psi_ij = np.kron(e_i, e_j)
        psi_ji = np.kron(e_j, e_i)
        P[:, idx] = (psi_ij - psi_ji) / np.sqrt(2)

    H_antisym = P.conj().T @ H @ P
    eps_asym, C_asym = np.linalg.eigh(H_antisym)
    disting_eps, disting_C = np.linalg.eigh(H)
    exch_energy = abs(eps_asym[0] - 0.5*(disting_eps[1] + disting_eps[2]))
    direct_energy = 0.5 * (disting_eps[1] + disting_eps[2])
    # direct_energy = np.einsum(
    #     'i, j, ijkl, k, l ->',
    #     c_l[:,0].conj(), c_r[:,0].conj(), basis._ulr, c_l[:,0], c_r[:,0], optimize=True
    # )
    # exch_energy = eps_asym[0] - direct_energy
    print(f"Exchange:{abs(eps_asym[0] - 0.5*(disting_eps[1] + disting_eps[2])):.5f}")

    # # Anti-symmetric WF 
    # indisting_basis = ODMorse(
    #     l=l,
    #     grid_length=grid_length,
    #     num_grid_points=num_grid_points_sinc,
    #     _a=a,
    #     alpha=alpha,
    #     potential=potential,
    #     anti_symmetric=True,
    #     dvr=True,
    # )
    # eps_hf, C_hf = TIHF(indisting_basis.h, indisting_basis.u, num_func=num_func, n_particles=n_particles)
    # h_mo = C_hf.conj().T @ indisting_basis.h @ C_hf       # shape (L,L)
    # u_mo = np.einsum(
    #     'ip, jq, ijkl, kr, ls -> pqrs',
    #     C_hf.conj(), C_hf.conj(), indisting_basis.u, C_hf, C_hf,
    #     optimize=True
    # )
    # occ = [0,1]
    # J = np.zeros((n_particles, n_particles))
    # K = np.zeros((n_particles, n_particles))
    # for i in occ:
    #     for j in occ:
    #         J[i, j] = u_mo[i, i, j, j]
    #         K[i, j] = u_mo[i, j, j, i]
    # E_HF = sum(eps_hf[i] for i in occ) \
    #     - 0.5 * sum(J[i, j] - K[i, j] for i in occ for j in occ)
    # print(f"HF energy: {E_HF:.5f}")
    return exch_energy, eps_asym, disting_eps, direct_energy
    # rho = np.zeros((indisting_basis.l,indisting_basis.l), dtype=np.complex128)
    # for i in range(n_particles):
    #     rho += np.outer(indisting_C[:,i], np.conj(indisting_C[:,i]).T)
    # # Find total HF energy
    # E_2body = 0
    # E_ex = 0
    # E_dir = 0
    # for i in range(n_particles):
    #     for j in range(n_particles):
    #         E_2body += 0.5 * rho[i, i] * rho[j, j] * (indisting_basis.u[i, j, i, j] - indisting_basis.u[i, j, j, i])
    #         # Compute the diect and exchange terms
    #         E_ex += 0.5 * rho[i, i] * rho[j, j] * indisting_basis.exchange[i, j, i, j].real
    #         E_dir += 0.5 * rho[i, i] * rho[j, j] * indisting_basis.direct[i, j, i, j].real
    # # E_1body = np.sum(np.einsum('ij,ij->ij', indisting_basis.h, rho))
    # E_1body = np.trace(indisting_basis.h @ rho).real
    # E = E_1body + E_2body
    # # 1) Build MO‐transformed integrals
    # h_mo = indisting_C.conj().T @ indisting_basis.h @ indisting_C       # shape (L,L)
    # u_mo = np.einsum(
    #     'pi, qj, ijkl, rk, sl -> pqrs',
    #    indisting_C.conj(), indisting_C.conj(), indisting_basis.u, indisting_C, indisting_C,
    #     optimize=True
    # )

    # # 2) Identify occupied orbitals (first n_particles)
    # occ = range(n_particles)

    # # 3) Compute Coulomb & exchange sums
    # J = np.zeros((n_particles,n_particles))
    # K = np.zeros((n_particles,n_particles))
    # for i in occ:
    #     for j in occ:
    #         J[i,j] = u_mo[i,i,j,j]
    #         K[i,j] = u_mo[i,j,j,i]

    # # 4) HF total energy
    # E_HF = sum(indisting_eps[i] for i in occ) \
    #     - 0.5 * sum(J[i,j] - K[i,j] for i in occ for j in occ)
    # print("HF energy =", E_HF)
    
    
    
    
    
    
    # # calculate ground state overlap
    # S = 0
    
    # print(f"Overlap between distinguishable and indistinguishable ground state: {S:.5f}")
    # print(f"Distinguishable energies: {disting_eps[:6]}")
    # print(f"Indistuinguishable energies: {indisting_eps[:6]}")
    # print(f"HF energy: {E}")
    # print(f"Deviation in ground state estimate: {np.abs(disting_eps[0] - E):.5f}")
    # print(f'Relative error: {np.abs(disting_eps[0] - E) / E.real:.5f}')
    # print(f"Exchange energy: {E_ex:.5f}")
    # print(f"Direct energy: {E_dir:.5f}")
    # print(f"Exchange fraction: {E_ex.real / (E_ex.real + E_dir.real):.5%}")
    
    breakpoint()
    # return eps_asym, E, disting_eps, S, E_ex, E_dir, 
if __name__ == "__main__":
    separations = [5, 10, 15, 25, 50, 75, 100]
    separations = [20]
    separations = [1, 2, 3, 5, 10, 20, 25]
    alpha = 1.0
    a = 0.25
    num_grid_points = 4_001
    l = 10
    n = 2 # Number of particles
    num_func = 4
    grid_length = 400
    asym_en = []
    E_n = []   
    prod_e = []
    overlaps = []
    exchange_term = []
    direct_term = []
    for d in separations:    
        params_close = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, d]
        # params0306 = [63.68144808, 43.05416999 ,10.69127355 ,10.90371128, d] #16.02656697]
        # paramsI_also_long = [ 95.70685722 , 76.66934364 , 54.2000149  , 10., d]
        # params = [61.48354464, 37.12075554, 22.36820274, 8.1535532, 16.58949561]
        params = params_close
        params = [76.50992129, 48.1146782 , 13.351184 ,  13.68434784 , d]
        print(f"Separation: {d}")
        # Should maybe test this also, and put in thesis for parameters in th emiddle of config I and config II?
        # Currently use params found from config II
        potential = MorsePotentialDW(
            *params,
                # D_a=70,
                # D_b=70,
                # k_a=31,
                # k_b=25,
                # d=d
            )
        exchange_en, eps_asym, eps, direct_en = make_comparison(potential=potential, alpha=alpha, a=a, num_grid_points=num_grid_points, grid_length=grid_length, num_func=num_func, l=l, n_particles=n)
        exchange_term.append(exchange_en)
        direct_term.append(direct_en)
        asym_en.append(eps_asym[0])
        E_n.append(0.5 * (eps[2] + eps[1] ))
        # eps_asym, E, prod_en, S_, exch, direct = make_comparison(potential=potential, alpha=alpha, a=a, num_grid_points=num_grid_points, grid_length=grid_length, num_func=num_func, l=l, n_particles=n)
        # asym_en.append(eps_asym[0])
        # E_n.append(E)
        # exchange_term.append(exch)
        # direct_term.append(direct)
        # prod_e.append(prod_en[0])
        # overlaps.append(S_)
        # print("\n")
    delta_E = np.array(E_n) - np.array(asym_en)
    matplotlib.style.use('seaborn-v0_8-deep')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4))
    ax[0].plot(separations, E_n, 'x-', color=b, label='Hartree energy')
    ax[0].plot(separations, asym_en, 'x-', color=g, label='CI energy')
    ax[0].set_xlabel('Separation [a.u.]')
    ax[0].set_ylabel('Energy [a.u.]')
    ax[0].set_title('Ground state energies')
    ax[1].plot(separations, np.asarray(delta_E), 'x-', color=r, label=r'$\Delta E$')
    ax[1].set_xlabel('Separation [a.u.]')
    ax[1].set_ylabel(r'$\Delta E$ [a.u.]')
    ax[1].set_title('Ground state deviation')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    # fig.subplots_adjust(left=0.1,
    #                 right=0.9,  # increased margin on the right
    #                 top=0.9,
    #                 bottom=0.15,
    #                 wspace=0.3)
    plt.savefig("../doc/figs/exchange_shift.pdf")
    plt.show()
    # exit()
    # data = {
    #     "separations": separations,
    #     "asym_energies": asym_en,
    #     "prod_energies": prod_e,
    #     "E_n": E_n,
    #     "overlaps": overlaps,
    #     "exchange_terms": exchange_term,
    #     "direct_terms": direct_term,
    # }
    # import pickle
    # with open("data/distinguishable_particle_breakdown0306.pkl", "wb") as f:
    #     pickle.dump(data, f)


    # exit()
    # # # Load data
    # import pickle
    # from utils.visualization import find_figsize
    # with open("data/distinguishable_particle_breakdown.pkl", "rb") as f:
    #     data = pickle.load(f)

    # E_n = data['E_n']
    # separations = data['separations']
    # asym_en = data['asym_energies']
    # breakpoint()
    # separations = data['separations']
    # asym_en = data['asym_energies']
    # prod_e = data['prod_energies']
    # E_n = data['E_n']
    # matplotlib.style.use('seaborn-v0_8')
    # colors = sns.color_palette()
    # b = colors[0]
    # g = colors[1]
    # r = colors[2]
    # delta_E = np.array(E_n) - np.array(asym_en)

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.45))
    # ax[0].plot(separations, [np.abs(E) for E in E_n], 'o-', color=r, label='HF energy')
    # # ax[0].plot(separations, [E.real for E in prod_e], 'o-', color=g, label='Distinguishable product energy')
    # ax[0].plot(separations, [np.abs(E) for E in asym_en], 'o-', color=b, label='Hartree energy')
    # ax[0].set_xlabel('Separation [a.u.]')
    # ax[0].set_ylabel('Energy [a.u.]')
    # ax[0].set_title('Ground state energies')
    # ax[1].plot(separations, delta_E, 'o-', color=r, label=r'$\Delta E$')
    # ax[1].set_xlabel('Separation [a.u.]')
    # ax[1].set_ylabel(r'$\Delta E$ [a.u.]')
    # ax[1].set_title('Deviation in ground state estimate')
    # ax[0].legend()
    # ax[1].legend()
    # plt.tight_layout()
    # fig.subplots_adjust(left=0.1,
    #                 right=0.9,  # increased margin on the right
    #                 top=0.9,
    #                 bottom=0.15,
    #                 wspace=0.3)
    # plt.savefig("../doc/figs/exchange_shift.pdf")
    # plt.show()
    # breakpoint()

