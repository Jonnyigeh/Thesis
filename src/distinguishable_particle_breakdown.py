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

def TIHF(h, u, num_func, n_particles, max_iters=1000, epsilon=1e-18, verbose=True):
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
                density_matrix += np.outer(np.conj(C[:, i]), C[:, i])
            fock = np.einsum('ij, aibj->ab', density_matrix, u, dtype=np.complex128)        # Compute the two-body operator potential
            fock += h                                                                       # Add the one-body operator hamiltonian
            breakpoint()
            return fock

        energy, C = scipy.linalg.eigh(np.eye(h.shape[0]), subset_by_index=[0, num_func - 1])
        fock = fill_fock_matrix(C)
        converged=False
        delta_E = 0.0
        e_list = []
        with tqdm.tqdm(total=max_iters,
                desc=rf"[Minimization progress, $\Delta E$ = {delta_E:.8f}]",
                position=0,
                colour="green",
                leave=True) as pbar:
            for i in range(max_iters):
                energy_new, C_ = scipy.linalg.eigh(fock)
                e_list.append(energy_new[0])
                delta_E = np.linalg.norm(energy_new - energy) / len(energy)
                pbar.set_description(
                    rf"[Optimization progress, $\Delta E$ = {delta_E:.8f}]"
                )
                pbar.update(1)
                if delta_E < epsilon:
                    if verbose:
                        print(f"Converged in {i} iterations.")
                    converged=True
                    break
                C = C_
                energy = energy_new
                fock = fill_fock_matrix(C)

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

def make_comparison(potential, alpha, a, num_grid_points, grid_length, num_func, l, n_particles):
    # Hartree-product WF
    disting_basis = qs.ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        _a=a,
        alpha=alpha,
        potential=potential
    )
    grid = disting_basis.grid
    num_l = num_func
    num_r = num_func
    breakpoint()
    disting_bhs = BHS(
        h_l=disting_basis.h_l,
        h_r=disting_basis.h_r,
        u_lr=disting_basis._ulr,
        num_basis_l=num_l,
        num_basis_r=num_r,
    )
    disting_eps_l, disting_c_l, disting_eps_r, disting_c_r = disting_bhs.solve()
    disting_h_l = disting_c_l.conj().T @ disting_basis.h_l @ disting_c_l
    disting_h_r = disting_c_r.conj().T @ disting_basis.h_r @ disting_c_r
    disting_u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', disting_c_l.conj(), disting_c_r.conj(), disting_basis._ulr, disting_c_l, disting_c_r)

    disting_h = np.kron(disting_h_l, np.eye(*disting_h_l.shape)) + np.kron(np.eye(*disting_h_r.shape), disting_h_r)
    disting_u = disting_u_lr.reshape(*disting_h.shape)
    
    disting_H = disting_h + disting_u
    disting_eps, disting_C = np.linalg.eigh(disting_H)

    # Anti-symmetric WF 
    indisting_basis = qs.ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        _a=a,
        alpha=alpha,
        potential=potential,
        anti_symmetric=True
    )
    # indisting_bhs = BHS(
    #     h_l=indisting_basis.h_l,
    #     h_r=indisting_basis.h_r,
    #     u_lr=indisting_basis._ulr,
    #     num_basis_l=num_l,
    #     num_basis_r=num_r,
    # )
    # indisting_eps_l, indisting_c_l, indisting_eps_r, indisting_c_r = indisting_bhs.solve()
    # indisting_h_l = indisting_c_l.conj().T @ indisting_basis.h_l @ indisting_c_l
    # indisting_h_r = indisting_c_r.conj().T @ indisting_basis.h_r @ indisting_c_r
    # indisting_u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', indisting_c_l.conj(), indisting_c_r.conj(), indisting_basis._ulr, indisting_c_l, indisting_c_r)
    # indisting_h = np.kron(indisting_h_l, np.eye(*indisting_h_l.shape)) - np.kron(np.eye(*indisting_h_l.shape), indisting_h_l)  + np.kron(np.eye(*indisting_h_r.shape), indisting_h_r) - np.kron(indisting_h_r, np.eye(*indisting_h_r.shape))
    # indisting_u = indisting_u_lr.reshape(*indisting_h.shape)
    # indisting_H = indisting_h + indisting_u
    # indisting_eps, indisting_C = np.linalg.eigh(indisting_H)

    indisting_eps, indisting_C = TIHF(indisting_basis.h, indisting_basis.u, num_func=num_func, n_particles=n_particles)

    # shift zero
    disting_eps -= disting_eps[0]
    indisting_eps -= indisting_eps[0]
    # compare the two
    print("Distinguishable particle system energy levels:", disting_eps[:6])
    print("Indistinguishable particle system energy levels:", indisting_eps[:6])
    print("Difference:", np.linalg.norm(disting_eps - indisting_eps))
    breakpoint()
if __name__ == "__main__":
    potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            D_a=35.0,
            D_b=35.0,
            k_a=15.0,
            k_b=15.0,
            d=35.0
        )
    alpha = 1.0
    a = 0.25
    num_grid_points = 1_001#1_001
    grid_length = 1.25 * potential.d
    l = 4
    n = 2 # Number of particles
    num_func = 5

    make_comparison(potential=potential, alpha=alpha, a=a, num_grid_points=num_grid_points, grid_length=grid_length, num_func=num_func, l=l, n_particles=n)
