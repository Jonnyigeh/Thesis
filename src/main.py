# Library imports
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy

# Local imports
import quantum_systems as qs        # Quantum systems library, will be swapped out with the new library at a later stage
from bipartite_hartree import BipartiteHartreeSolver as BHS

if __name__ == "__main__":
    # Define the potential
    potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            D_a=15.0,
            D_b=15.0,
            k_a=9.0,
            k_b=9.0,
            d=15.0
        )
    # Define the basis-parameters
    alpha = 1.0
    a = 0.25
    num_grid_points = 1001
    grid_length = 1.25 * potential.d
    l = 8
    # Set up the basis
    basis = qs.ODMorse(
        l=l,
        grid_length=grid_length,
        num_grid_points=num_grid_points,
        _a=a,
        alpha=alpha,
        potential=potential
    )
    grid = basis.grid
    # Set up the bipartite Hartree solver
    bhs = BHS(
        h_l=basis.h_l,
        h_r=basis.h_r,
        u_lr=basis._ulr,
        num_basis_l=basis.l_sub,
        num_basis_r=basis.l_sub,
    )
    eps_l, c_l, eps_r, c_r = bhs.solve()
    
    # Transform the Hamiltonian matrix into the new hartree basis, and find the new basis functions from the coefficient matrices
    h_l = c_l.conj().T @ basis.h_l @ c_l
    h_r = c_r.conj().T @ basis.h_r @ c_r
    u_lr = np.einsum('ai, bj, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), basis._ulr, c_l, c_r)
    spf_l = c_l @ basis.spf_l
    spf_r = c_r @ basis.spf_r
    # Set up the Hamiltonian matrix in the Hatree basis
    h = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)
    u = u_lr.reshape(*h.shape)
    H = h + u
    # Diagonalize the Hamiltonian matrix
    eps, C = np.linalg.eigh(H)

