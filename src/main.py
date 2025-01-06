# Library imports
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
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
    def find_entropy_energy(D_l=35, D_r=35, k_l=15, k_r=15, d=50):
        # Define the potential
        potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
                D_a=D_l,
                D_b=D_r,
                k_a=k_l,
                k_b=k_r,
                d=d
            )
        # Define the basis-parameters
        alpha = 1.0
        a = 0.25
        num_grid_points = 10_001#1_001
        grid_length = 1.25 * potential.d
        l = 12
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
        num_l = 10
        num_r = 10
        bhs = BHS(
            h_l=basis.h_l,
            h_r=basis.h_r,
            u_lr=basis._ulr,
            num_basis_l=num_l,
            num_basis_r=num_r,
        )
        # ..aaaand solve it
        eps_l, c_l, eps_r, c_r = bhs.solve()
        
        # Transform the Hamiltonian matrix into the new hartree basis, and find the new basis functions from the coefficient matrices
        h_l = c_l.conj().T @ basis.h_l @ c_l
        h_r = c_r.conj().T @ basis.h_r @ c_r
        u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), basis._ulr, c_l, c_r)
        spf_l = c_l.T @ basis.spf_l
        spf_r = c_r.T @ basis.spf_r
        # Set up the Hamiltonian matrix in the Hartree basis
        h = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)
        u = u_lr.reshape(*h.shape)
        H = h + u
        # Diagonalize the Hamiltonian matrix
        eps, C = np.linalg.eigh(H)
        # Shift the energy zero-point
        eps -= eps[0]
        # Find the density matrix
        D0 = np.einsum('p, q -> pq', C[0], C[0].conj().T).reshape(num_l, num_r, num_l, num_r)
        D1 = np.einsum('p, q -> pq', C[1], C[1].conj().T).reshape(num_l, num_r, num_l, num_r)
        D2 = np.einsum('p, q -> pq', C[2], C[2].conj().T).reshape(num_l, num_r, num_l, num_r)
        # Trace out the subsystems
        D0_l = np.trace(D0, axis1=0, axis2=2)
        D0_r = np.trace(D0, axis1=1, axis2=3)
        D1_l = np.trace(D1, axis1=0, axis2=2)
        D1_r = np.trace(D1, axis1=1, axis2=3)
        D2_l = np.trace(D2, axis1=0, axis2=2)
        D2_r = np.trace(D2, axis1=1, axis2=3)
        # Find the entropies
        eigs0l = np.linalg.eigvalsh(D0_l)
        eigs0r = np.linalg.eigvalsh(D0_r)
        eigs1l = np.linalg.eigvalsh(D1_l)
        eigs1r = np.linalg.eigvalsh(D1_r)
        eigs2l = np.linalg.eigvalsh(D2_l)
        eigs2r = np.linalg.eigvalsh(D2_r)
        s0l = -np.sum(eigs0l * np.log(eigs0l + 1e-15))
        s0r = -np.sum(eigs0r * np.log(eigs0r + 1e-15))
        s1l = -np.sum(eigs1l * np.log(eigs1l + 1e-15))
        s1r = -np.sum(eigs1r * np.log(eigs1r + 1e-15))
        s2l = -np.sum(eigs2l * np.log(eigs2l + 1e-15))
        s2r = -np.sum(eigs2r * np.log(eigs2r + 1e-15))
        # Tell me
        print('\n')
        print(f"Distance d = {d}")
        print(f"Entropy in subsystem for ground: {s0l}")
        print(f"Entropy in subsystem for 1st: {s1l}")
        print(f"Entropy in subsystem for 2nd: {s2l}")
        print(f"Energies: {eps[:6]}")
        # print(f"Entropy right subsystem for ground: {s0r}")
    
    for d in [20, 30, 40, 60, 70, 80, 90, 100, 110, 115, 120]:
        find_entropy_energy(d=d)
    
    
    # print(f"Entropy right subsystem for 1st: {s1r}")
    # print(f"Entropy right subsystem for 2nd: {s2r}")

    breakpoint()