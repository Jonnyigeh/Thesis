import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib
import numba
import tqdm
import seaborn as sns
import scipy


# from quantum-system package
import quantum_systems as qs
from TDHF import TDHF_Solver
from bipartite_hartree import BipartiteHartreeSolver

@numba.njit
def _shielded_coulomb(x_1, x_2, alpha, a):
    return alpha / np.sqrt((x_1 - x_2) ** 2 + a**2)

def TIHF(h, u, num_func, n_particles, max_iters=10_000, epsilon=1e-12, verbose=True):
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

            return fock, density_matrix

        # energy, C = scipy.linalg.eigh(np.eye(h.shape[0]))#, subset_by_index=[0, num_func - 1])
        energy, C = scipy.linalg.eigh(h)#, subset_by_index=[0, num_func - 1])
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
                energy_new, C_ = scipy.linalg.eigh(fock)#, subset_by_index=[0, num_func - 1])
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
                # Update with damping
                _, density_matrix_new = fill_fock_matrix(C)
                density_matrix = alpha * density_matrix_new + (1 - alpha) * density_matrix
                fock = np.einsum('ij, aibj-> ab', density_matrix, u, dtype=np.complex128) + h



        if not converged:
            # pass
            if verbose:
                raise RuntimeError(f"The solver failed to converged after maximum number (iters={max_iters}) of iterations was reached.")
            
        return energy, C, e_list

def normalization(n, lmbda, c):
        return (
            (scipy.special.factorial(n) * (2 * lmbda - 2 * n - 1) * c / scipy.special.gamma(2 * lmbda - n))**0.5 # Gamma(n+1) = factorial(n)
            # (scipy.special.factorial(n) * (2 * lmbda - 2 * n - 1) / scipy.special.gamma(2 * lmbda - n))**0.5
        )


def morse_function(x, n, lmbda, x_e, c):
    """
    Single-well Morsepotential eigenfunction of degree n. 
    
    
    params:
    x: np.array
        Grid points
    n: int
        Degree of the Morse potential
    lmbda: float
        potential variable lambda = sqrt(2D)/a
    x_e: float
        Center of the Morse potential
    c: float
        Width of the Morse potential (not directly, but controls the width)
    """
    z = 2 * lmbda * np.exp(-c * (x - x_e))
    return (
        normalization(n, lmbda, c) *
            z**(lmbda - n - 0.5) * np.exp(-z / 2) * scipy.special.genlaguerre(n, 2 * lmbda - 2 * n - 1)(z)
    )
def reversed_morse_function(x, n, lmbda, x_e, c):
    x = x[::-1]
    x_e *= -1.0
    z = 2 * lmbda * np.exp(-c * (x - x_e))
    return (
        normalization(n, lmbda, c) *
            z**(lmbda - n - 0.5) * np.exp(-z / 2) * scipy.special.genlaguerre(n, 2 * lmbda - 2 * n - 1)(z)
    )

def single_morse(x, D_a, k_a, x_a):
    a = np.sqrt(k_a / (2 * D_a))
    return D_a * (1 - np.exp(-a * (x - x_a)))**2

def HO_potential(x, omega):
    return 0.5 * omega**2 * x**2

def HO_function(x, n, omega):
    return (
        (omega / np.pi)**0.25 * 1 / np.sqrt(2**n * scipy.special.factorial(n)) * np.exp(-omega * x**2 / 2) * scipy.special.hermite(n)(np.sqrt(omega) * x)
    )

def compute_ho_eigenenergies(omega, n):
    en = np.zeros(n)
    for i in range(n):
        en[i] = omega * (i + 0.5)
    return en


def compute_eigenenergies(c, D, l):
    # nu = c / (2 * np.pi) * np.sqrt(2 * D)
    hnu = 2 * c * np.sqrt(D / 2)
    E_n = np.zeros(l)
    for n in range(l):
        # E_n[n] = nu * (n + 0.5) - (nu * (n + 0.5))**2 / (4 * D)
        E_n[n] = hnu * (n + 0.5) - (c * hnu * (n + 0.5)**2) / np.sqrt(8 * D)

    return E_n

if __name__ == "__main__":
    single_pot_testing = False
    if single_pot_testing:
        grid = np.linspace(-2, 10, 1001)
        D = 9.0
        k = 3.0
        x_a = 0.0
        V = single_morse(grid, D, k, x_a)
        V_ = HO_potential(grid, 0.25)
        # plt.plot(grid, V)
        # plt.show()
        a = np.sqrt(k / (2 * D))
        lmbda = np.sqrt(2 * D) / a
        print('max l= ', int(lmbda - 0.5))
        l = 9
        spf = np.zeros((l, len(grid)))
        spf_ho = np.zeros((l, len(grid)))
        for i in range(l):
            spf[i] = morse_function(grid, i, lmbda, x_a, a)
            spf_ho[i] = HO_function(grid, i, 0.25)
        
        en = compute_eigenenergies(a, D, l)
        en_ho = compute_ho_eigenenergies(0.25, l)

        dx = grid[1] - grid[0]
        h_diag = 1 / (dx**2) + V[1:-1]
        h_off_diag = -1 / (2 * dx**2) * np.ones(len(grid) - 3)
        h = np.diag(h_diag) + np.diag(h_off_diag, k=1) + np.diag(h_off_diag, k=-1)
        eps, c = scipy.linalg.eigh(h, subset_by_index=[0, l-1])
        # eps, c = scipy.linalg.eigh_tridiagonal(h_diag, h_off_diag, select='i', select_range=(0, l-1))
        spf_num = np.zeros((l, len(grid)))
        spf_num[:,1:-1] = c.T / np.sqrt(dx)
        print('analytical: ', en)
        print('\n')
        print('numerical: ', eps)
        exit()
        sns.set_theme()
        fig, ax = plt.subplots()
        # ax.plot(grid, V)
        spf = np.abs(spf)**2
        spf_num = np.abs(spf_num)**2
        legend_elements = [matplotlib.lines.Line2D([0], [0], linestyle='-', color='black', lw=2, label='Analytical'),
                    matplotlib.lines.Line2D([0], [0], linestyle='--', color='black', lw=2, label='Numerical')]
        fig.legend(handles=legend_elements)

        for i in range(l):
            ax.plot(grid, spf[i], linestyle='-')
        ax.set_prop_cycle(None)
        for i in range(l):
            ax.plot(grid, spf_num[i], linestyle='--')
        ax.set_ylim(-1, 1)
        ax.set_xlim(-2,4)
        plt.show()



    morse = False
    if morse:
        potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            D_a=15.0,
            D_b=15.0,
            k_a=9.0,
            k_b=9.0,
            d=15.0
        )
        basis = qs.ODMorse(l=8, grid_length=1.25 * potential.d, num_grid_points=1001, _a=0.25, alpha=1.0, potential=potential)
        grid = basis.grid
        u_lr = basis._ulr
        h_l = basis.h_l
        h_r = basis.h_r
        num_basis_l = basis.l_sub
        num_basis_r = basis.l_sub
        bipartite_solver = BipartiteHartreeSolver(h_l, h_r, u_lr, num_basis_l, num_basis_r)
        eps_l, c_l, eps_r, c_r = bipartite_solver.solve()

        h_old = basis.h
        u_old = basis.u
        h_l_transf = c_l.conj().T @ h_l @ c_l
        h_r_transf = c_l.conj().T @ h_r @ c_l
        # u_lr_transf = c_l.conj().T @ u_lr @ c_r
        u_lr_transf = np.einsum('ai, bj, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), basis._ulr, c_l, c_r)
        h = np.kron(h_l_transf, np.eye(*h_l_transf.shape)) + np.kron(np.eye(*h_r_transf.shape), h_r_transf)
        u = u_lr_transf.reshape(*h.shape)
        H = h + u
        eps, c = scipy.linalg.eigh(H)
        print(eps)
        breakpoint()

        exit()
        lmbda = basis.a_lmbda
        a = basis.a
        b = basis.b
        d = basis.d
        D_a = basis.D_a
        D_b = basis.D_b
        D = min(-D_a, -D_b)

        spf_l_bi = np.abs(c_l @ basis.spf_l) ** 2
        spf_r_bi = np.abs(c_r @ basis.spf_r) ** 2
        # sns.set_theme()
        # fig, ax = plt.subplots()
        # ax.set_xlim(-11, 11)
        # ax.plot(grid, potential(grid) - potential(0), color='y')
        # for p in range(num_basis_r):
        #     # ax.plot(grid, np.abs(morse_function(grid, p, lmbda, -d/2, a))**2)
        #     # ax.plot(grid, np.abs(reversed_morse_function(grid, p, lmbda, d/2, b))**2)
        #     ax.plot(grid, spf_a[p])
        #     ax.plot(grid, spf_b[p])
        # minima = int(np.min(potential(grid) - potential(0)))
        # ax.set_ylim(minima, 5.0)
        # plt.show()
        # plt.show(block=False)
        # plt.pause(4)
        # plt.close()
        # breakpoint()

        # Set up the eigenvector matrix in the bipartite system
        spf_l = basis.spf_l
        spf_r = basis.spf_r
        spf_l_transf = c_l @ basis.spf_l
        spf_r_transf = c_r @ basis.spf_r
        
        density_matrix_l = np.zeros(c_l.shape, dtype=np.complex128)
        density_matrix_r = np.zeros(c_r.shape, dtype=np.complex128)
        for i in range(num_basis_l):
            density_matrix_l += np.outer(c_l[:,i], np.conj(c_l[:,i]).T)
            density_matrix_r += np.outer(c_r[:,i], np.conj(c_r[:,i]).T)
        # Check that the density matrices are normalized
        print(np.trace(density_matrix_l/4), np.trace(density_matrix_r/4))
        

        # # Compare eigenfunctions
        # gs = matplotlib.gridspec.GridSpec(3, 2)
        # fig = plt.figure(layout='tight')
        # ax00 = fig.add_subplot(gs[0,0])
        # ax01 = fig.add_subplot(gs[0,1])
        # ax10 = fig.add_subplot(gs[1,0])
        # ax11 = fig.add_subplot(gs[1,1])
        # comp1 = fig.add_subplot(gs[2,0])
        # comp2 = fig.add_subplot(gs[2,1])
        # comp1.plot(grid, (potential(grid) - potential(0)) / np.max(-potential(grid) + potential(0)))
        # comp2.plot(grid, (potential(grid) - potential(0)) / np.max(-potential(grid) + potential(0)))
        # comp2.set_ylim(-1, 1)
        # comp1.set_ylim(-1, 1)
        # for i in range(num_basis_l):
        #     ax00.plot(grid, np.abs(spf_l[i])**2)
        #     ax10.plot(grid, np.abs(spf_l_transf[i])**2, linestyle='--')
        #     ax01.plot(grid, np.abs(spf_r[i])**2)
        #     ax11.plot(grid, np.abs(spf_r_transf[i])**2, linestyle='--')
        #     comp1.plot(grid, np.abs(spf_l[i])**2)
        #     comp1.plot(grid, np.abs(spf_l_transf[i])**2, linestyle='--')
        #     comp2.plot(grid, np.abs(spf_r[i])**2)
        #     comp2.plot(grid, np.abs(spf_r_transf[i])**2, linestyle='--')
        # ax00.set_title('Left well')
        # ax10.set_title('Left well (bipartite)')
        # ax01.set_title('Right well')
        # ax11.set_title('Right well (bipartite)')
        # comp1.set_title('Left well comparison')
        # comp2.set_title('Right well comparison')
        # plt.show()
        # # breakpoint()
        # # exit()

        # # One-body density
        # obd = np.zeros(basis.num_grid_points)
        # obd_l = np.einsum('mi, mn, ni->i', spf_l_transf, density_matrix_l, spf_l_transf)
        # obd_r = np.einsum('mi, mn, ni->i', spf_r_transf, density_matrix_r, spf_r_transf)
        # fig, ax = plt.subplots()
        # ax.plot(grid, obd_l, label='left well')
        # ax.plot(grid, obd_r, label='right well')
        # ax.plot(grid, potential(grid) - potential(0))
        # ax.set_ylim(-5, 2)
        # ax.legend()
        # plt.show()
        # Make a distinguishable CI calculation
        breakpoint()

        system = qs.GeneralOrbitalSystem(n=2, basis_set=basis, anti_symmetrize=True)
        exit()
        # solver = TDHF_Solver(system=system, potential=potential)
        # overlap, obd, C, t = solver.solve()
        # for i in range(c_l.shape[1]):
        #     print("left: ", np.abs(c_l[:,i])**2, 'right: ', np.abs(c_r[:,i])**2)
        #     print('\n')
        # import pandas as pd
        # df = pd.DataFrame(basis.spf_unified)
        # for col in df.columns:
        #     if np.sum(df[col]) > 0.1:
        #         breakpoint()
        # Calculate overlap between left and right well



    # TODO: Finn ut relasjon mellom D, c og n-gridpoints for å få optimal struktur. Må finne en form som gir god mixing.
    
    ho = True
    if ho:
        # potential = qs.quantum_dots.one_dim.one_dim_potentials.DWPotential(
        #     omega = 0.25,
        #     l = 5.0
        # )
        # from quantum_systems.quantum_dots.one_dim.one_dim_qd import _compute_orbital_integrals, _compute_inner_integral ,_shielded_coulomb, _trapz_prep
        length = 200
        num_grid_points = 10_001
        grid = np.linspace(-length, length, num_grid_points)
        seps = [3, 5, 10, 15, 20, 30, 50, 200]
        l = 5
        _a = 0.25
        alpha = 0.1#1.0
        for sep in seps:
            x_l = -int(sep/2)
            x_r = int(sep/2)
            basis1 = qs.quantum_dots.one_dim.one_dim_qd.ODHO(l=l, grid_length=length, num_grid_points=num_grid_points, grid=grid, x0=x_l, a=_a, alpha=alpha)
            basis2 = qs.quantum_dots.one_dim.one_dim_qd.ODHO(l=l, grid_length=length, num_grid_points=num_grid_points, grid=grid, x0=x_r, a=_a, alpha=alpha)
            spf_l = basis1.spf
            spf_r = basis2.spf

            spf_unified = np.zeros((l * l, num_grid_points), dtype=np.complex128)
            idx = 0
            for i in range(l):
                for j in range(l):
                    spf_unified[idx, :] = spf_l[i,:] * spf_r[j,:]
                    idx += 1
            
            inner_integral = qs.quantum_dots.one_dim.one_dim_qd._compute_inner_integral(
            spf_unified,
            l,
            num_grid_points,
            grid,
            alpha,
            _a
            )   

            u = qs.quantum_dots.one_dim.one_dim_qd._compute_orbital_integrals(
                spf_unified,
                l,
                inner_integral,
                grid
            )  
            h_l = basis1.h
            h_r = basis2.h
            u_lr = u
            num_basis_l = l
            num_basis_r = l

            bipartite_solver = BipartiteHartreeSolver(h_l, h_r, u_lr, num_basis_l, num_basis_r)
            eps_l, c_l, eps_r, c_r = bipartite_solver.solve()
            h_l_transf = c_l.conj().T @ h_l @ c_l
            h_r_transf = c_r.conj().T @ h_r @ c_r
            u_lr_transf = np.einsum('ia, jb, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), u_lr, c_l, c_r)
            h = np.kron(h_l_transf, np.eye(*h_l_transf.shape)) + np.kron(np.eye(*h_r_transf.shape), h_r_transf)
            ulr = u_lr_transf.reshape(*h.shape)
            eps, C = np.linalg.eigh(h + ulr)
            # print("distinguishable")
            # print(eps_l, eps_r)
            print(eps)

        # Fermionic system
        print("indistinguishable")
        l = l ** 2
        n_particles = 2
        # cap = 10
        # V_func = lambda x, cap: np.clip(0.5 * 0.25 **2 * (x - int(sep/2))**2,0,cap) + np.clip(0.5 * 0.25 ** 2 * (x + int(sep/2)) ** 2,0,cap)
        # V = V_func(grid[1:-1], cap)
        for sep in seps:
            V_l = 0.5 * 0.25 **2 * (grid[1:-1] + int(sep/2))**2
            V_r = 0.5 * 0.25 ** 2 * (grid[1:-1] - int(sep/2))**2
            V = np.minimum(V_l, V_r) # combine the two potentials
            # mid_region = (grid[1:-1] > -int(sep/4)) & (grid[1:-1] < int(sep/4))         # Define the region where the potentials overlap
            # V[mid_region] = 1_000_000      # Set the overlapping region to infinity
            dx = grid[1] - grid[0]
            h_diag = 1 / (dx**2) + V
            h_off_diag = - 1 / (2 * dx**2) * np.ones(num_grid_points - 3)
            eps, C = scipy.linalg.eigh_tridiagonal(h_diag, h_off_diag, select="i", select_range=(0, l - 1))
            spf = np.zeros((l, num_grid_points), dtype=np.complex128)
            spf[:, 1:-1] = C.T / np.sqrt(dx)
            eigen_energies = eps
            h = np.diag(eps)
            
            # coulomb = np.zeros((num_grid_points, num_grid_points), dtype=np.complex128)
            # for i in range(num_grid_points):
            #     coulomb[i] = _shielded_coulomb(grid[i], grid, alpha, _a)
            # ulr_ind = np.einsum('ix, jy, xy, kx, ly -> ijkl', spf.conj(), spf.conj(), coulomb, spf, spf, optimize=True) - np.einsum('ix, jy, xy, ky, lx -> ijkl', spf.conj(), spf.conj(), coulomb, spf, spf, optimize=True)
            coulomb = _shielded_coulomb(
                grid[None, 1:-1], grid[1:-1, None], alpha, _a
            )
            ulr_ind = np.einsum(
                "pa, qb, pc, qd, pq -> abcd", C, C, C, C, coulomb, optimize=True
            )
            # plt.plot(grid[1:-1], np.clip(V,0,0.5), grid, np.abs(spf[0])**2, grid, np.abs(spf[1])**2, grid, np.abs(spf[2])**2)
            # plt.show()
            n_particles=2
            ind_eps, ind_C, e_list = TIHF(h, ulr_ind, num_func=l, n_particles=n_particles, verbose=False)
            rho = np.zeros((l,l), dtype=np.complex128)
            for i in range(n_particles):
                rho += np.outer(ind_C[:,i], np.conj(ind_C[:,i]).T)
            # Find total HF energy
            E_2body = 0
            for i in range(l):
                for j in range(l):
                    E_2body += 0.5 * rho[i, i] * rho[j, j] * (ulr_ind[i, j, i, j] - ulr_ind[i, j, j, i])
            E_1body = np.sum(np.einsum('ij,ij->ij', h, rho))
            E = E_1body + E_2body

            print(ind_eps[:6])
            print(E)
        breakpoint()
    
    # basis = qs.ODQD(l=15, grid_length=10, num_grid_points=201, a=0.25, alpha=1.0, potential=potential)
    # systema = qs.GeneralOrbitalSystem(n=2, basis_set=basis, anti_symmetrize=True)
    # anal_energy = lambda n, D, a: 2 * D / a **2 - (np.sqrt(2 * D)/a - 0.5) ** 2 
    # tdhf_solver = TDHF_Solver(system=systema, potential=potential)
    # overlap, obd, C, t = tdhf_solver.solve()
    # tdhf_solver.visualize(obd=obd, overlap=overlap, t=t / (2 * np.pi))
    