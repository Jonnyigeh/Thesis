import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy


# from quantum-system package
import quantum_systems as qs
print('\n')
from TDHF import TDHF_Solver
from bipartite_hartree import BipartiteHartreeSolver

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



    morse = True
    if morse:
        potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            D_a=15.0,
            D_b=15.0,
            k_a=9.0,
            k_b=9.0,
            d=15.0
        )
        basis = qs.ODMorse(l=4, grid_length=1.25 * potential.d, num_grid_points=1001, _a=0.25, alpha=1.0, potential=potential)
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
        h_l_transf = c_l.conj().T @ h_l @ c_r
        h_r_transf = c_l.conj().T @ h_r @ c_r
        u_lr_transf = c_l.conj().T @ u_lr @ c_r
        h = np.kron(h_l_transf, np.eye(*h_l_transf.shape)) + np.kron(np.eye(*h_r_transf.shape), h_r_transf)
        u = u_lr_transf.reshape(*h.shape)
        H = h + u
        eps, c = scipy.linalg.eigh(H)
        print(eps)
        breakpoint()

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
    
    ho = False
    if ho:
        potential = qs.quantum_dots.one_dim.one_dim_potentials.DWPotential(
            omega = 0.25,
            l = 5.0
        )
        # from quantum_systems.quantum_dots.one_dim.one_dim_qd import _compute_orbital_integrals, _compute_inner_integral ,_shielded_coulomb, _trapz_prep
        length = 5
        sep = 1
        n_points = 201
        grid_left = np.linspace(-length, length, n_points)
        grid_right = np.linspace(-length + sep, length + sep, n_points)
        basis1 = qs.quantum_dots.one_dim.one_dim_qd.ODHO(l = 8, grid_length=length, num_grid_points=n_points, grid=grid_left)
        basis2 = qs.quantum_dots.one_dim.one_dim_qd.ODHO(l = 8, grid_length=length, num_grid_points=n_points, grid=grid_right)
        spf_l = basis1.spf
        spf_r = basis2.spf
        grid = basis1.grid
        alpha = basis1.alpha
        _a = basis1.a
        l=basis1.l
        num_grid_points = basis1.num_grid_points
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
        print(eps_l, eps_r)
        breakpoint()
    
    # basis = qs.ODQD(l=15, grid_length=10, num_grid_points=201, a=0.25, alpha=1.0, potential=potential)
    # systema = qs.GeneralOrbitalSystem(n=2, basis_set=basis, anti_symmetrize=True)
    # anal_energy = lambda n, D, a: 2 * D / a **2 - (np.sqrt(2 * D)/a - 0.5) ** 2 
    # tdhf_solver = TDHF_Solver(system=systema, potential=potential)
    # overlap, obd, C, t = tdhf_solver.solve()
    # tdhf_solver.visualize(obd=obd, overlap=overlap, t=t / (2 * np.pi))
    