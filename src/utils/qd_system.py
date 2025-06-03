import numba
import numpy as np
import scipy.special
import scipy.linalg
import time


from utils.potential import (
    MorsePotentialDW
)

@numba.njit
def _shielded_coulomb(x_1, x_2, alpha, a):
    return alpha / np.sqrt((x_1 - x_2) ** 2 + a**2)



class ODMorse():
    """
    something something
    Analytical expressions are from the book "Ideas of Quantum Chemistry" by Lucjan Piela, 2014
    Equations are from the article "The Morse oscillator in position space, momentum space, and phase space" by Dahl and Springborg,
    in doi:10.1063/1.453761
    """

    MorsePotentialDW = MorsePotentialDW

    def __init__(
            self,
            l,
            grid_length,
            num_grid_points,
            _a=0.25,
            alpha=1.0,
            D_a=10.0,
            D_b=10.0,
            k_a=1.0,
            k_b=1.0,
            d=15.0,
            anti_symmetric=False,
            potential=None,
            visualize=False,
            verbose=False,
            dvr=False,
    ):
        if potential is None:
            potential = MorsePotentialDW(D_a=D_a, D_b=D_b, k_a=k_a, k_b=k_b, d=d)
        self._potential = potential
        self.anti_symmetric = anti_symmetric
        self.dvr = dvr
        self.D_a = potential.D_a
        self.D_b = potential.D_b
        self._a = _a
        self.alpha = alpha
        self.k_a = k_a
        self.k_b = k_b
        self.a = potential.a
        self.b = potential.b
        self.d = potential.d
        self.a_lmbda = np.sqrt(2 * self.D_a) / self.a
        self.b_lmbda = np.sqrt(2 * self.D_b) / self.b
        # Set total number of basis functions for the composite system, i.e l ^ 2, since we are using product states.
        self.l_sub = l
        self.l = l ** 2
        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.grid = np.linspace(
            -self.grid_length/2, self.grid_length/2, self.num_grid_points
        )
        ## Visualization, and comparison of analytical and numerical basis functions
        # self.setup_analytical_basis()
        if self.anti_symmetric:
            self.setup_fermionic_basis()
        elif self.dvr:
            self.setup_DVR_basis()
        else:
            self.setup_distinguishable_basis()
        if visualize:
            import matplotlib.pyplot as plt
            self.a_eigen_energies = self.compute_eigenenergies(self.a, self.D_a, self.l_sub)
            self.b_eigen_energies = self.compute_eigenenergies(self.b, self.D_b, self.l_sub)
            fig, ax = plt.subplots(2, 2)
            compare_ax = fig.add_subplot(212)
            for f in range(self.spf_a.shape[0]):
                ax[0,0].plot(self.grid, np.abs(self.spf_a[f]) ** 2, linestyle='-')
                compare_ax.plot(self.grid, np.abs(self.spf_a[f]) ** 2, linestyle='-')
            compare_ax.set_prop_cycle(None)
            for f in range(self.spf_l.shape[0]):
                ax[0,1].plot(self.grid, np.abs(self.spf_l[f]) ** 2, linestyle='--')
                compare_ax.plot(self.grid, np.abs(self.spf_l[f]) ** 2, linestyle='--' )
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], linestyle='-', color='black', lw=2, label='Analytical'),
                    Line2D([0], [0], linestyle='--', color='black', lw=2, label='Numerical')]
            fig.legend(handles=legend_elements)
            plt.show()
            print("Analytical energies: left well: ", self.a_eigen_energies, "right well: ", self.b_eigen_energies)
            print('\n')
            print("numerical energies: left well: ", self.eigen_energies_l, "right well: ", self.eigen_energies_r)


    def setup_fermionic_basis(self):
        dx = self.grid[1] - self.grid[0]
        V = np.clip(self._potential(self.grid[1:-1]), 0, 100)
        V -= np.min(V)
        h_diag = 1 / (dx**2) + V
        h_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        t1 = time.time()
        print("Start diagonalization of the Hamiltonian..")
        eps, C = scipy.linalg.eigh_tridiagonal(h_diag, h_off_diag, select="i", select_range=(0, self.l - 1))
        t2 = time.time()
        print("Diagonalization completed in ", t2 - t1, " seconds.")
        self.spf = np.zeros((self.l, self.num_grid_points), dtype=np.complex128)
        self.spf[:, 1:-1] = C.T / np.sqrt(dx)
        self.eigen_energies = eps
        self.h = np.diag(eps)
        self.s = np.eye(self.l)
        coulomb = np.zeros((self.num_grid_points, self.num_grid_points), dtype=np.complex128)
        for i in range(self.num_grid_points):
            coulomb[i] = _shielded_coulomb(self.grid[i], self.grid, self.alpha, self._a)
        self.u = np.einsum('ix, jy, xy, kx, ly -> ijkl', self.spf.conj(), self.spf.conj(), coulomb, self.spf, self.spf, optimize=True) - np.einsum('ix, jy, xy, ky, lx -> ijkl', self.spf.conj(), self.spf.conj(), coulomb, self.spf, self.spf, optimize=True)

    def setup_distinguishable_basis(self):
        dx = self.grid[1] - self.grid[0]
        # Find the eigenbasis for each well separately, left = A, right = B - accounting for Dirichlet BC, i.e the WF should go to zero at the end-points
        V_l = np.clip(self._potential.left_pot(self.grid[1:-1]), 0, 100) # clip potential at 100, more natural, also prevents bugs with infinite potential wall.
        V_r = np.clip(self._potential.right_pot(self.grid[1:-1]), 0, 100)
        self.V = V_l + V_r - np.min(V_l + V_r)
        h_l_diag = 1 / (dx**2) + V_l
        h_l_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        h_r_diag = 1 / (dx**2) + V_r
        h_r_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        eps_l, C_l = scipy.linalg.eigh_tridiagonal(h_l_diag, h_l_off_diag, select="i", select_range=(0, self.l_sub - 1))
        eps_r, C_r = scipy.linalg.eigh_tridiagonal(h_r_diag, h_r_off_diag, select="i", select_range=(0, self.l_sub - 1))
        # Enforce single-particle functions goes to zero on boundary, and that they are normalized (on a discrete grid)
        self.spf_l = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        self.spf_l[:, 1:-1] = C_l.T / np.sqrt(dx)
        self.spf_l /= np.linalg.norm(self.spf_l, axis=1)[:, None]
        self.eigen_energies_l = eps_l
        self.spf_r = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        self.spf_r[:, 1:-1] = C_r.T / np.sqrt(dx)
        self.spf_r /= np.linalg.norm(self.spf_r, axis=1)[:, None]
        self.eigen_energies_r = eps_r
        # Set up the Hamiltonian matrices for each well
        self._h_l = np.diag(eps_l)
        self._h_r = np.diag(eps_r)
        
        self._ulr = self.find_u_matrix(grid=self.grid, alpha=self.alpha, a=self._a, spf_l=self.spf_l, spf_r=self.spf_r)
        self._h = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        self._u = self._ulr.reshape(*self._h.shape)
        self.s = np.eye(self.l)

    def setup_DVR_basis(self):
        self.dx = self.grid[1] - self.grid[0]
        self.left_grid = self.grid[:self.num_grid_points // 2]
        self.right_grid = self.grid[self.num_grid_points // 2:]
        self.basis_func_left = np.zeros((self.num_grid_points // 2, self.num_grid_points // 2))
        self.basis_func_right = np.zeros((self.num_grid_points // 2, self.num_grid_points // 2))
        for i in range(self.num_grid_points // 2):
            self.basis_func_left[i] = self.sinc_dvr_func(self.left_grid, self.left_grid[i], self.dx)
            self.basis_func_right[i] = self.sinc_dvr_func(self.right_grid, self.right_grid[i], self.dx)

        V_l = np.clip(self._potential.left_pot(self.left_grid), 0, 100) # clip potential at 100, more natural, also prevents bugs with infinite potential wall.
        V_r = np.clip(self._potential.right_pot(self.right_grid), 0, 100)
        h_l_diag = np.pi ** 2 / (6 * self.dx ** 2) * np.ones(self.num_grid_points // 2)
        h_r_diag = np.pi ** 2 / (6 * self.dx ** 2) * np.ones(self.num_grid_points // 2)
        h_l = np.zeros((self.num_grid_points // 2, self.num_grid_points // 2))
        h_r = np.zeros((self.num_grid_points // 2, self.num_grid_points // 2))
        for i in range(self.num_grid_points // 2):
            for j in range(self.num_grid_points // 2):
                if i == j:
                    continue
                h_l[i,j] = self.sinc_dvr_kinetic(i, j, self.dx) 
                h_r[i,j] = self.sinc_dvr_kinetic(i, j, self.dx)
        self.no_1bpot_h_l = h_l + np.diag(h_l_diag) / self.dx
        self.no_1bpot_h_r = h_r + np.diag(h_r_diag) / self.dx
        h_l += (np.diag(h_l_diag) + np.diag(V_l)) / self.dx
        h_r += (np.diag(h_r_diag) + np.diag(V_r)) / self.dx
        u = self.u_dvr()

        self._h_l = h_l
        self._h_r = h_r
        self._ulr = u

                

    def u_dvr(self):
        coulomb = np.zeros((self.num_grid_points//2, self.num_grid_points//2), dtype=np.complex128)
        for i in range(self.num_grid_points // 2):
            for j in range(self.num_grid_points // 2):
                coulomb[i, j] = _shielded_coulomb(self.left_grid[i], self.right_grid[j], self.alpha, self.a)
        
        u = np.einsum('ik, kl, jl -> ij', self.basis_func_left / np.sqrt(self.dx), coulomb, self.basis_func_right / np.sqrt(self.dx), optimize=True)
        
        return u
    
    def sinc_dvr_func(self, x, xi, dx):
        scaled_x = (x - xi) / dx
        func = 1 / dx * np.sinc(scaled_x)
        return func

    def sinc_dvr_kinetic(self, i, j, dx):
        assert i != j, "i and j must be different"
        return (-1) ** (i - j) / (dx ** 2 * (i - j) ** 2)


    def setup_analytical_basis(self):
        raise NotImplementedError
        # self.a_eigen_energies = self.compute_eigenenergies(self.a, self.D_a, self.l_sub)
        # self.b_eigen_energies = self.compute_eigenenergies(self.b, self.D_b, self.l_sub)

        # # Construct the Hamiltonian matrices for the two wells
        # self.h_a = np.diag(self.a_eigen_energies).astype(np.complex128)
        # self.h_b = np.diag(self.b_eigen_energies).astype(np.complex128)
        # self.s = np.eye(self.l_sub) # The eigenstates are orthogonal, so the overlap matrix is the identity matrix

        # # Allocate memory for the single-particle functions for each well
        # self.spf_a = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        # self.spf_b = np.zeros((self.l_sub, self.num_grid_points), dtype=np.complex128)
        # # Evaluate the single-particle functions on the grid, and store them in the spf matrices
        # for p in range(self.l_sub):
        #     self.spf_a[p] = self.morse_function(self.grid, p, self.a_lmbda, -self.d / 2, self.a)
        #     self.spf_b[p] = self.morse_function(self.grid, p, self.b_lmbda, self.d / 2, self.b, reversed=True)

        # # Find the unified basis \psi_{lr}(x) = \psi_r(x) * \psi_l(x) by element-wise multiplication
        # self.spf_unified = np.zeros((self.l_sub * self.l_sub, self.num_grid_points), dtype=np.complex128)
        # idx = 0
        # for i in range(self.l_sub):
        #     for j in range(self.l_sub):
        #         self.spf_unified[idx, :] = self.spf_a[i,:] * self.spf_b[j,:]
        #         idx += 1
        
        # # Find the matrix elements in two steps: first the inner integral <p|V|q> for each pair of states p and q
        # # then the outer integral <pq|V|rs> for each pair of states p, q, r and s. Do this for each well. Or together?
        # inner_integral = _compute_inner_integral(
        #     self.spf_unified,
        #     self.l,
        #     self.num_grid_points,
        #     self.grid,
        #     self.alpha,
        #     self._a
        # )
        
        # self.u = _compute_orbital_integrals(
        #     self.spf_unified,
        #     self.l,
        #     inner_integral,
        #     self.grid
        # )        
        
    def morse_function(self, x, n, lmbda, x_e, c, reversed=False):
        """
        Single-well Morsepotential eigenfunction of degree n. Analytical expressions from "Ideas of Quantum Chemistry" by Lucjan Piela.
        
        
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
        if reversed:
            x = x[::-1]
            x_e *= -1.0
        z = 2 * lmbda * np.exp(-c * (x - x_e))
        return (
            self.normalization(n, lmbda, c) *
             z**(lmbda - n - 0.5) * np.exp(-z / 2) * scipy.special.genlaguerre(n, 2 * lmbda - 2 * n - 1)(z)
        )
    
    def normalization(self, n, lmbda, c):
        return (
            (scipy.special.factorial(n) * (2 * lmbda - 2 * n - 1) * c / scipy.special.gamma(2 * lmbda - n))**0.5 # Gamma(n+1) = factorial(n)
        )

    def compute_eigenenergies(self, c, D, l):
        hnu = 2 * c * np.sqrt(D / 2)
        E_n = np.zeros(l)
        for n in range(l):
            E_n[n] = hnu * (n + 0.5) - (c * hnu * (n + 0.5)**2) / np.sqrt(8 * D)

        return E_n

    def find_u_matrix(self, grid, alpha, a, spf_l, spf_r)->np.ndarray:
        """
        Calculate the interaction matrix elements of shielded Coulomb (1D) by einstein-summation. A bit slower, and less accurate than the trapezoidal numeric integration
        but is more easily done for a bipartite system when this is required. Shielded coulomb is alpha / |x1 - x2 + a|

        args:
        grid: np.array
            Grid (in position space) where the interactions should be computed
        alpha: float
            Scaling parameter in coulomb
        a: float
            Shielding parameter in coulomb
        spf_l: np.array
            Single-particle functions in the left well.
        spf_r: np.array
            Single-particle functions in the right well.
        """
        l = spf_l.shape[0] # number of basis functions
        num_grid = spf_l.shape[1] # number of grid points
        u = np.zeros((l,l,l,l), dtype=np.complex128)
        # Find the Coulomb-interactions on the grid
        coulomb = np.zeros((num_grid, num_grid), dtype=np.complex128)
        for i in range(num_grid):
            coulomb[i] = _shielded_coulomb(grid[i], grid, alpha, a)
        
        # # Make the integration (a sum over indices)
        u = np.einsum('ix, jy, xy, kx, ly -> ijkl', spf_l.conj(), spf_r.conj(), coulomb, spf_l, spf_r, optimize=True)
        
        return u


    def construct_position_integrals(self, lmbda, spf, l):
        """
        Analytical expressions for the position integrals in the Morse potential basis.
        Taken from doi:10.1088/0953-4075/38/7/004
        """
        position = np.zeros((1, self.l, self.l), dtype=spf.dtype)
        N = lmbda - 0.5
        # Loop through all n < m, to find elements <n|x|m>
        for n in range(l - 1):
            for m in range(n+1, l):
                pre_factor = 2 * (-1)**(m-n+1) /((m-n) * (2 * N -m))
                position[0, n, m] = (
                    pre_factor * np.sqrt((N-n) * (N-m) * scipy.special.gamma(2 * N - m + 1) * scipy.special.factorial(m) / 
                    (scipy.special.gamma(2 * N - n + 1) * scipy.special.factorial(n)))
                )
        # Symmetrize the position matrix with the hermitian conjugate, i.e adding <m|x|n> (since <n|x|m> = <m|x|n>^H)
        position[0, :, :] += position[0, :, :].conj().T
        # Diagonal elements <n|x|n>
        for n in range(self.l):
            position[0, n, n] = (
                np.log(2 * N + 1) + scipy.special.psi(2 * N - n + 1) - scipy.special.psi(2 * N - 2 * n + 1) - scipy.special.psi(2 * N - 2 * n)
            )

        return position

    @property
    def potential(self):
        return self._potential
    
    @potential.getter # Superfluous, but this is the getter for the potential
    def potential(self):
        return self._potential
    
    @potential.setter
    def potential(self, potential):
        self._potential = potential
        self.a = potential.a
        self.b = potential.b
        self.d = potential.d
        print(f"New potential set, reinitializing basis..")
        self.setup_basis() # Reinitialize the basis when the potential is changed
        print(f"New basis initialized.")

    @property
    def h(self):
        return self._h

    @property
    def u(self):
        return self._u

    @property
    def h_l(self):
        return self._h_l
    
    @property
    def h_r(self):
        return self._h_r

    @h.setter
    def h(self, h):
        self._h = h


    @u.setter
    def u(self, u):
        self._u = u
    
    @h_l.setter
    def h_l(self, h_l):
        self._h_l = h_l
    
    @h_r.setter
    def h_r(self, h_r):
        self._h_r = h_r