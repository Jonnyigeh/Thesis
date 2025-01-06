import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Local imports
import quantum_systems as qs
from bipartite_hartree import BipartiteHartreeSolver as BHS


class Optimizer:
    """
    Optimization of parameters for the Morse double well potential. 
    Loss function is either Von Neumann entropy or the energy of the system through a $\zeta$-parameter from Zhao article.
    """
        
    def __init__(self,
                l=10,
                grid_length=15,
                num_grid_points=2001,
                a=0.25,
                alpha=1.0,
                max_iter=1000,
                tol=1e-10,
                verbose=True,
                params=None
    ):
        """
        Parameters:
        l : int
            Number of basis functions
        grid_length : float
            Length of the grid
        num_grid_points : int
            Number of grid points
        a : float
            Shielding paramter in Coulomb interaction
        alpha : float
            Scaling parameter in Coulomb interaction
        max_iter : int
            Maximum number of iterations in optimization
        tol : float
            Convergence criteria for optimization
        verbose : bool
            Print information during optimization
        """
        self.l = l
        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.a = a
        self.alpha = alpha 
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        if params is None:
            params = [15.0, 15.0, 9.0, 9.0, 15.0]
        self.params = params
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *self.params,
        )
        

    def _initialize_basis(self):
        self.basis = qs.ODMorse(
            l=self.l,
            grid_length=self.grid_length,
            num_grid_points=self.num_grid_points,
            _a=self.a,
            alpha=self.alpha,
            potential=self.potential
        )
        self.grid = self.basis.grid


    def _transform_basis(self, basis, c_l, c_r):
        shape = c_l.shape[0]
        new_u = np.einsum('ai, bj, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), basis.u.reshape(shape,shape,shape,shape), c_l, c_r)
        h_l = c_l.conj().T @ basis.h_l @ c_l
        h_r = c_r.conj().T @ basis.h_r @ c_r
        new_h = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)

        basis.h = new_h
        basis.u = new_u

    def _find_overlap_matrix(self, old_basis, new_basis):
        S = np.zeros((old_basis.num_basis, new_basis.num_basis), dtype=np.complex128)
        for i in range(old_basis.num_basis):
            for j in range(new_basis.num_basis):
                # S[i,j] = np.sum(old_basis.spf[:,i] * new_basis.spf[:,j])
                S[i,j] = np.vdot(old_basis.spf[:,i], new_basis.spf[:,j])
        
        return S

    def _transform_coefficients(self, c, S):
        return c @ S
   
    def _find_VN_entropies(self, rho=None):
        if rho is None:
            rho = self._rho
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log(eigs + 1e-15))    
        
    
    # @property
    # def entropy(self):
    #     return self._entropy
    

    # def _make_density_matrix(self, C):
    #     self._rho = np.outer(C, C.conj().T)
    

    @property
    def rho(self):
        return self._rho


    def _constraint(self, params, l):
        """Constraint for the optimization problem
        We need to make sure that the parameters are such that we can still fit our basis functions within the potential wells.
        """
        D_l, D_r, k_l, k_r = params
        left_constraint = 2 * D_l / np.sqrt(k_l) - np.ceil(l + 0.5)
        right_constraint = 2 * D_r / np.sqrt(k_r) - np.ceil(l + 0.5)

        return min(left_constraint, right_constraint)


    def _solve(self):
        num_l = 2
        num_r = 2
        bhs = BHS(
            h_l=self.basis.h_l,
            h_r=self.basis.h_r,
            u_lr=self.basis._ulr,
            num_basis_l=num_l,
            num_basis_r=num_r,
        )
        eps_l, c_l, eps_r, c_r = bhs.solve()
        self._transform_basis(self.basis, c_l, c_r)
        H = self.basis.h + self.basis.u
        eps, C = np.linalg.eigh(H)

        self.eigen_energies = eps
        self.coefficients = C

    def _update_basis(self, params):
        new_potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params
        )
        self.basis.potential = new_potential


    def _loss(self):
        # Make density matrix and calculate entropies
        self._make_density_matrix(self.coefficients)
        self._find_VN_entropies()
        n_states = len(self.eigen_energies)
        entropies = np.zeros(n_states)
        for i in range(n_states):
            rho = np.outer(self.coefficients[:,i], self.coefficients[:,i].conj().T)
            entropies[i] = self._find_VN_entropies(rho)

    
    def _objective(self, params):
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
        )
        self._update_basis()
        self._solve()

        return self._loss()


    def _run_optimization(self):
        # Initial calculations
        self._initialize_basis()