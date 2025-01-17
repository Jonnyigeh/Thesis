import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
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
                l=15,
                num_func=4,
                grid_length=400,
                num_grid_points=4_001,
                a=0.25,
                alpha=1.0,
                max_iter=1_000,
                tol=1e-10,
                n_particles=2,
                verbose=True,
                params=None,
                config='I'
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
        self.num_func = num_func
        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.a = a
        self.alpha = alpha 
        self.max_iter = max_iter
        self.tol = tol
        self.n_particles = n_particles
        self.verbose = verbose
        self.params = params
        self.config = config
        if self.config == 'I':
            self.target = np.zeros(num_func)
        elif self.config == 'II':
            self.target = np.zeros(num_func)
            self.target[1] = 1.0
            self.target[2] = 1.0
        # if params is None:
        #     params = [15.0, 15.0, 9.0, 9.0, 15.0]
        # self.params = params
        # self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
        #     *self.params,
        # )

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
        new_u = np.einsum('ia, jb, ijkl, kc, ld -> abcd', c_l.conj(), c_r.conj(), basis._ulr, c_l, c_r)
        h_l = c_l.conj().T @ basis.h_l @ c_l
        h_r = c_r.conj().T @ basis.h_r @ c_r
        new_h = np.kron(h_l, np.eye(*h_l.shape)) + np.kron(np.eye(*h_r.shape), h_r)
        new_u = new_u.reshape(*new_h.shape)

        basis._h = new_h
        basis._u = new_u


    def _constraint(self, params):
        """Constraint for the optimization problem
        We need to make sure that the parameters are such that we can still fit our basis functions within the potential wells.
        """
        D_l, D_r, k_l, k_r = params
        left_constraint = 2 * D_l / np.sqrt(k_l) - np.ceil(self.l + 0.5)
        right_constraint = 2 * D_r / np.sqrt(k_r) - np.ceil(self.l + 0.5)

        return min(left_constraint, right_constraint)


    def _solve(self, params):
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
        )
        self._initialize_basis()
        num_l = self.num_func
        num_r = self.num_func
        bhs = BHS(
            h_l=self.basis.h_l,
            h_r=self.basis.h_r,
            u_lr=self.basis._ulr,
            num_basis_l=num_l,
            num_basis_r=num_r,
        )
        
        eps_l, c_l, eps_r, c_r = bhs.solve()
        self._transform_basis(self.basis, c_l, c_r)
        H = self.basis._h + self.basis._u
        eps, C = np.linalg.eigh(H)

        self.eigen_energies = eps
        return eps, C


    def _find_VN_entropies(self, rho):
        """Find entropy from reduced density matrix"""
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log(eigs + 1e-15))    
    

    def _make_density_matrix(self, C):
        self._rho = np.zeros((self.num_func ** 2, self.num_func ** 2), dtype=np.complex128)
        for n in range(self.n_particles):
            self._rho += np.outer(C[:, n], np.conj(C[:, n]).T)

    def _objective(self, params):
        eps, C = self._solve(params)
        # find reduced density matrix
        self._make_density_matrix(C)
        rho = np.trace(self._rho.reshape(self.num_func, self.num_func, self.num_func, self.num_func), axis1=0, axis2=2)
        # and then entropy
        S = self._find_VN_entropies(rho)
        
        return np.linalg.norm(S - self.target)

    def optimize(self):
        constraints = [
            {'type': 'ineq', 'fun': lambda params: self._constraint(params)}
        ]

        result = minimize(
            self._objective,
            self.params,
            method='SLSQP',
            constraints=constraints,
            options={'disp': self.verbose, 'maxiter': self.max_iter}
        )
        self.params = result.x
        return result

if __name__ == '__main__':
    D_l = 35.0
    D_r = 35.0
    k_l = 15.0
    k_r = 15.0
    params = [D_l, D_r, k_l, k_r]
    optimizer = Optimizer(params=params)
    res = optimizer.optimize()
    breakpoint()
