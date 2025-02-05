import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import svdvals

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
                d=100,
                a=0.25,
                alpha=1.0,
                max_iter=2_000,
                tol=1e-4,
                n_particles=2,
                scaling=1e-1,
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
        # self.d = d
        self.alpha = alpha 
        self.max_iter = max_iter
        self.tol = tol
        self.n_particles = n_particles
        self.verbose = verbose
        self.scaling = scaling
        self.params = np.asarray(params)
        self.config = config
        if self.config == 'I':
            self.target = np.zeros(num_func)
        elif self.config == 'II':
            self.target = np.zeros(num_func)
            self.target[0] = 0.0
            self.target[1] = 1.0
            self.target[2] = 1.0
            self.target[3] = 0.0
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
        try:
            D_l, D_r, k_l, k_r, d = params
        except:
            D_l, D_r, k_l, k_r = params
        left_constraint = 2 * D_l / np.sqrt(k_l) - np.ceil(self.l + 0.5)
        right_constraint = 2 * D_r / np.sqrt(k_r) - np.ceil(self.l + 0.5)

        return min(left_constraint, right_constraint)
    

    def _solve(self, params):
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
            # d=self.d
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
        self.eps_l = eps_l - eps_l[0]
        self.eps_r = eps_r - eps_r[0]
        self.trans_energies = {
            'w_l1': float(round(self.eps_l[1] / (2 * np.pi), 5)),
            'w_r1': float(round(self.eps_r[1] / (2 * np.pi), 5)),
            'w_L1 + w_R1': float(round((self.eps_l[1] + self.eps_r[1]) / (2 * np.pi), 5)),
            'w_l2': float(round(self.eps_l[2] / (2 * np.pi), 5)),
            'w_r2': float(round(self.eps_r[2] / (2 * np.pi), 5)),
        }
        self._transform_basis(self.basis, c_l, c_r)
        H = self.basis._h + self.basis._u
        eps, C = np.linalg.eigh(H)
        self.eigen_energies = eps
        
        return eps, C


    def _find_VN_entropies(self, rho):
        """Find entropy from reduced density matrix"""
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log2(eigs + 1e-15))    
    
    def _find_svd_entropies(self, C):
        """Find entropy from SVD of coefficient matrix"""
        entropies = np.zeros(self.num_func)
        for i in range(self.num_func):
            vals = (svdvals(C[:,i].reshape(self.num_func,self.num_func))) **2
            entropies[i] = - np.sum(vals * np.log2(vals + 1e-15))
        
        return entropies

    

    def _make_density_matrix(self, C):
        # self._rho = np.zeros((self.num_func ** 2, self.num_func ** 2), dtype=np.complex128)
        # for n in range(self.n_particles):
        #     self._rho += np.outer(C[n], np.conj(C[n]).T)
        self._rho = np.outer(C, np.conj(C).T)


    def _objective(self, params):
        eps, C = self._solve(params / self.scaling)
        # find reduced density matrix
        self.S = np.zeros(self.num_func)
        for i in range(self.num_func):
            self._make_density_matrix(C[:,i]) # Each column is the energy eigenstates
            rho = np.trace(self._rho.reshape(self.num_func, self.num_func, self.num_func, self.num_func), axis1=0, axis2=2)
            # and then entropy
            self.S[i] = self._find_VN_entropies(rho)
        
        self.svd_entropy = self._find_svd_entropies(C)
        energy_penalty = 0
        detuning_penalty = 0
        self.ZZ = np.abs(eps[4] - eps[2] - eps[1] + eps[0])
        if self.config == 'I':
            detuning_penalty = -min(0.5, np.abs(self.eps_l[1] - self.eps_r[1]))
        
        if self.config == 'II':
            detuning_penalty = np.abs(self.eps_l[1] - self.eps_r[1])
        
        return np.linalg.norm(self.S - self.target) + self.ZZ + detuning_penalty


    def optimize(self):
        def _callback(params):
            """Callback function for optimization"""
            if self.counter % 100 == 0:
                print(f"Iteration: {self.counter}")
                print(f"Parameters: {params / self.scaling}")
                print(f"Objective: {self._objective(params)}")
                print(f"ZZ param: {self.ZZ}")
                print(f"Energies: {self.eigen_energies[:6] / (2 * np.pi)}")
                print(f"Transition energies: {self.trans_energies}")
                print(f"Entropy: {self.S}")
            self.counter += 1
            self.entropies.append(self.S[0])
        constraints = [
            {'type': 'ineq', 'fun': lambda params: self._constraint(params / self.scaling)}
        ]
        # bounds = [(15, 100),  # D_l must be positive
        #   (15, 100),  # D_r must be positive
        #   (5, 75),  # k_l must be positive
        #   (5, 75)]  # k_r must be positive
        bounds = [(10 * self.scaling, 100 * self.scaling),  # D_l must be positive
            (10 * self.scaling, 100 * self.scaling),  # D_r must be positive
            (5 * self.scaling, 100 * self.scaling),  # k_l must be positive
            (5 * self.scaling, 100 * self.scaling),
            (90 * self.scaling, 300 * self.scaling)]  # k_r must be positive
        self.counter = 0
        self.entropies = []
        result = minimize(
            self._objective,
            self.params * self.scaling,
            method='COBYQA',
            bounds=bounds,
            constraints=constraints,
            callback=_callback,
            tol=self.tol,
            options={'disp': self.verbose, 'maxiter': self.max_iter, 'final_tr_radius': self.tol}
        ) 
        self.params = result.x / self.scaling
        return result

if __name__ == '__main__':
    D_l = 51.0
    D_r = 51.0
    k_l = 40.0
    k_r = 40.0
    d= 100.0
    params = [D_l, D_r, k_l, k_r, d]
    # params =  [45,  55,  35,  45, 100,] # Somehow has very low ZZ-parameter? Start for finding config I - honestly seems to be the perfect parameters.
    params = [59.20710113, 59.44370983 , 32.42994617 ,45.31466205, 100.35488958] #Slightly better, found after optimize above params
    # params = [75.27694031, 59.87547933, 18.5059748, 18.8518254, 107.10304619] # Old params from before normalization of spf_l & spf_r 
    # params = [75, 60, 19, 19, 105] # To perform a search around this minima
    # params = [99.99841511, 41.37852048, 16.36579386, 17.41846244, 111.66647596] # Found from optimizing config II from config I
    # params = [100, 40, 16, 17, 110] # Found from optimizing config II from config I
    ins = Optimizer(params=params, tol=1e-12, verbose=False, config='I')
    res = ins.optimize()
    breakpoint()
    exit()
    # Randomly initialized parameters
    info = {}
    score = 100
    for i in range(25):
        D_l = np.random.randint(10, 100)
        D_r = np.random.randint(10, 100)
        k_l = np.random.randint(5, 100)
        k_r = np.random.randint(5, 100)
        d = np.random.randint(80, 300)
        params = [D_l, D_r, k_l, k_r, d]
        ins = Optimizer(params=params, tol=1e-10, verbose=False, config='I')
        res = ins.optimize()
        info[i] = {
            'initial_params': params,
            'parameters': ins.params.tolist(),
            'entropy': ins.entropies,
            'S': ins.S.tolist(),
            'result': res.fun,
            'energies': ins.eigen_energies.tolist()
        }
        if res.fun < score:
            score = res.fun
            best_params = ins.params.tolist()
            iteration = i
    info['best'] = {
        'parameters': best_params,
        'score': score,
        'iteration': iteration,
        'targets': ins.target.tolist()
    }
    print(f"Best parameters: {best_params}")
    print(f"Best iteration: {iteration}")
    print(f"Best score: {score}")
    import json
    try:
        with open('data/optimization_results_280125_measurementconfig.json', 'w') as f:
            json.dump(info, f, indent=4)
    except:
        breakpoint()
    breakpoint()
