import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from scipy.linalg import svdvals
from scipy.optimize import minimize, differential_evolution, dual_annealing

# Local imports
import quantum_systems as qs
from sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS


class Optimizer:
    """
    Optimization of parameters for the Morse double well potential. 
    """
    def __init__(self,
                l=25,
                grid_length=200,
                num_grid_points=400,
                a=0.25,
                alpha=1.0,
                max_iter=1_000,
                tol=1e-6,
                n_particles=2,
                scaling=1.0,
                verbose=True,
                params=None,
                dvr=False,
                config='I'
    ):
        """
        something informative..
        """
        self.num_l = 8
        self.num_r = 8
        self.l = l
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
            self.target = np.zeros(self.num_l ** 2)
        elif self.config == 'II':
            self.target = np.zeros(self.num_l ** 2)
            self.target[0] = 0.0
            self.target[1] = 1.0
            self.target[2] = 1.0
            self.target[3] = 0.0
            self.target[4] = 0.0
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
    

    def set_system(self, params):
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
        )
        self.basis = qs.ODMorse(
            l=self.l,
            grid_length=self.grid_length,
            num_grid_points=self.num_grid_points,
            _a=self.a,
            alpha=self.alpha,
            potential=self.potential,
            dvr=True,
        )
        # self.H = self.basis.H
        # self.E, self.C = np.linalg.eigh(self.H)
        self.h_l = self.basis._h_l
        self.h_r = self.basis._h_r
        self.u_lr = self.basis._ulr
        self.bhs = sinc_BHS(self.h_l, self.h_r, self.u_lr, self.num_l, self.num_r)
        self.bhs.solve()
        self.eps_l = self.bhs.eps_l
        self.eps_r = self.bhs.eps_r
        self.c_l = self.bhs.c_l
        self.c_r = self.bhs.c_r
        self.h_l = self.c_l.conj().T @ self.h_l @ self.c_l
        self.h_r = self.c_r.conj().T @ self.h_r @ self.c_r
        self.u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', self.c_l.conj(), self.c_r.conj(), self.u_lr, self.c_l, self.c_r)
        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        U = self.u_lr.reshape(*H.shape)
        self.H = H + U

    def _solve(self, params):
        self.set_system(params)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]
        self.trans_energies = {
            'w_l1': float(round(eps_l[1] / (2 * np.pi), 5)),
            'w_r1': float(round(eps_r[1] / (2 * np.pi), 5)),
            'w_L1 + w_R1': float(round((eps_l[1] + eps_r[1]) / (2 * np.pi), 5)),
            'w_l2': float(round(eps_l[2] / (2 * np.pi), 5)),
            'w_r2': float(round(eps_r[2] / (2 * np.pi), 5)),
        }
        
        eps, C = np.linalg.eigh(self.H)
        self.eigen_energies = eps
        
        return eps, C
    
    def _find_VN_entropies(self, rho):
        """Find entropy from reduced density matrix"""
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log2(eigs + 1e-15))    
    def _make_density_matrix(self, C):
        # self._rho = np.zeros((self.num_func ** 2, self.num_func ** 2), dtype=np.complex128)
        # for n in range(self.n_particles):
        #     self._rho += np.outer(C[n], np.conj(C[n]).T)
        self._rho = np.outer(C, np.conj(C))

    def objective(self, params):
        eps, C = self._solve(params / self.scaling)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]
        # Needed if not using COBYQA
        # def constrained_objective(params):
        #     constraint_value = self._constraint(params / self.scaling)
        #     penalty = 1e6 * max(0, -constraint_value)  # Penalize constraint violations
        #     return penalty
        self.S = np.zeros(len(eps))
        for i in range(len(eps)):
            self._make_density_matrix(C[:,i]) # Each column is the energy eigenstates
            rho = np.trace(self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0]), axis1=0, axis2=2)
            # and then entropy
            self.S[i] = self._find_VN_entropies(rho)
        # const_penalty = constrained_objective(params)
        const_penalty = 0
        self.ZZ = 0
        if self.config == 'I':
            self.ZZ = np.abs(eps[4] - eps[2] - eps[1] + eps[0])
            detuning_penalty = 0.2 - np.abs(self.eps_l[0] - self.eps_r[0])
        elif self.config == 'II':
            self.ZZ = np.abs(eps[5] - eps[2] - eps[1] + eps[0])
            detuning_penalty = np.abs(self.eps_l[1] - self.eps_r[1])

        entropy_penalty = np.linalg.norm(self.S[:len(self.target)] - self.target) ** 2

        return entropy_penalty + self.ZZ + detuning_penalty + const_penalty

    def optimize(self):
        def _callback(params, *kwargs):
            """Callback function for optimization"""
            if self.counter % 20 == 0 or self.counter == 0:
                print(f"Iteration: {self.counter}")
                print(f"Parameters: {params / self.scaling}")
                print(f"Objective: {self.objective(params)}")
                print(f"ZZ param: {self.ZZ}")
                print(f"Energies: {self.eigen_energies[:6] / (2 * np.pi)}")
                print(f"Transition energies: {self.trans_energies}")
                print(f"Entropy: {self.S[:10]}")
            self.counter += 1
            self.entropies.append(self.S[0])
        constraints = [
            {'type': 'ineq', 'fun': lambda params: self._constraint(params / self.scaling)}
        ]
        bounds = [(10 * self.scaling, 100 * self.scaling),  # D_l must be positive
            (10 * self.scaling, 100 * self.scaling),  # D_r must be positive
            (10 * self.scaling, 100 * self.scaling),  # k_l must be positive
            (10 * self.scaling, 100 * self.scaling),
            (30 * self.scaling, 150 * self.scaling)]  # k_r must be positive
        self.counter = 0
        self.entropies = []
        # Use of minimize
        result = minimize(
            self.objective,
            self.params * self.scaling,
            method='COBYQA',
            bounds=bounds,
            constraints=constraints,
            callback=_callback,
            # tol=self.tol,
            options={'disp': self.verbose, 'maxiter': self.max_iter, 'final_tr_radius': self.tol}
        ) 
        

        self.params = result.x / self.scaling
        return result
    
if __name__ == '__main__':
    params = [50, 50, 50, 50, 50]
    params_I = [54.49928043, 64.82014356, 15.30251959,  5.        , 31.17089668]
    params_I = [89.05318891, 70.55927467, 46.69855885, 5., 66.12799976]
    # params_II = [54.55311912, 63.99547101, 15.48518718, 15.31795171, 31.70703971]
    # params_maybe_II = [54.55311912, 63.99547098, 15.48518719, 15.31795172, 31.70703973]
    params_II = [88.9432289, 70.22836355, 11.09930862, 11.17075134, 70.55100164]
    paramsII_long =[62.85734129, 60.36558025, 20., 20.15532797, 200.]
    paramsII_also_long = [77.13829616,  86.31735811,  34.57979308,  34.45182811, 188.08796131]
    paramsI_also_long = [ 95.70685722 , 76.66934364 , 54.2000149  , 10.        , 189.46694742]


    params_II = [73.44693037, 71.99175625 ,29.16144963 ,29.16767609, 42.79831711]
    params_I = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
    ins = Optimizer(params=params_II, tol=1e-10, verbose=False, config='II', dvr=True)
    res = ins.optimize()

    breakpoint()
    exit()


    info = {}
    score = 100
    for i in range(25):
        D_l = np.random.randint(10, 100)
        D_r = np.random.randint(10, 100)
        k_l = np.random.randint(10, 100)
        k_r = np.random.randint(10, 100)
        d = np.random.randint(30, 150)
        params = [D_l, D_r, k_l, k_r, d]
        ins = Optimizer(params=params, tol=1e-10, verbose=False, config='II')
        res = ins.optimize()
        info[i] = {
            'initial_params': params,
            'parameters': ins.params.tolist(),
            # 'entropy': ins.entropies,
            'result': res.fun,
            'energies': ins.eigen_energies[:6].tolist(),
            'transition': ins.trans_energies,
            'ground_state': [ins.eps_l[0], ins.eps_r[0]],
            'ZZ': ins.ZZ,
            'detuning': np.abs(ins.eps_l[1] - ins.eps_r[1]),
        }
        print(np.abs(res.fun))
        if np.abs(res.fun) < score:
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
            with open('data/sinc_optimization_results_040425_measurementconfig.json', 'w') as f:
                json.dump(info, f, indent=4)
        except:
            breakpoint()
    breakpoint()