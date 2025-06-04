import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from scipy.linalg import svdvals
from scipy.optimize import minimize, differential_evolution, dual_annealing

# Local imports
from utils.sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS
from utils.potential import MorsePotentialDW
from utils.qd_system import ODMorse


class Optimizer:
    """
    Optimization of parameters for the Morse double well potential. 
    """
    def __init__(self,
                l=25,
                grid_length=200,
                num_grid_points=400,
                a=0.1,
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
        self.num_l = 4
        self.num_r = 4
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
            self.target[1] = 1.0
            self.target[2] = 1.0

        self.idx_01 = 0*self.num_r + 1
        self.idx_10 = 1*self.num_r + 0
        self.eig_idx_01 = self.idx_01
        self.eig_idx_10 = self.idx_10
    
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
        self.potential = MorsePotentialDW(
            *params,
        )
        self.basis = ODMorse(
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

    def old_objective(self, params):
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
            detuning_penalty = 10 * np.abs(self.eps_l[1] - self.eps_r[1])

        entropy_penalty = np.linalg.norm(self.S[:len(self.target)] - self.target) ** 2

        return entropy_penalty + self.ZZ + detuning_penalty + const_penalty

    def objective(self, params):
        # 1) Solve static Hamiltonian at these params
        eps, C = self._solve(params / self.scaling)
        # single‐particle spacings (zero‐grounded)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]

        # Find overlap with logical states |10> and |01>
        overlaps10 = np.abs(C[self.idx_10, :])**2
        overlaps01 = np.abs(C[self.idx_01, :])**2
        # Find idx_01 and idx_10 in the reduced basis
        self.eig_idx_01 = np.argmax(overlaps01)
        self.eig_idx_10 = np.argmax(overlaps10)

        # Compute entropies
        self.S = np.zeros(len(eps))
        for i in range(len(eps)):
            self._make_density_matrix(C[:,i])
            rho = np.trace(
                self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0],
                                self.h_l.shape[0], self.h_l.shape[0]),
                axis1=0, axis2=2
            )
            self.S[i] = self._find_VN_entropies(rho)

        # Entropy penalty: push S[qubit]→1 and S[others]→0
        target = np.zeros_like(self.S)
        w_J = 0
        if self.config == 'II':
            w_J       = 10.0
            target[self.eig_idx_01] = 1.0
            target[self.eig_idx_10] = 1.0
            w_spec = 1.0
        else:
            w_spec = 100
        qubit_idxs = [self.eig_idx_01, self.eig_idx_10]
        weights = np.ones_like(self.S)
        for k in range(len(self.S)):
            if k not in {self.eig_idx_01, self.eig_idx_10}: 
                weights[k] = w_spec

        entropy_penalty = np.sum(weights * (self.S - target)**2)
        # entropy_penalty = np.linalg.norm(self.S - target)**2
        self.ZZ = 0

        # logical sum‐frequency for the first excited:
        sum_w1 = eps_l[1] + eps_r[1]

        # second excited on left and right:
        w_l2 = eps_l[2]
        w_r2 = eps_r[2]

        # penalty strength:
        kappa_trans = 1e-1   # you can increase this to 1.0 or 10.0 if needed

        # distance‐squared from the sum‐frequency:
        d1 = abs(sum_w1 - w_l2)
        d2 = abs(sum_w1 - w_r2)

        trans_penalty = kappa_trans / (d1*d1 + 1e-12) \
                    + kappa_trans / (d2*d2 + 1e-12)

        # ZZ‐term: keep your original “iSWAP” detuning penalty
        if self.config == 'I':
            self.ZZ = abs(eps[4] - eps[2] - eps[1] + eps[0])
            detune = 0.4 - abs(eps_l[0] - eps_r[0]) + (0.4 - abs(eps_l[1] - eps_r[1]))
            
        else:  # config II
            self.ZZ = abs(eps[4] - eps[2] - eps[1] + eps[0])
            detune = 100 * abs(eps_l[1] - eps_r[1]) 

        # Native two‐qubit coupling J = |<01|H|10>|
        #    requires that you know idx_01, idx_10 in your reduced basis
        i01, i10 = self.idx_01, self.idx_10
        H_red = self.H  # already in reduced Hartree basis
        self.J     = abs(H_red[i01, i10])

        # 7) Spectator‐level gap penalty:
        #    for every level k not in the qubit manifold, 
        #    add kappa/(min_i |eps[k]–eps[i]|)^2
        kappa = 1e-1
        others = [k for k in range(len(eps)) if k not in qubit_idxs]
        gap_penalty = 0.0
        for k in others:
            # find smallest detuning to either qubit level
            gap = min(abs(eps[k] - eps[q]) for q in qubit_idxs)
            gap_penalty += kappa / (gap**2 + 1e-12)

        # 8) Assemble final cost:
        #    - we want large J => subtract w_J*J
        #    - we want large detune => subtract w_detune*detune
        #    + penalties for entropy, ZZ, gap
        
        w_detune  = 1.0
        cost = (
            entropy_penalty
        + self.ZZ
        + gap_penalty
        - w_J * np.log10(self.J + 1e-12)  
        + w_detune * detune
        + trans_penalty
        )

        # optional: print diagnostics every so often
        if self.verbose and self.counter % 20 == 0:
            print(f"cost:{cost:.3e}  J:{J:.3e}  gap:{gap_penalty:.3e}  ZZ:{self.ZZ:.3e}")

        return cost


    def optimize(self):
        def _callback(params, *kwargs):
            """Callback function for optimization"""
            if self.counter % 20 == 0 or self.counter == 0:
                print(f"Iteration: {self.counter}")
                print(f"Config: {self.config}")
                print(f"Coupling: {self.J}")
                print(f"Current indices: |01> = {self.eig_idx_01},|10>= {self.eig_idx_10}")
                print(f"Parameters: {params / self.scaling}")
                print(f"Objective: {self.objective(params)}")
                print(f"ZZ param: {self.ZZ}")
                print(f"Energies: {self.eigen_energies[:6] / (2 * np.pi)}")
                print(f"Transition energies: {self.trans_energies}")
                print(f"Entropy: {self.S[:15]}")
            self.counter += 1
            self.entropies.append(self.S[0])
        constraints = [
            {'type': 'ineq', 'fun': lambda params: self._constraint(params / self.scaling)}
        ]
        bounds = [(10 * self.scaling, 80 * self.scaling),  # D_l must be positive
            (10 * self.scaling, 80 * self.scaling),  # D_r must be positive
            (5 * self.scaling, 30 * self.scaling),  # k_l must be positive
            (5 * self.scaling, 30 * self.scaling),
            (15 * self.scaling, 30 * self.scaling)]  # k_r must be positive
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
    
    def test_params(self, params, config='I'):
        """Test the parameters and return the results"""
        self.set_system(params)
        eps, C = self._solve(params)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]
        def test_objective(params):
            self.S = np.zeros(len(eps))
            for i in range(len(eps)):
                self._make_density_matrix(C[:,i]) # Each column is the energy eigenstates
                rho = np.trace(self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0]), axis1=0, axis2=2)
                # and then entropy
                self.S[i] = self._find_VN_entropies(rho)
            const_penalty = 0
            self.ZZ = 0
            if config == 'I':
                self.ZZ = np.abs(eps[4] - eps[2] - eps[1] + eps[0])
                detuning_penalty = 0.2 - np.abs(self.eps_l[0] - self.eps_r[0])
            elif config == 'II':
                self.ZZ = np.abs(eps[5] - eps[2] - eps[1] + eps[0])
                detuning_penalty = 10 * np.abs(self.eps_l[1] - self.eps_r[1])
            entropy_penalty = np.linalg.norm(self.S[:len(self.target)] - self.target) ** 2

            return entropy_penalty + self.ZZ + detuning_penalty + const_penalty
        
        J = abs(self.H[self.idx_01, self.idx_10])
        print("Coupling |01⟩↔|10⟩:", J)
        print(f"Current indices: |01> = {self.eig_idx_01},|10>= {self.eig_idx_10}")
        print('timescale:', np.pi / (2 * J))
        cost = test_objective(params)
        print(f"Configuration: {config}")
        # print(f'Min gap: {min_gap / (2 * np.pi)} GHz')
        print(f"Parameters: {params}")
        print(f"Objective: {cost}")
        print(f"ZZ param: {self.ZZ}")
        print(f"Energies: {self.eigen_energies[:6] / (2 * np.pi)}")
        print(f"Transition energies: {self.trans_energies}")
        print(f"Entropy: {self.S[:10]}")
    
        
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

    # 28.05:
    params_II = [73.44552758, 71.99208131, 29.16163181, 29.16725463, 42.80228627]
    params_I = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]

    # 29.05 new objective function
    params_II =  [71.16885088, 70.54826573, 29.43632779,  29.43749691,  49.98418216]
    # params_II =  [71.16716389 ,70.64673147 ,29.43534355 ,29.43717471 ,48.99924303]
    # params_II = [50, 50, 15, 15, 25]
    ## [70.86052864 43.48554971 29.71132851 11.18930789 26.55392469]
    # 02.06 
    # params_II = [47.96777414, 50.42158537, 13.19020138, 13.15246072, 16.37341536] # 
    params_II = [63.68144808, 43.05416999 ,10.69127355 ,10.90371128, 16.02656697]
    params_I = [61.48354464, 37.12075554, 22.36820274, 8.1535532, 16.58949561]

    ins = Optimizer(params=params_II, tol=1e-8, verbose=False, config='I', dvr=True)
    # res = ins.optimize()
    ins.test_params(params_I, config='I')

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