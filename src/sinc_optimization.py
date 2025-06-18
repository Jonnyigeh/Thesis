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
                num_func=4,
                grid_length=200,
                num_grid_points=400,
                a=0.1,
                alpha=1.0,
                max_iter=2_000,
                tol=1e-6,
                n_particles=2,
                scaling=1.0,
                verbose=True,
                params=None,
                init_params=None,
                dvr=False,
                search=False,
                config='I'
    ):
        """
        something informative..
        """
        self.num_l = num_func
        self.num_r = num_func
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
        self.init_params = np.asarray(init_params)
        self.search = search
        self.config = config
        if self.config == 'I':
            self.target = np.zeros(self.num_l ** 2)
        elif self.config == 'II':
            self.target = np.zeros(self.num_l ** 2)
            self.target[1] = 1.0
            self.target[2] = 1.0

        self.idx_01 = 0*self.num_r + 1
        self.idx_10 = 1*self.num_r + 0
        self.idx_11 = 1*self.num_r + 1
        if self.search:
            self.set_system(params, initial=True)
        else:
            self.set_system(init_params, initial=True)
        e, C = np.linalg.eigh(self.H)
        overlaps10 = np.abs(C[self.idx_10, :])**2
        overlaps01 = np.abs(C[self.idx_01, :])**2
        overlaps11 = np.abs(C[self.idx_11, :])**2
        # Find idx_01 and idx_10 in the reduced basis, ie eigenstate with maximum overlap with |01> and |10>
        self.eig_idx_01 = np.argmax(overlaps01)
        self.eig_idx_10 = np.argmax(overlaps10)
        self.eig_idx_11 = np.argmax(overlaps11)

        self.psi01 = np.zeros_like(C[:,1])
        self.psi10 = np.zeros_like(C[:,2])
        self.psi01[self.idx_01] = 1.0
        self.psi10[self.idx_10] = 1.0
        # print(abs(C[:,0])**2)
        # print(abs(C[:,1])**2)
        # print(abs(C[:,2])**2)
        # print(abs(C[:,3])**2)
        # print('what')
        # breakpoint()

        

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
    

    def set_system(self, params, initial=False):
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
        self.u_lr4 = self.basis._ulr4d
        self.bhs = sinc_BHS(self.h_l, self.h_r, self.u_lr, self.num_l, self.num_r)
        self.bhs.solve()
        self.eps_l = self.bhs.eps_l
        self.eps_r = self.bhs.eps_r
        if initial:
            self.c_l = self.bhs.c_l
            self.c_r = self.bhs.c_r
        self.h_l = self.c_l.conj().T @ self.h_l @ self.c_l
        self.h_r = self.c_r.conj().T @ self.h_r @ self.c_r
        # self.u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', self.c_l.conj(), self.c_r.conj(), self.u_lr, self.c_l, self.c_r)

        # New attempt, now using a 4D tensor when rotating the systme, but the collapsed 2D tensor is used inside BHS (only need diagonal part there)
        # breakpoint()
        # import time
        # t1 = time.time()
        # self.u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', self.c_l.conj(), self.c_r.conj(), self.u_lr4, self.c_l, self.c_r, optimize=True)
        # t2 = time.time()
        # breakpoint()

        # New attempt nr.2, using two-step einsum to avoid memory issues. Possible due to sparsity in u matrix (i!=j, k!=l is zero by construction) 
        M = np.einsum('ia, ij, ic -> acj', self.c_l.conj(), self.u_lr, self.c_l, optimize=True)
        self.u_lr = np.einsum('acj, jb, jd -> abcd', M, self.c_r.conj(), self.c_r, optimize=True)

        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        U = self.u_lr.reshape(*H.shape)
        self.H = H + U

    def _solve(self, params):
        self.set_system(params, initial=self.search)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]
        self.trans_energies = {
            'w_l1': float(round(eps_l[1] / (2 * np.pi), 5)),
            'w_r1': float(round(eps_r[1] / (2 * np.pi), 5)),
            'w_L1 + w_R1': float(round((eps_l[1] + eps_r[1]) / (2 * np.pi), 5)),
            'w_l2': float(round(eps_l[2] / (2 * np.pi), 5)),
            'w_r2': float(round(eps_r[2] / (2 * np.pi), 5)),
            '|01>->|02>': float(round(abs(eps_r[2] - eps_r[1]) / (2 * np.pi), 5)),
            '|10>->|20>': float(round(abs(eps_l[2] - eps_l[1]) / (2 * np.pi), 5)),
            '|01>->|10>': float(round(abs(eps_l[1] - eps_r[1]) / (2 * np.pi), 5)),
                                
        }
        
        eps, self.C = np.linalg.eigh(self.H)
        self.eigen_energies = eps
        
        return eps, self.C
    
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

    def _objective(self, params):
        # Solve static Hamiltonian at these params
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
        # Set som weights for various cost terms
        w_J = 0 # native two‐qubit coupling
        w_olap = 0 # overlap with logical states |10> and |01>
        w_spec = 1.0 # Spctator level entropy
        k_twobody = 0.0 # Two‐body detuning
        if self.config == 'II':
            w_J       = 10000.0
            target[self.eig_idx_01] = 1.0
            target[self.eig_idx_10] = 1.0
            w_olap = 10.0
            k_twobody = 10
        else:
            w_spec = 100
        qubit_idxs = [self.eig_idx_01, self.eig_idx_10]
        weights = np.ones_like(self.S)
        for k in range(len(self.S)):
            if k not in qubit_idxs: 
                # for spectator levels, we want to push S[k]→0
                weights[k] = w_spec

        # Compute the overlap with the logical states
        # |01> and |10> in the reduced basis
        # Note: C is the eigenvectors of the Hamiltonian. Cost is scaled in cost term further below.
        good_overlap = 0.0
        for k in qubit_idxs:
            good_overlap += abs(C[self.idx_01, k])**2
            good_overlap += abs(C[self.idx_10, k])**2
        # Penalize 'wrong' overlaps
        bad_overlap = 0.0
        spec_indx = [j for j in range(len(eps)) if j not in [self.idx_01, self.idx_10]]
        for k in [self.idx_01, self.idx_10]:
            for j in spec_indx:
                bad_overlap += abs(C[j, k])**2

        entropy_penalty = np.sum(weights * (self.S - target)**2)
        # entropy_penalty = np.linalg.norm(self.S - target)**2
        # logical sum‐frequency for the first excited:
        sum_w1 = eps_l[1] + eps_r[1]
        # second excited on left and right:
        w_l2 = eps_l[2]
        w_r2 = eps_r[2]

        # # Penalize transitions to the second excited state:
        k_trans = 10 
        # # distance‐squared from the sum‐frequency:
        d1 = abs(sum_w1 - w_l2)
        d2 = abs(sum_w1 - w_r2)
        trans_penalty = 0.0
        trans_penalty = k_trans / (d1*d1 + 1e-12) \
                    + k_trans / (d2*d2 + 1e-12)


        target_twobody = eps_l[1] + eps_r[0] # Excitation energy from |00> to |10> (we enforce this to be equal to |00>->|01> in the relevant config)
        # Penalize two-body resonance with wrong spectator levels
        twobody_gap_penalty = 0.0
        for i in range(1, len(eps_l)):
            for j in range(1, len(eps_r)):
                # if (i,j) in [(1,0), (0,1)]:
                #     # skip the two-body resonance with the qubit levels, should not be 
                #     continue
                # spectator level
                spectator = eps_l[i] + eps_r[j]
                twobody_gap_penalty += k_twobody / (abs(spectator - target_twobody)**2 + 1e-12)



        # ZZ‐term
        self.ZZ = 0
        if self.config == 'I':
            # self.ZZ = abs(eps[4] - eps[2] - eps[1] + eps[0])
            detune = 0.4 - abs(eps_l[0] - eps_r[0]) + (0.4 - abs(eps_l[1] - eps_r[1]))
            
        else:  # config II
            # self.ZZ = abs(eps[4] - eps[2] - eps[1] + eps[0])
            detune = 100 * abs(eps_l[1] - eps_r[1]) 

        # Native two‐qubit coupling J = |<01|H|10>|
        self.J     = abs(self.H[self.idx_01, self.idx_10])
        # Spectator‐level gap penalty:
        #    for every level k not in the qubit manifold, 
        #    add kappa/(min_i |eps[k]–eps[i]|)^2
        kappa = 1e-1
        others = [k for k in range(len(eps)) if k not in qubit_idxs]
        gap_penalty = 0.0
        for k in others:
            # find smallest detuning to either qubit level
            gap = min(abs(eps[k] - eps[q]) for q in qubit_idxs)
            gap_penalty += kappa / (gap**2 + 1e-12)

        # Off‐diagonal penalty: for every level j not in the qubit manifold,
        kappa_off = 1.0
        offdiag_penalty = 0.0
        for j in range(len(eps)):
            if j not in [self.idx_01, self.idx_10]:
                offdiag_penalty += kappa_off * abs(self.H[self.idx_01, j])**2
                offdiag_penalty += kappa_off * abs(self.H[self.idx_10, j])**2
        # Assemble cost function
        w_detune  = 1.0
        cost = (
            entropy_penalty
        + self.ZZ
        + gap_penalty
        + twobody_gap_penalty
        - w_J * self.J
        - w_olap * good_overlap
        + w_olap * bad_overlap
        + offdiag_penalty
        + w_detune * detune
        + trans_penalty
        )
        self.cost = {
            'entropy_penalty': entropy_penalty,
            'ZZ': self.ZZ,
            'gap_penalty': gap_penalty,
            'twobody_gap_penalty': twobody_gap_penalty,
            'J': self.J,
            'good_overlap': good_overlap,
            'bad_overlap': bad_overlap,
            'offdiag_penalty': offdiag_penalty,
            'detune': detune,
            'trans_penalty': trans_penalty,
        }


        # optional: print diagnostics every so often
        if self.verbose and self.counter % 20 == 0:
            print(f"cost:{cost:.3e}  J:{J:.3e}  gap:{gap_penalty:.3e}  ZZ:{self.ZZ:.3e}")

        return cost

    # J coupling focused objective function
    def __objective(self, params):
        """Objective function for the optimization problem"""
        # Solve static Hamiltonian at these params
        eps, C = self._solve(params / self.scaling)
        # single‐particle spacings (zero‐grounded)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]
        # Higher order spacings
        eps_l2 = eps_l[2] - eps_l[1]
        eps_r2 = eps_r[2] - eps_r[1]
        # # Find indices
        overlaps10 = np.abs(C[self.idx_10, :])**2
        overlaps01 = np.abs(C[self.idx_01, :])**2
        # Find idx_01 and idx_10 in the reduced basis, ie eigenstate with maximum overlap with |01> and |10>
        self.eig_idx_01 = np.argmax(overlaps01)
        self.eig_idx_10 = np.argmax(overlaps10)
        # Punish low anharmonicity
        k_anharm = 10.0
        anh_penalty = k_anharm * (eps_l2)**2 + k_anharm * (eps_r2)**2

        self.ZZ = 0
        w_J = 1
        w_O = 0.1
        # Reward strong 01↔10 coupling:
        self.J = abs(self.H[self.idx_01, self.idx_10])
        coupling_reward = - w_J * self.J

        # Reward maximal qubit‐manifold overlap:
        overlap_reward = - w_O * (
            abs(C[self.idx_01, self.eig_idx_01])**2
        + abs(C[self.idx_10, self.eig_idx_10])**2
        )



        # Compute entropies
        w_S = 0.01
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
        if self.config == 'II':
            target[self.eig_idx_01] = 1.0
            target[self.eig_idx_10] = 1.0
        else:
            pass

        # entropy_penalty = np.linalg.norm(self.S - target)**2
        # # Set qubit indices
        # qubit_idxs = [self.eig_idx_01, self.eig_idx_10]
        # # Compute the overlap with the logical states
        # # |01> and |10> in the reduced basis
        # # Note: C is the eigenvectors of the Hamiltonian. Cost is scaled in cost term further below.
        # good_overlap = 0.0
        # for k in qubit_idxs:
        #     good_overlap += abs(C[self.idx_01, k])**2
        #     good_overlap += abs(C[self.idx_10, k])**2
        # # Penalize 'wrong' overlaps
        # bad_overlap = 0.0
        # spec_indx = [j for j in range(len(eps)) if j not in [self.idx_01, self.idx_10]]
        # for k in qubit_idxs:
        #     for j in spec_indx:
        #         bad_overlap += abs(C[j, k])**2

        # # Coupling reward
        # self.J     = abs(self.H[self.idx_01, self.idx_10])
        # coupling_reward = (10 - self.J)

        # cost = (
        #     anh_penalty
        #     + entropy_penalty
        #     + (2 - good_overlap) # Overlap should be 1, for both. Maximum is 2, minimum is 0
        #     + bad_overlap
        #     + coupling_reward
        # )
        trans_energy = np.abs(eps_l[1] - eps_r[1])

        if self.config == 'II':
            cost = (
                # anh_penalty 
                + w_S * np.sum((self.S - target)**2)
                + (1 + coupling_reward)
                + trans_energy
                # + overlap_reward
            )
        else:  # config I
            cost = (
                w_S * np.sum((self.S - target)**2)
            )


        return cost
    # two level objective (for l = 2)
    def objective(self, params):
        """Objective function for the optimization problem"""
        # Solve static Hamiltonian at these params
        eps, C = self._solve(params / self.scaling)
        # single‐particle spacings (zero‐grounded)
        eps_l = self.eps_l - self.eps_l[0]
        eps_r = self.eps_r - self.eps_r[0]
        # # Find indices
        overlaps10 = np.abs(C[self.idx_10, :])**2
        overlaps01 = np.abs(C[self.idx_01, :])**2
        # Find idx_01 and idx_10 in the reduced basis, ie eigenstate with maximum overlap with |01> and |10>
        self.eig_idx_01 = np.argmax(overlaps01)
        self.eig_idx_10 = np.argmax(overlaps10)
        if self.eig_idx_01 == self.eig_idx_10:
            if self.eig_idx_01 == 1:
                self.eig_idx_01 = 2
            elif self.eig_idx_10 == 2:
                self.eig_idx_10 = 1
            else:
                raise ValueError(f"Eigenstate indices are wrong 01: {self.eig_idx_01}, 10: {self.eig_idx_10}. ")
        
        target = np.zeros_like(eps)
        self.delta_E = np.abs(eps[2] - eps[1])  
        wrong_delta_E = np.abs(eps[1] - eps[0]) + np.abs(eps[3] - eps[2])
        w_delta_E = 0.0
        w_J = 0.0
        w_S = 10
        w_trans = 0.0
        w_offdiag = 0.0
        w_sym = 0.0
        w_fidel = 0.0
        if self.config == 'II':
            target[self.eig_idx_01] = 1.0
            target[self.eig_idx_10] = 1.0
            w_J = 0.01
            w_trans = 0.1
            w_offdiag = 10.0
            w_cross_diag = 0.0
            w_delta_E = 10.0
            w_sym = 1.0
            w_fidel = 100
        else:
            w_offdiag = 10000
            w_cross_diag = w_offdiag

        # symmetry penalty
        symmetry_penalty = 0.0
        symmetry_penalty += abs(self.params[0] - self.params[1])
        symmetry_penalty += abs(self.params[2] - self.params[3])

        # Fidelity penalty
        fidelity_penalty = 0.0
        psi_plus = (self.psi01 + self.psi10) / np.sqrt(2)
        psi_minus = (self.psi01 - self.psi10) / np.sqrt(2)
        fidelity_penalty += abs(psi_plus.conj() @ C[:, self.eig_idx_01])**2
        fidelity_penalty += abs(psi_minus.conj() @ C[:, self.eig_idx_10])**2



        # Calculate off-diagonal penalty
        offdiag_penalty = 0.0
        for i in range(len(eps)):
            for j in range(len(eps)):
                if i != j:
                    offdiag_penalty += w_offdiag * abs(self.H[i, j])**2
                elif (i,j) in [(self.idx_01, self.idx_10), (self.idx_10, self.idx_01), (0,3), (3,0)]:
                    # Cross-diagonal penalty for the qubit states
                    offdiag_penalty += w_cross_diag * abs(self.H[i, j])**2
        # Reward strong 01↔10 coupling:
        self.J = abs(self.H[self.idx_01, self.idx_10])
        coupling_penalty =  w_J / self.J
        trans_energy = np.abs(eps_l[1] - eps_r[1])
        self.ZZ = 0
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
        J_deltaE = - np.abs(self.J / (self.delta_E + 1e-12))
        cost = (
            w_S * np.sum((self.S - target)**2)
            + w_delta_E * J_deltaE
            + coupling_penalty
            + w_trans * trans_energy
            + offdiag_penalty
            # + w_delta_E * self.delta_E
            # + w_delta_E * wrong_delta_E
            + w_sym * symmetry_penalty
            + w_fidel * fidelity_penalty
        )
        self.cost = {
            'entropy_penalty': w_S * np.sum((self.S - target)**2),
            'ZZ': self.ZZ,
            'gap_penalty': self.delta_E,  # Not used in this case
            'twobody_gap_penalty': 0.0,  # Not used in this case
            'J': self.J,
            'good_overlap': 0.0,  # Not used in this case
            'bad_overlap': 0.0,  # Not used in this case
            'offdiag_penalty': 0.0,  # Not used in this case
            'detune': trans_energy,  # Not used in this case
        }


        return cost

    def optimize(self):
        def _callback(params, *kwargs):
            """Callback function for optimization"""
            if self.counter % 20 == 0 or self.counter == 0:
                print(f"Iteration: {self.counter}")
                print(f"Coupling")
                print(f"Config: {self.config}")
                print(f"Coupling: {self.J}")
                print(f"Delta E: {self.delta_E}")
                # print(f'{self.H.real[:6,:6]}')
                # print(f'eigenvectors, logical block: {np.abs(self.C[np.ix_([0,1,4,5], [0,1,4,5])])**2}') ##{np.abs(self.C[:6,:6])**2}')
                print(f'1st: {np.abs(self.C[:,0])**2},\n 2nd: {np.abs(self.C[:,1])**2},\n 3rd: {np.abs(self.C[:,2])**2},\n 4th: {np.abs(self.C[:,3])**2}\n, 5th: {np.abs(self.C[:,4])**2},\n 6th: {np.abs(self.C[:,5])**2}')
                print(f"Current indices: |01> = {self.eig_idx_01},|10>= {self.eig_idx_10}, |11>= {self.eig_idx_11}")
                print(f"Parameters: {np.asarray(params / self.scaling)}")
                print(f"Objective: {self.objective(params)}")
                print(f"ZZ param: {self.ZZ}")
                print(f"Energies: {self.eigen_energies[:len(self.eigen_energies)] / (2 * np.pi)}")
                print(f"Transition energies: {self.trans_energies}")
                print(f"Entropy: {self.S[:self.l]}")
            self.counter += 1
            self.entropies.append(self.S[0])
            self.params = params / self.scaling
        constraints = [
            {'type': 'ineq', 'fun': lambda params: self._constraint(params / self.scaling)}
        ]
        bounds = [(4 * self.scaling, 150 * self.scaling),  # D_l must be positive
            (4 * self.scaling, 150 * self.scaling),  # D_r must be positive
            (3 * self.scaling, 50 * self.scaling),  # k_l must be positive
            (3 * self.scaling, 50 * self.scaling),
            (5 * self.scaling, 25 * self.scaling)]  # k_r must be positive
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
            options={'disp': self.verbose, 'maxiter': self.max_iter, 'final_tr_radius': self.tol, 'initial_tr_radius': 5.0}
        ) 
        print(f"Optimization result: {result}")
        # print(f"Final cost, entropy_penalty {self.cost['entropy_penalty']}," + 
        #         f"ZZ {self.cost['ZZ']}, gap_penalty {self.cost['gap_penalty']}, twobody_gap_penalty {self.cost['twobody_gap_penalty']}, J {self.cost['J']}," +
        #         f"good_overlap {self.cost['good_overlap']}, bad_overlap {self.cost['bad_overlap']}, offdiag_penalty {self.cost['offdiag_penalty']}," +
        #         f"detune {self.cost['detune']}")
        print(f"Entropy: {self.S}, Coupling: {self.J}, Delta E: {self.delta_E}, Energies: {self.eigen_energies}, Hamiltonian: {self.H.real}")
        self.params = result.x / self.scaling
        
        return result
    
    def test_params(self, params, config='II'):
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



    # Attempt at l=4
    params= [50, 50, 20, 20, 40]
    params_I = [52.49089987, 53.62683999, 11.07673065 ,16.52311111 ,19.65248418]
    # 10.06 fixed some minor scaling error in coulomb interaction
    params_I = [65.00091526, 38.39692472, 25.00070405,  8.72380963, 18.49693257]
    params = [76.50992129, 48.1146782 , 13.351184 ,  13.68434784 ,18.19766692]


    # 10.06 2nd. 
    params_II = [76.73049659 ,48.73421941 ,13.57163477 ,13.81777894 ,16.64430113]
    params_I  = [80.  ,       67.73885159 ,30.  ,       22.60731225 ,16.65149379]

    #10.06 3rd 
    params_II =  [76.68721384, 48.86793519, 13.8406338  ,14.09016223 ,16.64768406]
    params_I = [73.34278003, 59.58514677, 31.82915013 ,21.00838547, 16.66399601]
    # params_II = [40, 40, 20, 20, 30]
    


    # 11.06 num_func=2
    params_I = [50, 50, 10, 10, 15]
    # params_II = [30, 30, 5, 5, 5]
    # params_II = [29.25989332 ,32.39735744  ,5.06592519 , 5.03878633 , 6.54364108]
    # params_II = [50.84165245 ,50.83516511, 10.02227946, 10.02234144,  5.00000169]
    # params_I = [49.25366164 ,52.01135821 , 9.99855117,  4.87048312 , 6.74518928]
    # params_II = [18.38477631, 18.38477631,  2.    ,      2.  ,        5.        ]


    params_init = [49.34681143, 50.95476102 , 9.16921973 , 9.1547418 ,  5.        ]
    params_II = [44.32614953 ,47.77907786  ,3.13291496 , 3.12745435 , 5.        ]
    params_I = [48.514703 ,  50.90995246 ,10.29958596  ,9.08639849  ,5.        ]


    params_init = [43.1724973,  45.95418964 , 2.00245395 , 2.    ,      5.        ]
    params_II = [43.1724973,  45.95418964 , 2.00245395 , 2.    ,      5.        ]
    params_I = [43.04984933, 46.04854472,  2.01756909 , 2.81522255 , 5.00399675]


    params_init = [48.09036408 ,47.25786891, 13.61383221,  1.48153242 ,17.46706477]
    # params_init = [79.99997074, 79.89948011,  5.0005172 ,  5.00067906,  5.        ]
    params_I = [79.99997074, 79.89948011,  5.0005172 ,  5.00067906,  5.        ]
    params_II = [80.2327351  ,80.97045131  ,2.0019729  , 2.50222853 , 5.        ]   
    params = [22.57803111 ,28.42655626  ,3.        ,  3.1781328,   5.42763218]
    init_params = [55.65289394, 53.3737098 , 18.32127221 ,16.85784673, 37.00170438]
    params = [40, 40, 7,7, 10]
    # 18.06 L = 2
    init_params = [39.94917866, 40.08475402  ,9.44196147 , 8.48598234 ,10.30854679]
    params_II_from_init = [40.02035412, 41.97386199 , 6.99384561  ,7.01237375 , 9.96140797]
    # ins = Optimizer(params=params, init_params=init_params, tol=1e-7, verbose=False, config='II', dvr=True, num_func=2, search=False)
    # ins.optimize()

    # 18.06 L = 4 (all systems nominally have 4 levels)
    # params = [60, 55, 10, 5, 15]
    # params_I = [50.61022309, 49.97873181, 15.15957671 ,14.7814281 , 25.        ]
    # params_II = [50.56567102, 55.19190111 ,14.44324624 ,14.34208258 ,24.97394011]


    params_I = [62.17088395, 60.73364357 ,19.89474221 ,21.81940414, 15.        ]
    params_II = [45, 45, 12,12, 7]
    init_params = params_I
    ins = Optimizer(params=params_II, init_params=params_I, tol=1e-8, verbose=False, config='II', dvr=True, num_func=4, search=False)
    ins.optimize()

    breakpoint()
    exit()


    for i in range(18):
        if i < 10:
            D_l = params_II[0] + np.random.randint(-6, 6)
            D_r = D_l
            k_l = params_II[2] + np.random.randint(-3, 3)
            k_r = k_l
            d = params_II[4] + np.random.randint(-2 ,2)
            params = [D_l, D_r, k_l, k_r, d]
        else:
            D_l = np.random.randint(30, 80)
            k_l = np.random.randint(1, 20)
            d = np.random.randint(3, 15)
            params = [D_l, D_l, k_l, k_l, d]
        ins = Optimizer(params=params, init_params=params_I, tol=1e-6, verbose=False, config='II', dvr=True, num_func=4, search=False)
        res = ins.optimize()

        print(f"Iteration {i}: {res.fun}")
        if res.fun < 1e-6:
            print(f"Found good parameters: {ins.params}")
        # ins = Optimizer(params=params_init, init_params=params_init, tol=1e-6, verbose=False, config='II', dvr=True, num_func=2, search=False)
        # res = ins.optimize()
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
        ins = Optimizer(params=params, tol=1e-12, verbose=False, config='II')
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