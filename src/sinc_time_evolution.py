import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
from scipy.optimize import minimize
from scipy.special import erf
from tqdm import tqdm

# Local imports
from utils.potential import MorsePotentialDW
from utils.qd_system import ODMorse
from utils.sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS  
from utils.visualization import find_figsize


class time_evolution:
    def __init__(self,
                 params1,
                 params2,
                 l=25,
                 num_lr=8,
                 num_grid_points=400,
                 grid_length=200,
                 num_particles=2,
                 a=0.1,
                 alpha=1.0,
                 dt=0.1,
                 t_max=1.0,
                 chill_before=0.3,
                 ramp_up=1.5,      
                 plateau=0.5,        # time to sit at C_II
                 ramp_down=1.5,      # time to go from C_II → C_I
                 chill_after=1.0,    # time to sit at C_I at end
                 ramp='cosine',
                 integrator='U',
                 hartree=False):
        self.params1 = params1
        self.params2 = params2
        self.l = l
        self.num_l = num_lr
        self.num_r = num_lr
        self.num_grid_points = num_grid_points
        self.grid_length = grid_length
        self.num_particles = num_particles
        self.a = a
        self.alpha = alpha
        self.dt = dt
        # Set times for ramping and staying still
        self.chill_before = chill_before
        self.ramp_up      = ramp_up
        self.plateau      = plateau
        self.ramp_down    = ramp_down
        self.chill_after  = chill_after
        # compute total time
        self.t_max = (chill_before + ramp_up +
                      plateau   + ramp_down +
                      chill_after)
        # build time array
        self.t = np.arange(0, self.t_max, self.dt)
        self.num_steps = len(self.t)

        # precompute segment boundaries
        self.t1 = chill_before
        self.t2 = self.t1 + ramp_up
        self.t3 = self.t2 + plateau
        self.t4 = self.t3 + ramp_down
        # self.t_max = t_max
        # self.ramp_time = ramp_time
        # self.t_start = stay_still_time
        # self.t_end = stay_still_time
        # self.ramp = ramp
        # self.t = np.arange(0, self.t_max, self.dt)
        # self.num_steps = len(self.t)
        self.integrator = integrator
        self.hartree = hartree
        self.grid = np.linspace(-grid_length/2, grid_length/2, num_grid_points)

        # New attempt, try to evolve in config II base, and map it back into config I base.
        # So initially,
        # # At the moment, we have  
        E1, C1, H1, c_l1, c_r1 = self.set_system(params1)
        E2, C2, H2, c_l2, c_r2 = self.set_system(params2)
        self.E = E1
        self.C = C1
        self.H = H1
        self.H0 = self.H.copy()
        # Set the transformation matrices
        self.c_l1 = c_l1
        self.c_r1 = c_r1
        self.c_l2 = c_l2
        self.c_r2 = c_r2
        self.T1 = np.kron(c_l1, c_r1)
        self.T2 = np.kron(c_l2, c_r2)
        # self.C0 = self.C.copy()
        # self.E0 = self.E.copy()
        self.C0 = np.eye(self.h_l.shape[0] * self.h_r.shape[0])
        self.E0 = np.add.outer(self.eps_l, self.eps_r).ravel()
        # Find logical state indices
        self.idx_01 = 0 * self.num_l + 1
        self.idx_10 = 1 * self.num_r + 0
        self.idx_11 = 1 * self.num_r + 1
        row_10 = self.C0.conj().T[self.idx_10]
        row_01 = self.C0.conj().T[self.idx_01]
        row_11 = self.C0.conj().T[self.idx_11]
        self.eig_idx_10 = self.idx_10 #  np.argmax(np.abs(row_10)**2)
        self.eig_idx_01 = self.idx_01 # np.argmax(np.abs(row_01)**2)
        self.eig_idx_11 = self.idx_11 #np.argmax(np.abs(row_11)**2)
        # Set up the eigenveectors we wish to track
        self.psi00 = self.C0[:, 0] # |00⟩ # idx 0 * num_l + 0 = 0
        self.psi01 = self.C0[:, self.eig_idx_01] # |01⟩ # idx 0 * num_l + 1 = 1
        self.psi10 = self.C0[:, self.eig_idx_10] # |10⟩ # idx 1 * num_l + 0 = 4
        self.psi11 = self.C0[:, self.eig_idx_11] # |11⟩ # idx 1 * num_l + 1 = 5
        # self.psi00 = self.C0[:, 0]           # |00> at index 0*nr + 0
        # self.psi01 = self.C0[:, self.idx_01] # |01> at index 0*nr + 1
        # self.psi10 = self.C0[:, self.idx_10] # |10> at index 1*nr + 0
        # self.psi11 = self.C0[:, self.idx_11] # |11> at index 1*nr + 1
        # self.C = self.C0.copy() # Copy the initial state
        # self.S0 = np.zeros(len(self.E0))
        # for i in range(len(self.E0)):
        #     self._make_density_matrix(self.C0[:,i]) # Each column is the energy eigenstates
        #     rho = np.trace(self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0]), axis1=0, axis2=2)
        #     # and then entropy
        #     self.S0[i] = self._find_VN_entropies(rho)
        # idxs = [0, self.idx_01, self.idx_10, self.idx_11]
        # H_logical = self.H[np.ix_(idxs, idxs)]
        # print("off-diags:", H_logical - np.diag(np.diag(H_logical)))

    
    def set_system(self, params):
        manual_params=False
        if manual_params:
            print("Using manual parameters for the Morse potential.")
            # params = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
            # params = [61.48354464, 37.12075554, 22.36820274, 8.1535532, 16.58949561]
            # params = self.params2
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
        c_l = self.bhs.c_l
        c_r = self.bhs.c_r
        self.h_l = c_l.conj().T @ self.h_l @ c_l
        self.h_r = c_r.conj().T @ self.h_r @ c_r
        # self.u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', c_l.conj(), c_r.conj(), self.u_lr, c_l, c_r)
        # New attempt, 2-step solution using the full coulomb matrix
        M = np.einsum('ia, ij, ic -> acj', c_l.conj(), self.u_lr, c_l, optimize=True)
        self.u_lr = np.einsum('acj, jb, jd -> abcd', M, c_r.conj(), c_r, optimize=True)

        # Test a new solution for finding u - since u_lr is not a 4-tesnro, so the above might be incorrect (it is convrted to a 2-tensor through the Sinc BHS)
        # self.u_lr = self.c_l.conj().T @ self.u_lr @ self.c_r
        # u_diag = self.u_lr.flatten()
        # U = np.diag(u_diag)
        self.H_dist = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        U = self.u_lr.reshape(*self.H_dist.shape)
        H = self.H_dist + U
        E, C = np.linalg.eigh(H)

        return E, C, H, c_l, c_r
    
    


    def update_system(self, params, bhs=False):
        ## Test to update the system continually in C_II config basis, instead of C_I 
        # Update one-body hamiltonian with new potential
        self.potential = MorsePotentialDW(
            # *self.params2,
            *params,
        )
        new_V_l = np.clip(self.potential.left_pot(self.basis.left_grid), 0, 100)
        new_V_r = np.clip(self.potential.right_pot(self.basis.right_grid), 0, 100)
        newh_l = self.basis.no_1bpot_h_l + np.diag(new_V_l) 
        newh_r = self.basis.no_1bpot_h_r + np.diag(new_V_r) 
        self.u_lr = self.basis._ulr
        # And solve new Hartree equations (only done once, as we don't want to rediagonalize the whole system every time as this removs any accumulated phases)
        if bhs:
            self.bhs = sinc_BHS(newh_l, newh_r, self.u_lr, self.num_l, self.num_r)
            self.bhs.solve()
            self.eps_l = self.bhs.eps_l
            self.eps_r = self.bhs.eps_r
            c_l = self.bhs.c_l
            c_r = self.bhs.c_r
        else:
            c_l = self.c_l
            c_r = self.c_r
        # We transform the new one-body hamiltonian to the DVR basis
        self.h_r = c_r.conj().T @ newh_r @ c_r
        self.h_l = c_l.conj().T @ newh_l @ c_l
        # self.u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', self.c_l.conj(), self.c_r.conj(), self.u_lr, self.c_l, self.c_r)
        # New attempt, 2-step solution using the full coulomb matrix
        M = np.einsum('ia, ij, ic -> acj', c_l.conj(), self.u_lr, c_l, optimize=True)
        self.u_lr = np.einsum('acj, jb, jd -> abcd', M, c_r.conj(), c_r, optimize=True)
        # Test a new solution for finding u - since u_lr is not a 4-tesnro, so the above might be incorrect (it is convrted to a 2-tensor through the Sinc BHS)
        # self.u_lr = self.c_l.conj().T @ self.u_lr @ self.c_r
        # u_diag = self.u_lr.flatten()
        # U = np.diag(u_diag)

        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        U = self.u_lr.reshape(*H.shape)
        newH = H + U
        self.commutator = self.H @ newH - newH @ self.H
        self.H = newH
        # self.apply_pulse()
        self.E, self._C = np.linalg.eigh(self.H)
        self.S = np.zeros(len(self.E))
        for i in range(len(self.E)):
            self._make_density_matrix(self.C[:,i])
            rho = np.trace(
                self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0],
                                self.h_l.shape[0], self.h_l.shape[0]),
                axis1=0, axis2=2
            )
            self.S[i] = self._find_VN_entropies(rho)
    
    def _update_system_with_CI(self, params):
        # Update one-body hamiltonian with new potential
        self.potential = MorsePotentialDW(
            # *self.params2,
            *params,
        )
        new_V_l = np.clip(self.potential.left_pot(self.basis.left_grid), 0, 100)
        new_V_r = np.clip(self.potential.right_pot(self.basis.right_grid), 0, 100)
        newh_l = self.basis.no_1bpot_h_l + np.diag(new_V_l) / self.basis.dx
        newh_r = self.basis.no_1bpot_h_r + np.diag(new_V_r) / self.basis.dx
        self.u_lr = self.basis._ulr
        # And solve new Hartree equations (only done once, as we don't want to rediagonalize the whole system every time as this removs any acuumulated phases)
        # self.bhs = sinc_BHS(newh_l, newh_r, self.u_lr, self.num_l, self.num_r)
        # self.bhs.solve()
        # self.eps_l = self.bhs.eps_l
        # self.eps_r = self.bhs.eps_r
        # self.c_l = self.bhs.c_l
        # self.c_r = self.bhs.c_r
        # We transform the new one-body hamiltonian to the DVR basis
        self.h_r = self.c_r.conj().T @ newh_r @ self.c_r
        self.h_l = self.c_l.conj().T @ newh_l @ self.c_l
        # self.u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', self.c_l.conj(), self.c_r.conj(), self.u_lr, self.c_l, self.c_r)
        # New attempt, 2-step solution using the full coulomb matrix
        M = np.einsum('ia, ij, ic -> acj', self.c_l.conj(), self.u_lr, self.c_l, optimize=True)
        self.u_lr = np.einsum('acj, jb, jd -> abcd', M, self.c_r.conj(), self.c_r, optimize=True)
        # Test a new solution for finding u - since u_lr is not a 4-tesnro, so the above might be incorrect (it is convrted to a 2-tensor through the Sinc BHS)
        # self.u_lr = self.c_l.conj().T @ self.u_lr @ self.c_r
        # u_diag = self.u_lr.flatten()
        # U = np.diag(u_diag)

        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        U = self.u_lr.reshape(*H.shape)
        newH = H + U
        self.commutator = self.H @ newH - newH @ self.H
        self.H = newH
        # self.apply_pulse()
        self.E, self._C = np.linalg.eigh(self.H)
        self.S = np.zeros(len(self.E))
        for i in range(len(self.E)):
            self._make_density_matrix(self.C[:,i])
            rho = np.trace(
                self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0],
                                self.h_l.shape[0], self.h_l.shape[0]),
                axis1=0, axis2=2
            )
            self.S[i] = self._find_VN_entropies(rho)
        
    
    def U_step(self):
        """Compute the time evolution of the system using the U-propagator."""
        U = scipy.linalg.expm(-1j * self.H * self.dt)
        UUdag = U @ U.conj().T
        UdagU = U.conj().T @ U
        assert np.allclose(UUdag, np.eye(*UUdag.shape)), "U is not unitary"
        assert np.allclose(UdagU, np.eye(*UdagU.shape)), "Udag is not unitary"
        self.C = U @ self.C
        # self.psi = U @ self.psi
        self.psi00 = U @ self.psi00
        self.psi01 = U @ self.psi01
        self.psi10 = U @ self.psi10
        self.psi11 = U @ self.psi11
    
    def crank_nicolson_step(self, t=0):
        """Integrate the system using the Crank-Nicolson method."""
        I = np.eye(self.H.shape[0])
        H_current = self.H_prev
        H_next = self.H
        H_mid = 0.5 * (H_current + H_next)
        # A = np.linalg.inv(I + 1j * self.dt / 2 * H_next) @ (I - 1j * self.dt / 2 * H_current)
        A = np.linalg.inv(I + 1j * self.dt / 2 * H_mid) @ (I - 1j * self.dt / 2 * H_mid)
        self.C = A @ self.C
        self.psi00 = A @ self.psi00
        self.psi01 = A @ self.psi01
        self.psi10 = A @ self.psi10
        self.psi11 = A @ self.psi11


    
    def _lambda_t(self, t):
        """Return the time-dependent parameter lambda."""
        lmbda = np.zeros_like(t)
        # masks
        m0 = (t <=  self.t1)
        m1 = (t >  self.t1) & (t <= self.t2)
        m2 = (t >  self.t2) & (t <= self.t3)
        m3 = (t >  self.t3) & (t <= self.t4)
        m4 = (t >  self.t4)
        # assign segments
        lmbda[m0] = 0.0
        # up‐ramp: cosine from 0->1
        lmbda[m1] = 0.5*(1 - np.cos(np.pi*(t[m1]-self.t1)/self.ramp_up))
        # plateau
        lmbda[m2] = 1.0
        # down‐ramp: cosine from 1->0
        lmbda[m3] = 0.5*(1 + np.cos(np.pi*(t[m3]-self.t3)/self.ramp_down))
        # final chill
        lmbda[m4] = 0.0

        return lmbda
    def apply_pulse(self, delta=0.5):
                # H' in the C0 (logical/tensor product) basis
                H_prime_C0 = np.zeros((self.C0.shape[0], self.C0.shape[0]), dtype=complex)
                H_prime_C0[1, 4] = delta
                H_prime_C0[4, 1] = delta  # Hermitian conjugate

                # Map H' into DVR basis
                H_prime = self.C0 @ H_prime_C0 @ self.C0.conj().T

                # Add to full Hamiltonian
                self.H += H_prime
    def _evolve(self):
        """Evolve the system for a single time step."""
        if self.integrator == 'U':
            step = self.U_step
        elif self.integrator == 'CN':
            step = self.crank_nicolson_step
        else:
            raise NotImplementedError(f"Integrator {self.integrator} not implemented.")
        lmbda = self._lambda_t(self.t) # Time-dependent parameter
        # lmbda = np.zeros_like(self.t)
        plt.plot(self.t, lmbda)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        # Set up overlap arrays
        self.overlap = np.zeros((self.num_steps, *self.C0.shape), dtype=np.complex128)
        # |00>
        self.psi00_00_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi00_01_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi00_10_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi00_11_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        # |01>
        self.psi01_00_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi01_01_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi01_10_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi01_11_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        # |10>
        self.psi10_00_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi10_01_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi10_10_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi10_11_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        # |11>
        self.psi11_00_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi11_01_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi11_10_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.psi11_11_overlap = np.zeros((self.num_steps, 1), dtype=np.complex128)
        # # Set some vectors to calculate the overlaps
        psi00 = self.psi00.copy()
        psi01 = self.psi01.copy()
        psi10 = self.psi10.copy()
        psi11 = self.psi11.copy()    


        # # Set first element of matrices to identity
        self.psi00_00_overlap[0] = np.vdot(psi00.T, self.psi00)
        self.psi00_01_overlap[0] = np.vdot(psi01.T, self.psi00)
        self.psi00_10_overlap[0] = np.vdot(psi10.T, self.psi00)
        self.psi00_11_overlap[0] = np.vdot(psi11.T, self.psi00)
        self.psi01_00_overlap[0] = np.vdot(psi00.T, self.psi01)
        self.psi01_01_overlap[0] = np.vdot(psi01.T, self.psi01)
        self.psi01_10_overlap[0] = np.vdot(psi10.T, self.psi01)
        self.psi01_11_overlap[0] = np.vdot(psi11.T, self.psi01)
        self.psi10_00_overlap[0] = np.vdot(psi00.T, self.psi10)
        self.psi10_01_overlap[0] = np.vdot(psi01.T, self.psi10)
        self.psi10_10_overlap[0] = np.vdot(psi10.T, self.psi10)
        self.psi10_11_overlap[0] = np.vdot(psi11.T, self.psi10)
        self.psi11_00_overlap[0] = np.vdot(psi00.T, self.psi11)
        self.psi11_01_overlap[0] = np.vdot(psi01.T, self.psi11)
        self.psi11_10_overlap[0] = np.vdot(psi10.T, self.psi11)
        self.psi11_11_overlap[0] = np.vdot(psi11.T, self.psi11)


        # Save also the states evolution..
        self.psi_00t = np.zeros((self.num_steps, *psi00.shape), dtype=np.complex128)
        self.psi_01t = np.zeros((self.num_steps, *psi01.shape), dtype=np.complex128)
        self.psi_10t = np.zeros((self.num_steps, *psi10.shape), dtype=np.complex128)
        self.psi_11t = np.zeros((self.num_steps, *psi11.shape), dtype=np.complex128)
        


        # self.overlap = np.zeros((self.num_steps, *self.C0[0].shape), dtype=np.complex128)
        # self.psi_t = np.zeros((self.num_steps, *self.psi.shape), dtype=np.complex128)
        # self.psi_t[0] = self.psi.copy()
        self.energies = np.zeros((self.num_steps, *self.E.shape))
        # self.overlap[0] = self.C0.conj().T @ self.psi
        self.overlap[0] = self.C0.conj().T @ self.C
        self.pop10to01 = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.pop01to10 = np.zeros((self.num_steps, 1), dtype=np.complex128)
        self.energies[0] = self.E
        self.population = np.zeros((self.num_steps, self.num_l * self.num_r, self.num_l * self.num_r), dtype=np.complex128)
        self.population[0] = np.abs(self.C)**2
        with tqdm(total=self.num_steps - 1,
                  position=0,
                  colour='green',
                  leave=True) as pbar:
            for i in range(1, self.num_steps):
                params = self.update_params(lmbda[i])
                self.params = params
                self.c_l = self.c_l1
                self.c_r = self.c_r1
                self.H_prev = self.H.copy() # Save the previous Hamiltonian
                self.update_system(params)
                
                step()
                self.overlap[i] =   self.C0.conj().T @ self.C #
                self.population[i] = np.abs(self.C)**2
                # Calculate all overlaps with the logical states
                # |00>
                self.psi00_00_overlap[i] = np.vdot(psi00, self.psi00)
                self.psi00_01_overlap[i] = np.vdot(psi01, self.psi00)
                self.psi00_10_overlap[i] = np.vdot(psi10, self.psi00)
                self.psi00_11_overlap[i] = np.vdot(psi11, self.psi00)
                # |01>
                self.psi01_00_overlap[i] = np.vdot(psi00, self.psi01)
                self.psi01_01_overlap[i] = np.vdot(psi01, self.psi01)
                self.psi01_10_overlap[i] = np.vdot(psi10, self.psi01)
                self.psi01_11_overlap[i] = np.vdot(psi11, self.psi01)
                # |10>
                self.psi10_00_overlap[i] = np.vdot(psi00, self.psi10)
                self.psi10_01_overlap[i] = np.vdot(psi01, self.psi10)
                self.psi10_10_overlap[i] = np.vdot(psi10, self.psi10)
                self.psi10_11_overlap[i] = np.vdot(psi11, self.psi10)
                # |11>
                self.psi11_00_overlap[i] = np.vdot(psi00, self.psi11)
                self.psi11_01_overlap[i] = np.vdot(psi01, self.psi11)
                self.psi11_10_overlap[i] = np.vdot(psi10, self.psi11)
                self.psi11_11_overlap[i] = np.vdot(psi11, self.psi11)
                # Save the states evolution
                self.psi_00t[i] = self.psi00.copy()
                self.psi_01t[i] = self.psi01.copy()
                self.psi_10t[i] = self.psi10.copy()
                self.psi_11t[i] = self.psi11.copy()

                self.pop01to10[i] = np.abs(self.psi01[self.idx_10])**2
                self.pop10to01[i] = np.abs(self.psi10[self.idx_01])**2
                self.energies[i] = self.E
                pbar.set_description(
                    # rf'[Commutator = {np.linalg.norm(self.commutator):.3e}]'
                    rf'[Entropy: {np.array([self.S[0], self.S[1], self.S[2], self.S[3]])}]'
                )
                pbar.update(1)
        overlaps = {
            '|00⟩': [self.psi00_00_overlap, self.psi00_01_overlap, self.psi00_10_overlap, self.psi00_11_overlap],
            '|01⟩': [self.psi01_00_overlap, self.psi01_01_overlap, self.psi01_10_overlap, self.psi01_11_overlap],
            '|10⟩': [self.psi10_00_overlap, self.psi10_01_overlap, self.psi10_10_overlap, self.psi10_11_overlap],
            '|11⟩': [self.psi11_00_overlap, self.psi11_01_overlap, self.psi11_10_overlap, self.psi11_11_overlap],
        }
        state_evo = {
            '|00⟩': self.psi_00t,
            '|01⟩': self.psi_01t,
            '|10⟩': self.psi_10t,
            '|11⟩': self.psi_11t,
        }
        return self.overlap, self.energies, overlaps, state_evo


    def _find_VN_entropies(self, rho):
        """Find entropy from reduced density matrix"""
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log2(eigs + 1e-15))    
    def _make_density_matrix(self, C):
        # self._rho = np.zeros((self.num_l ** 2, self.num_l ** 2), dtype=np.complex128)
        # for n in range(2):
        #     self._rho += np.outer(C[n], np.conj(C[n]).T)
        self._rho = np.outer(C, np.conj(C))
    def run_parameter_change(self):
        lmbda = self._lambda_t(self.t)
        num_steps = len(lmbda)
        C_matrix = np.zeros((num_steps, *self.C.shape), dtype=np.complex128)
        # C_matrix[0] = self.C0.copy()
        energies = np.zeros((num_steps, *self.E0.shape))
        # energies[0] = self.E0.copy()
        for i in tqdm(range(num_steps)):
            params = self.update_params(lmbda[i])
            self.update_system(params)
            self.S = np.zeros(self.num_l ** 2)
            for j in range(self.num_l ** 2):
                self._make_density_matrix(self.C[:,j]) # Each column is the energy eigenstates
                rho = np.trace(self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0]), axis1=0, axis2=2)
                # and then entropy
                self.S[j] = self._find_VN_entropies(rho)
            C_matrix[i] = self.C.copy()
            energies[i] = self.E.copy()

        return C_matrix, energies, num_steps

    
    def update_params(self, lmbda):
        """Update the parameters of the system."""
        params = (1 - lmbda) * np.array(self.params1) + lmbda * np.array(self.params2)
        return params

    
    def _plot_overlap(self, overlap=None, save=False):
        """Plot the overlap between the initial and current state."""
        matplotlib.style.use('seaborn-v0_8-deep')
        colors = sns.color_palette()
        # b = colors[0]
        # g = colors[1]
        # r = colors[2]
        if overlap is not None:
            self.psi00_00_overlap = overlap['psi00_00_overlap']
            self.psi00_01_overlap = overlap['psi00_01_overlap']
            self.psi00_10_overlap = overlap['psi00_10_overlap']
            self.psi00_11_overlap = overlap['psi00_11_overlap']
            self.psi01_00_overlap = overlap['psi01_00_overlap']
            self.psi01_01_overlap = overlap['psi01_01_overlap']
            self.psi01_10_overlap = overlap['psi01_10_overlap']
            self.psi01_11_overlap = overlap['psi01_11_overlap']
            self.psi10_00_overlap = overlap['psi10_00_overlap']
            self.psi10_01_overlap = overlap['psi10_01_overlap']
            self.psi10_10_overlap = overlap['psi10_10_overlap']
            self.psi10_11_overlap = overlap['psi10_11_overlap']
            self.psi11_00_overlap = overlap['psi11_00_overlap']
            self.psi11_01_overlap = overlap['psi11_01_overlap']
            self.psi11_10_overlap = overlap['psi11_10_overlap']
            self.psi11_11_overlap = overlap['psi11_11_overlap']



        
        n_steps = self.psi00_00_overlap.shape[0]
        t = np.arange(n_steps) * self.dt  # or your actual time array

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=find_figsize(1.2, 0.4))

        # # Initial |00⟩
        axs[0, 0].plot(t, np.abs(self.psi00_00_overlap)**2, label='|⟨00|ψ₀₀⟩|²')
        axs[0, 0].plot(t, np.abs(self.psi00_01_overlap)**2, label='|⟨01|ψ₀₀⟩|²')
        axs[0, 0].plot(t, np.abs(self.psi00_10_overlap)**2, label='|⟨10|ψ₀₀⟩|²')
        axs[0, 0].plot(t, np.abs(self.psi00_11_overlap)**2, label='|⟨11|ψ₀₀⟩|²')
        axs[0, 0].set_title('Initial |00⟩')
        axs[0, 0].set_ylabel('Probability')
        axs[0, 0].legend(fontsize='small')
        axs[0, 0].grid(True)

        # Initial |01⟩
        axs[0, 1].plot(t, np.abs(self.psi01_00_overlap)**2, label='|⟨00|ψ₀₁⟩|²')
        axs[0, 1].plot(t, np.abs(self.psi01_01_overlap)**2, label='|⟨01|ψ₀₁⟩|²')
        axs[0, 1].plot(t, np.abs(self.psi01_10_overlap)**2, label='|⟨10|ψ₀₁⟩|²')
        axs[0, 1].plot(t, np.abs(self.psi01_11_overlap)**2, label='|⟨11|ψ₀₁⟩|²')
        axs[0, 1].set_title('Initial |01⟩')
        axs[0, 1].legend(fontsize='small', loc='upper left')
        axs[0, 1].grid(True)

        # Initial |10⟩
        axs[1, 0].plot(t, np.abs(self.psi10_00_overlap)**2, label='|⟨00|ψ₁₀⟩|²')
        axs[1, 0].plot(t, np.abs(self.psi10_01_overlap)**2, label='|⟨01|ψ₁₀⟩|²')
        axs[1, 0].plot(t, np.abs(self.psi10_10_overlap)**2, label='|⟨10|ψ₁₀⟩|²')
        axs[1, 0].plot(t, np.abs(self.psi10_11_overlap)**2, label='|⟨11|ψ₁₀⟩|²')
        axs[1, 0].set_title('Initial |10⟩')
        axs[1, 0].set_xlabel('Time step')
        axs[1, 0].set_ylabel('Probability')
        axs[1, 0].legend(fontsize='small', loc='upper left')
        axs[1, 0].grid(True)

        # Initial |11⟩
        axs[1, 1].plot(t, np.abs(self.psi11_00_overlap)**2, label='|⟨00|ψ₁₁⟩|²')
        axs[1, 1].plot(t, np.abs(self.psi11_01_overlap)**2, label='|⟨01|ψ₁₁⟩|²')
        axs[1, 1].plot(t, np.abs(self.psi11_10_overlap)**2, label='|⟨10|ψ₁₁⟩|²')
        axs[1, 1].plot(t, np.abs(self.psi11_11_overlap)**2, label='|⟨11|ψ₁₁⟩|²')
        axs[1, 1].set_title('Initial |11⟩')
        axs[1, 1].set_xlabel('Time step')
        axs[1, 1].legend(fontsize='small')
        axs[1, 1].grid(True)

        # # Stat evolution
        # axs[2, 0].plot(t, np.abs(self.psi_00t)**2)
        # axs[2, 1].plot(t, np.abs(self.psi_01t)**2)
        # axs[3, 0].plot(t, np.abs(self.psi_10t)**2)
        # axs[3, 1].plot(t, np.abs(self.psi_11t)**2)
        # axs[3,1].legend(['00', '01', '02', '03', '10', '11', '12', '13', '20', '21'], ncol=2, loc='upper right', fontsize='small')

        def save_data(file=None):
            """Save the data to a file."""
            import pickle
            data = {
                't': t,
                'overlap': self.overlap,
                'energies': self.energies,
                'population': self.population,
                'psi00_00_overlap': self.psi00_00_overlap,
                'psi00_01_overlap': self.psi00_01_overlap,
                'psi00_10_overlap': self.psi00_10_overlap,
                'psi00_11_overlap': self.psi00_11_overlap,
                'psi01_00_overlap': self.psi01_00_overlap,
                'psi01_01_overlap': self.psi01_01_overlap,
                'psi01_10_overlap': self.psi01_10_overlap,
                'psi01_11_overlap': self.psi01_11_overlap,
                'psi10_00_overlap': self.psi10_00_overlap,
                'psi10_01_overlap': self.psi10_01_overlap,
                'psi10_10_overlap': self.psi10_10_overlap,
                'psi10_11_overlap': self.psi10_11_overlap,
                'psi11_00_overlap': self.psi11_00_overlap,
                'psi11_01_overlap': self.psi11_01_overlap,
                'psi11_10_overlap': self.psi11_10_overlap,
                'psi11_11_overlap': self.psi11_11_overlap
            }
            if file is None:
                file = 'data/time_evolution_data_for_4_basefunction_2306_SWAP_CN.pkl'
            try:
                with open(file, 'wb') as f:
                    pickle.dump(data, f)
            except:
                breakpoint()
        plt.tight_layout(rect=[0,0,0.95,1])
        # fig.subplots_adjust()
        if save:
            save_data() 
            plt.savefig('../doc/figs/time_evolution_4_basefunctions_2306_SWAP.pdf')
        plt.show()


if __name__ == '__main__':
    # # Need to optimize again I believe, as this is done with a different basis set.
    # params1 = [89.05318891, 70.55927467, 46.69855885, 5., 66.12799976]
    # params2 = [88.9432289, 70.22836355, 11.09930862, 11.17075134, 70.55100164]

    # params1 = [ 95.70685722 , 76.66934364 , 54.2000149  , 10.        , 189.46694742]
    # params2 = [77.13829616,  86.31735811,  34.57979308,  34.45182811, 188.08796131]

    # ## Without fucked interactoin
    # params1 = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
    # params2 = [73.44693037, 71.99175625 ,29.16144963 ,29.16767609, 42.79831711]

    # # 02.06
    # # params_1 = ...
    # # params_2 = [47.96777414, 50.42158537, 13.19020138, 13.15246072, 16.37341536]

    # ## 03.06
    # params2 = [63.68144808, 43.05416999 ,10.69127355 ,10.90371128, 16.02656697]
    # params1 = [61.48354464, 37.12075554, 22.36820274, 8.1535532, 16.58949561]

    # # 10.06 fixed some minor scaling error in coulomb interaction
    # params1 = [65.00091526, 38.39692472, 25.00070405,  8.72380963, 18.49693257]
    # params2 = [76.50992129, 48.1146782 , 13.351184 ,  13.68434784 ,18.19766692]

    # # 10.06 - l = 2
    # # params2 = [50, 50, 20, 20, 50]
    # # params1 = [52.49089987, 53.62683999, 11.07673065, 16.52311111, 49.65248418]

    # # 10.06 2nd
    # params_II = [76.73049659 ,48.73421941 ,13.57163477 ,13.81777894 ,16.64430113]
    # params_I  = [80.  ,       67.73885159 ,30.  ,       22.60731225 ,16.65149379]


    # 10.06 3rd
    # params_II =  [76.68721384, 48.86793519, 13.8406338  ,14.09016223 ,16.64768406]
    params_II = [73.01819495 ,63.7467338 , 23.5483224 , 23.84514281 ,16.22846972]
    params_I = [73.34278003, 59.58514677, 31.82915013 ,21.00838547, 16.66399601]



    # 11.06 
    params_II = [29.25989332 ,32.39735744  ,5.06592519 , 5.03878633 , 6.54364108]
    params_I = [30.9241989,  32.44680423 , 2.00231508 , 6.04736381 , 6.48452165]

    # 11.06: l=2
    params_II = [50.84165245 ,50.83516511, 10.02227946, 10.02234144,  5.00000169]
    params_I = [49.25366164 ,52.01135821 , 9.99855117,  4.87048312 , 6.74518928]

    # 13.06 l=2, optimized within C_I basis
    params_II = [49.34681143, 50.95476102 , 9.16921973 , 9.1547418 ,  5.        ]
    params_I = [48.514703 ,  50.90995246 ,10.29958596  ,9.08639849  ,5.        ]
    # Supposedly less off-diagonal leakage 
    params_II = [43.1724973,  45.95418964 , 2.00245395 , 2.    ,      5.        ]
    params_I = [43.04984933, 46.04854472,  2.01756909 , 2.81522255 , 5.00399675]


    # 17.06 Optimized with C_I basis
    # params_II = [80.23274818 ,80.97044973 , 2.00191414 , 2.5022086  , 5.        ]
    # params_I = [48.09036408 ,47.25786891, 13.61383221,  1.48153242 ,17.46706477]

    # 17.06 artificial coulomb interaction (a = 10)
    # params_II = [22.57803111, 28.42655626 , 3.     ,     3.1781328 ,  5.42763218]

    # 18.06 l=2
    # params_I = [39.94917866, 40.08475402  ,9.44196147 , 8.48598234 ,10.30854679]
    # params_II = [40.02035412, 41.97386199 , 6.99384561  ,7.01237375 , 9.96140797]

    # 18.06 l = 4
    # params_I = [50.61022309, 49.97873181, 15.15957671 ,14.7814281 , 25.        ]
    # params_II = [50.53251825, 51.4226878  ,14.31096688 ,14.27690273 ,24.51433339]
    # params_I = [50.61022309, 49.97873181, 15.15957671 ,14.7814281 , 25.        ]
    # params_II = [50.56567102, 55.19190111 ,14.44324624 ,14.34208258 ,24.97394011]
    # params_II = [48.94981732, 43.21488023,  8.95807028 , 7.46911853 , 5.1203213 ] # C_II
    # params_II = [61.79971733, 61.37898213,  6.65816402 ,14.49524261,  5.02628745]
    # params_II = [51.71090587 ,44.82893286 , 3.0057138 ,  3.50346097,  8.03724283] # J = 0.001, deltaE=0.003
    # #params-II[50.46023195 49.97528737 11.04692185 10.9523067  14.99927149] # Good mixing, but coupling term very small, t> 50k
    # #  [47.0945469  49.04711257  8.41262793  8.26698567 15.03532017] # Good mixing, coupling term 10^-4



    # 18.06 second run l = 4
    params_I = [62.17088395, 60.73364357 ,19.89474221 ,21.81940414, 15.        ]
    params_II = [62.97325982, 64.11742637 ,13.22714092 ,13.09781006 ,14.95744294]


    ins = time_evolution(params1=params_I, params2=params_II, integrator='CN', 
                         alpha=1.0,
                         t_max=10.0, dt=0.1,  ramp='cosine',
                         num_lr=4,
                         plateau=19715,
                         ramp_up=500,
                         ramp_down=500,
                         chill_after=1500,
                         chill_before=10,)
    # import pickle
    # with open('data/time_evolution_data_for_4_basefunction_1806.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # ins._plot_overlap(overlap=data)
    # breakpoint()
    # exit()

    ins._evolve()
    ins._plot_overlap(save=False)
    # fidelity
    psi0 = np.column_stack([ ins.C0[:,0], ins.C0[:,1], ins.C0[:,4], ins.C0[:,5]])
    psit = np.column_stack([ins.psi00, ins.psi01, ins.psi10, ins.psi11])
    U_log = psi0.conj().T @ psit
    gate_data = {
        'C0': ins.C0,
        'psi00': ins.psi00,
        'psi01': ins.psi01,
        'psi10': ins.psi10,
        'psi11': ins.psi11,
        'U_log': U_log,
    }
    def classical_fidelity(U_log, U_ideal):
        P_log = np.abs(U_log)**2
        P_targt = np.abs(U_ideal)**2

        F_j = np.sum(np.sqrt(P_log * P_targt), axis=0)
        F_classical = np.mean(F_j)
        return F_classical

    import pickle
    with open('data/SWAP_gate_2306.pkl', 'wb') as f:
        pickle.dump(gate_data, f)
    try:
        psi0 = np.column_stack([ ins.C0[:,0], ins.C0[:,1], ins.C0[:,4], ins.C0[:,5]])
        psit = np.column_stack([ins.psi00, ins.psi01, ins.psi10, ins.psi11])
        U_ideal = np.array([ [1,0,0,0], [0,0,1,0], [0,1,0,0],[0,0,0,1]]) # SWAP
        # U_ideal = np.array([ [1,0,0,0], [0,(1+1j)/2,(1-1j)/2,0], [0,(1-1j)/2,(1+1j)/2,0],[0,0,0,1]]) # sqrt(SWAP)
        sq2U = np.sqrt(np.abs(U_ideal)**2)
        U_log = psi0.conj().T @ psit
        target = [0, 2, 1, 3]  # target permutation for |00⟩, |01⟩, |10⟩, |11⟩
        olap = np.trace(U_ideal.conj().T @ U_log)
        F_op = 0.25 * np.abs(olap)**2
        F_avg = (np.abs(olap)**2 + 4) / 20
        probs = abs(U_log[target, np.arange(4)])**2
        F_classical = probs.mean()
        print(f"Operator fidelity: {F_op:.4f}, Classical fidelity: {F_classical:.4f}, Average fidelity: {F_avg:.4f}")
        print(f"Eigenstate populatation, psi_1: {abs(ins.C[:,1])**2}, psi_2: {abs(ins.C[:,2])**2}, psi_3: {abs(ins.C[:,3])**2}, psi_0: {abs(ins.C[:,0])**2}")
    except: 
        breakpoint()
    breakpoint()
    exit()
    # # C, e, n = ins.run_parameter_change()
    # N = C.shape[1]
    # idx_00 = 0*ins.num_r + 0
    # idx_01 = 0*ins.num_r + 1
    # idx_10 = 1*ins.num_r + 0
    
    # overlap = np.zeros((n, N, N))
    # for t in range(n):
    #     M = ins.C0.conj().T @ C[t]
    #     overlap[t] = np.abs(M)**2
    # over01_0 = overlap[0, idx_01, :]    # shape (N,)
    # over10_0 = overlap[0, idx_10, :]
    # eig01 = np.argmax(over01_0)         # index j0 so that ψ_j0(0) ≈ |01⟩
    # eig10 = np.argmax(over10_0)         # index j1 so that ψ_j1(0) ≈ |10⟩
    # steps = np.arange(n)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    # # Panel 1: how |01⟩ populates its own eigenvector vs the other (“swapped”) one
    # ax1.plot(steps, overlap[:, idx_01, eig01], label="|01⟩→ψ₍01₎(t)")
    # ax1.plot(steps, overlap[:, idx_01, eig10], label="|01⟩→ψ₍10₎(t)")
    # ax1.set_ylabel("Pop. from |01⟩")
    # ax1.set_ylim(0, 1.05)
    # ax1.legend(fontsize="small")
    # ax1.set_title("Mixing of |01⟩ ↔ |10⟩")

    # # Panel 2: same for |10⟩
    # ax2.plot(steps, overlap[:, idx_10, eig10], label="|10⟩→ψ₍10₎(t)")
    # ax2.plot(steps, overlap[:, idx_10, eig01], label="|10⟩→ψ₍01₎(t)")
    # ax2.set_ylabel("Pop. from |10⟩")
    # ax2.set_ylim(0, 1.05)
    # ax2.legend(fontsize="small")

    # # Panel 3: lowest three eigenvalues vs step
    # for j in (0, 1, 2):
    #     ax3.plot(steps, e[:, j], label=f"E_{j}")
    # ax3.set_ylabel("Eigenvalues")
    # ax3.set_xlabel("Parameter Step")
    # ax3.legend(fontsize="small", ncol=3)
    # ax3.set_title("Lowest three eigenvalues")
    # plt.show()

    # # fig, ax = plt.subplots(4,1 )
    # # ax[1].plot(steps, np.abs((ins.C0.conj().T @ C)[:,:,1]) ** 2)
    # # ax[0].plot(steps, np.abs((ins.C0.conj().T @ C)[:,:,0]) ** 2)
    # # ax[2].plot(steps, np.abs((ins.C0.conj().T @ C)[:,:,4]) ** 2)
    # # ax[1].plot(steps, overlap[:, ins])
    # # ax[3].plot(steps, e[:,:6])
    # # ax[0].set_ylim(0, 1.1)
    # # ax[1].set_ylim(0, 1.1)
    # # ax[2].set_ylim(0, 1.1)
    # # ax[0].legend(['00', '01', '02', '03', '10', '02'])
    # # plt.show()

    # exit()
    # ins.run_parameter_change()
    # breakpoint()
    # exit()
