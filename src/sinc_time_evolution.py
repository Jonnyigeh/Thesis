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
                 a=0.25,
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

        self.set_system(params1)
        self.H0 = self.H.copy()
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
        self.eig_idx_10 =  np.argmax(np.abs(row_10)**2)
        self.eig_idx_01 = np.argmax(np.abs(row_01)**2)
        self.eig_idx_11 = np.argmax(np.abs(row_11)**2)
        # Set up the eigenveectors we wish to track
        self.psi00 = self.C0[:, 0] # |00⟩ # idx 0 * num_l + 0 = 0
        self.psi01 = self.C0[:, self.eig_idx_01] # |01⟩ # idx 0 * num_l + 1 = 1
        self.psi10 = self.C0[:, self.eig_idx_10] # |10⟩ # idx 1 * num_l + 0 = 4
        self.psi11 = self.C0[:, self.eig_idx_11] # |11⟩ # idx 1 * num_l + 1 = 5

        # self.S0 = np.zeros(len(self.E0))
        # for i in range(len(self.E0)):
        #     self._make_density_matrix(self.C0[:,i]) # Each column is the energy eigenstates
        #     rho = np.trace(self._rho.reshape(self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0], self.h_l.shape[0]), axis1=0, axis2=2)
        #     # and then entropy
        #     self.S0[i] = self._find_VN_entropies(rho)


    
    def set_system(self, params):
        # params = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
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
        self.c_l = self.bhs.c_l
        self.c_r = self.bhs.c_r
        self.h_l = self.c_l.conj().T @ self.h_l @ self.c_l
        self.h_r = self.c_r.conj().T @ self.h_r @ self.c_r
        # self.u_lr = np.einsum('ai, bj, ab, ak, bl -> ijkl', self.c_l.conj(), self.c_r.conj(), self.u_lr, self.c_l, self.c_r)
        # Test a new solution for finding u - since u_lr is not a 4-tesnro, so the above might be incorrect (it is convrted to a 2-tensor through the Sinc BHS)
        self.u_lr = self.c_l.conj().T @ self.u_lr @ self.c_r
        u_diag = self.u_lr.flatten()
        U = np.diag(u_diag)
        self.H_dist = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        # U = self.u_lr.reshape(*self.H_dist.shape)
        self.H = self.H_dist + U
        self.E, self.C = np.linalg.eigh(self.H)


    
    def update_system(self, params):
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
        # Test a new solution for finding u - since u_lr is not a 4-tesnro, so the above might be incorrect (it is convrted to a 2-tensor through the Sinc BHS)
        self.u_lr = self.c_l.conj().T @ self.u_lr @ self.c_r
        u_diag = self.u_lr.flatten()
        U = np.diag(u_diag)

        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r) 
        # U = self.u_lr.reshape(*H.shape)
        newH = H + U
        self.commutator = self.H @ newH - newH @ self.H
        self.H = newH
        # self.apply_pulse()
        self.E, _ = np.linalg.eigh(self.H)
        
    
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
        A = np.linalg.inv(I + 1j * self.dt / 2 * self.H) @ (I - 1j * self.dt / 2 * self.H)
        self.C = A @ self.C
    
    def _lambda_t(self, t):
        """Return the time-dependent parameter lambda."""
        # mask_start = t < self.t_start
        # mask1 = (t < self.ramp_time ) & (t >= self.t_start)
        # mask2 = (t >= self.ramp_time) & (t <= self.ramp_time)
        # mask3 = (t > self.ramp_time - self.t_end) & (t <= self.t_end + self.ramp_time)
        # mask_end = t > self.t_end + self.ramp_time

        # mask_start = t <= self.t_start
        # mask1 = (t > self.t_start) & (t <= self.ramp_time + self.t_start)
        # mask2 = (t > self.ramp_time + self.t_start) & (t <= self.t_max - self.t_end - self.t_end)
        # mask3 = (t > self.t_max - self.t_end - self.ramp_time) & (t <= self.t_max - self.t_end)
        # mask_end = t > self.t_max - self.t_end

        # lmbda = np.zeros_like(t)
        # if self.ramp=='cosine':
        #     lmbda[mask_start] = 0.0 # Stay still in config I
        #     lmbda[mask1] = 0.5 * (1 - np.cos(np.pi * (t[mask1] - self.t_start) / self.ramp_time)) # Up-ramp from config I -> config II
        #     lmbda[mask2] = 1 # Stay a while in config II
        #     lmbda[mask3] = 0.5 * (1 + np.cos(np.pi * (t[mask3] - (self.t_max - self.t_end - self.ramp_time)) / self.ramp_time)) # Down-ramp from config II -> config I
        #     lmbda[mask_end] = 0.0 # Stay still in config I
        # else:
        #     raise NotImplementedError(f"Ramp {self.ramp} not implemented.")
        lmbda = np.zeros_like(t)

        # masks
        m0 = (t <=  self.t1)
        m1 = (t >  self.t1) & (t <= self.t2)
        m2 = (t >  self.t2) & (t <= self.t3)
        m3 = (t >  self.t3) & (t <= self.t4)
        m4 = (t >  self.t4)

        # assign segments
        lmbda[m0] = 0.0
        # up‐ramp: cosine from 0→1
        lmbda[m1] = 0.5*(1 - np.cos(np.pi*(t[m1]-self.t1)/self.ramp_up))
        # plateau
        lmbda[m2] = 1.0
        # down‐ramp: cosine from 1→0
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
        # lmbda = self._lambda_t(self.t, self.ramp_time, self.t_max - self.ramp_time) # Time-dependent parameter
        # lmbda[-1] = 0.0 # Ensure that the last value is zero, until we fix that it is not :) TODO: Fix this
        # lmbda[0] = 0.0
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
        # Set some vectors to calculate the overlaps
        psi00 = self.psi00.copy()
        psi01 = self.psi01.copy()
        psi10 = self.psi10.copy()
        psi11 = self.psi11.copy()    



        # self.overlap = np.zeros((self.num_steps, *self.C0[0].shape), dtype=np.complex128)
        # self.psi_t = np.zeros((self.num_steps, *self.psi.shape), dtype=np.complex128)
        # self.psi_t[0] = self.psi.copy()
        self.energies = np.zeros((self.num_steps, *self.E.shape))
        # self.overlap[0] = self.C0.conj().T @ self.psi
        self.overlap[0] = self.C0.conj().T @ self.C
        self.energies[0] = self.E
        with tqdm(total=self.num_steps - 1,
                  position=0,
                  colour='green',
                  leave=True) as pbar:
            for i in range(1, self.num_steps):
                params = self.update_params(lmbda[i])
                self.params = params
                # self.apply_pulse()
                self.update_system(self.params)
                step()
                self.overlap[i] =   self.C0.conj().T @ self.C #
                # Calculate all overlaps with the logical states
                # |00>
                self.psi00_00_overlap[i] = np.vdot(psi00.conj().T, self.psi00)
                self.psi00_01_overlap[i] = np.vdot(psi01.conj().T, self.psi00)
                self.psi00_10_overlap[i] = np.vdot(psi10.conj().T, self.psi00)
                self.psi00_11_overlap[i] = np.vdot(psi11.conj().T, self.psi00)
                # |01>
                self.psi01_00_overlap[i] = np.vdot(psi00.conj().T, self.psi01)
                self.psi01_01_overlap[i] = np.vdot(psi01.conj().T, self.psi01)
                self.psi01_10_overlap[i] = np.vdot(psi10.conj().T, self.psi01)
                self.psi01_11_overlap[i] = np.vdot(psi11.conj().T, self.psi01)
                # |10>
                self.psi10_00_overlap[i] = np.vdot(psi00.conj().T, self.psi10)
                self.psi10_01_overlap[i] = np.vdot(psi01.conj().T, self.psi10)
                self.psi10_10_overlap[i] = np.vdot(psi10.conj().T, self.psi10)
                self.psi10_11_overlap[i] = np.vdot(psi11.conj().T, self.psi10)
                # |11>
                self.psi11_00_overlap[i] = np.vdot(psi00.conj().T, self.psi11)
                self.psi11_01_overlap[i] = np.vdot(psi01.conj().T, self.psi11)
                self.psi11_10_overlap[i] = np.vdot(psi10.conj().T, self.psi11)
                self.psi11_11_overlap[i] = np.vdot(psi11.conj().T, self.psi11)

                # self.overlap[i] = self.C0.conj().T @ self.psi #
                # self.psi_t[i] = self.psi.copy()
                self.energies[i] = self.E
                pbar.set_description(
                    rf'[Commutator = {np.linalg.norm(self.commutator):.3e}]'
                )
                pbar.update(1)
        overlaps = {
            '|00⟩': [self.psi00_00_overlap, self.psi00_01_overlap, self.psi00_10_overlap, self.psi00_11_overlap],
            '|01⟩': [self.psi01_00_overlap, self.psi01_01_overlap, self.psi01_10_overlap, self.psi01_11_overlap],
            '|10⟩': [self.psi10_00_overlap, self.psi10_01_overlap, self.psi10_10_overlap, self.psi10_11_overlap],
            '|11⟩': [self.psi11_00_overlap, self.psi11_01_overlap, self.psi11_10_overlap, self.psi11_11_overlap],
        }
        return self.overlap, self.energies, overlaps


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

    
    def _plot_overlap(self):
        """Plot the overlap between the initial and current state."""
        matplotlib.style.use('seaborn-v0_8')
        colors = sns.color_palette()
        b = colors[0]
        g = colors[1]
        r = colors[2]
        n_steps = self.psi00_00_overlap.shape[0]
        t = np.arange(n_steps)  # or your actual time array

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

        # Initial |00⟩
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
        axs[0, 1].legend(fontsize='small')
        axs[0, 1].grid(True)

        # Initial |10⟩
        axs[1, 0].plot(t, np.abs(self.psi10_00_overlap)**2, label='|⟨00|ψ₁₀⟩|²')
        axs[1, 0].plot(t, np.abs(self.psi10_01_overlap)**2, label='|⟨01|ψ₁₀⟩|²')
        axs[1, 0].plot(t, np.abs(self.psi10_10_overlap)**2, label='|⟨10|ψ₁₀⟩|²')
        axs[1, 0].plot(t, np.abs(self.psi10_11_overlap)**2, label='|⟨11|ψ₁₀⟩|²')
        axs[1, 0].set_title('Initial |10⟩')
        axs[1, 0].set_xlabel('Time step')
        axs[1, 0].set_ylabel('Probability')
        axs[1, 0].legend(fontsize='small')
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

        plt.tight_layout()
        plt.show()


        
        # fig, ax = plt.subplots(4, 1)
        # ax[0].plot(self.t, np.abs(self.overlap[:,:, 0])**2)
        # ax[1].plot(self.t, np.abs(self.overlap[:,:, 1])**2)
        # ax[2].plot(self.t, np.abs(self.overlap[:,:, 4])**2 )
        # pop_01_to_10 = abs(self.overlap[:, self.idx_01, self.eig_idx_10])**2
        # pop_10_to_01 = abs(self.overlap[:, self.idx_10, self.eig_idx_01])**2   
        # ax[3].plot(self.t, pop_01_to_10, color=b, label='|01⟩ to |10⟩')
        # ax[3].plot(self.t, pop_10_to_01, color=g, label='|10⟩ to |01⟩')
        # ax[0].set_ylim(0, 1.1)
        # ax[1].set_ylim(0, 1.1)
        # ax[2].set_ylim(0, 1.1)
        # ax[0].set_xlabel('Time')
        # ax[0].set_ylabel('Population')
        # ax[0].legend(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], ncol=2, loc='upper right', fontsize='small')
        # # fig, ax = plt.subplots(10, 1)
        # # ax[0].plot(self.t, np.abs(self.overlap[:, :, 0])**2, color=b)
        # # ax[1].plot(self.t, np.abs(self.overlap[:, :, 1])**2, color=g)
        # # ax[2].plot(self.t, np.abs(self.overlap[:, :, 2])**2, color=r)
        # # ax[3].plot(self.t, np.abs(self.overlap[:, :, 3])**2, color='orange')
        # # ax[4].plot(self.t, np.abs(self.overlap[:, :, 4])**2, color='purple')
        # # ax[5].plot(self.t, np.abs(self.overlap[:, :, 5])**2, color='brown')
        # # ax[6].plot(self.t, np.abs(self.overlap[:, :, 6])**2, color='pink')
        # # ax[7].plot(self.t, np.abs(self.overlap[:, :, 7])**2, color='gray')
        # # ax[8].plot(self.t, np.abs(self.overlap[:, :, 8])**2, color='cyan')
        # # ax[9].plot(self.t, np.abs(self.overlap[:, :, 9])**2, color='olive')
        # # ax[0].legend(['|00⟩', '|01⟩', '|02⟩', '|03>⟩', '|10>', '|11⟩', '|12⟩', '|13>', '|20⟩', '|21⟩',], ncol=2, loc='upper right', fontsize='small')
        # ax[0].set_ylim(0, 1.1)
        # ax[0].set_xlabel('Time')
        # ax[0].set_ylabel('Population')
        # fig.tight_layout()
        # plt.show()


if __name__ == '__main__':
    # Need to optimize again I believe, as this is done with a different basis set.
    params1 = [89.05318891, 70.55927467, 46.69855885, 5., 66.12799976]
    params2 = [88.9432289, 70.22836355, 11.09930862, 11.17075134, 70.55100164]

    params1 = [ 95.70685722 , 76.66934364 , 54.2000149  , 10.        , 189.46694742]
    params2 = [77.13829616,  86.31735811,  34.57979308,  34.45182811, 188.08796131]

    ## Without fucked interactoin
    params1 = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
    params2 = [73.44693037, 71.99175625 ,29.16144963 ,29.16767609, 42.79831711]

    # 02.06
    # params_1 = ...
    # params_2 = [47.96777414, 50.42158537, 13.19020138, 13.15246072, 16.37341536]

    ## 03.06
    params_2 = [63.68144808, 43.05416999 ,10.69127355 ,10.90371128, 16.02656697]
    params_1 = [61.48354464, 37.12075554, 22.36820274, 8.1535532, 16.58949561]


    ins = time_evolution(params1=params1, params2=params2, integrator='U', 
                         alpha=1.0,
                         t_max=10.0, dt=0.1,  ramp='cosine',
                         num_lr=4,
                         plateau=2,
                         ramp_up=4,
                         ramp_down=4,
                         chill_after=0.2,
                         chill_before=0.2,)
    ins._evolve()
    ins._plot_overlap()
    exit()
    # C, e, n = ins.run_parameter_change()
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
