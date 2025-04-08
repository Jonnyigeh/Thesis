import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.special import erf
from tqdm import tqdm

# Local imports
import quantum_systems as qs
from sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_BHS


class time_evolution:
    def __init__(self,
                 params1,
                 params2,
                 l=25,
                 num_lr=4,
                 num_grid_points=400,
                 grid_length=200,
                 num_particles=2,
                 a=0.25,
                 alpha=1.0,
                 dt=0.01,
                 t_max=1.0,
                 ramp_time=0.2,
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
        self.t_max = t_max
        self.ramp_time = ramp_time
        self.ramp = ramp
        self.t = np.arange(0, self.t_max, self.dt)
        self.num_steps = len(self.t)
        self.integrator = integrator
        self.hartree = hartree
        self.grid = np.linspace(-grid_length/2, grid_length/2, num_grid_points)

        self.set_system(params1)
        self.H0 = self.H.copy()
        self.C0 = self.C.copy()
        self.E0 = self.E.copy()
        self.psi = (self.C0[:,1] + self.C0[:,2]) / np.sqrt(2)
    
    def set_system(self, params):
        # params = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
        params = self.params2
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
        self.E, self.C = np.linalg.eigh(self.H)

    
    def update_system(self, params):
        # Update one-body hamiltonian with new potential
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *self.params2,
        )
        new_V_l = np.clip(self.potential.left_pot(self.basis.left_grid), 0, 100)
        new_V_r = np.clip(self.potential.right_pot(self.basis.right_grid), 0, 100)
        self.h_l = self.basis.no_1bpot_h_l + np.diag(new_V_l) / self.basis.dx
        self.h_r = self.basis.no_1bpot_h_r + np.diag(new_V_r) / self.basis.dx
        self.u_lr = self.basis._ulr
        # And solve new Hartree equations 
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
        newH = H + U
        # self.H = self.C0.conj().T @ newH @ self.C0
        self.H = newH
        self.E, _ = np.linalg.eigh(self.H)
        
    
    def U_step(self):
        """Compute the time evolution of the system using the U-propagator."""
        U = scipy.linalg.expm(-1j * self.H * self.dt)
        UUdag = U @ U.conj().T
        UdagU = U.conj().T @ U
        assert np.allclose(UUdag, np.eye(*UUdag.shape)), "U is not unitary"
        assert np.allclose(UdagU, np.eye(*UdagU.shape)), "Udag is not unitary"
        # self.C = U @ self.C
        self.psi = U @ self.psi
    
    def crank_nicolson_step(self, t=0):
        """Integrate the system using the Crank-Nicolson method."""
        I = np.eye(self.H.shape[0])
        A = np.linalg.inv(I + 1j * self.dt / 2 * self.H) @ (I - 1j * self.dt / 2 * self.H)
        self.C = A @ self.C
    
    def _lambda_t(self, t, t_upramp, t_downramp):
        """Return the time-dependent parameter lambda."""
        mask1 = t < t_upramp
        mask2 = (t >= t_upramp) & (t <= t_downramp)
        mask3 = t > t_downramp

        lmbda = np.zeros_like(t)
        if self.ramp=='cosine':
            lmbda[mask1] = 0.5 * (1 - np.cos(np.pi * t[mask1] / t_upramp)) # Up-ramp from config I -> config II
            lmbda[mask2] = 1 # Stay a while in config II
            lmbda[mask3] = 0.5 * (1 + np.cos(np.pi * (t[mask3] - t_downramp) / t_upramp)) # Down-ramp from config II -> config I
        else:
            raise NotImplementedError(f"Ramp {self.ramp} not implemented.")
        return lmbda
    
    def _evolve(self):
        """Evolve the system for a single time step."""
        if self.integrator == 'U':
            step = self.U_step
        elif self.integrator == 'CN':
            step = self.crank_nicolson_step
        else:
            raise NotImplementedError(f"Integrator {self.integrator} not implemented.")

        lmbda = self._lambda_t(self.t, self.ramp_time, self.t_max - self.ramp_time) # Time-dependent parameter
        lmbda[-1] = 0.0 # Ensure that the last value is zero, until we fix that it is not :) TODO: Fix this
        lmbda[0] = 0.0
        self.overlap = np.zeros((self.num_steps, *self.C0.shape), dtype=np.complex128)
        self.energies = np.zeros((self.num_steps, *self.E.shape))
        self.overlap[0] = self.C.copy() #self.C0.conj().T @ self.C
        self.energies[0] = self.E
        for i in tqdm(range(1, self.num_steps),
                      desc='Integrating',
                      total=self.num_steps - 1,
                      unit='step',
                      colour='green'):
            params = self.update_params(lmbda[i])
            self.params = params
            self.update_system(self.params)
            step()
            self.overlap[i] = self.psi.copy()#self.C0.conj().T @ self.C
            self.energies[i] = self.E

        return self.overlap, self.energies


    def _find_VN_entropies(self, rho):
        """Find entropy from reduced density matrix"""
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log2(eigs + 1e-15))    
    def _make_density_matrix(self, C):
        self._rho = np.zeros((self.num_l ** 2, self.num_l ** 2), dtype=np.complex128)
        for n in range(2):
            self._rho += np.outer(C[n], np.conj(C[n]).T)
        # self._rho = np.outer(C, np.conj(C))
    def run_parameter_change(self):
        lmbda = self._lambda_t(self.t, self.ramp_time, self.t_max - self.ramp_time)
        num_steps = len(lmbda)
        C_matrix = np.zeros((num_steps, *self.C.shape), dtype=np.complex128)
        C_matrix[0] = self.C0.copy()
        energies = np.zeros((num_steps, *self.E0.shape))
        energies[0] = self.E0.copy()
        for i in tqdm(range(1, num_steps)):
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
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(self.t, np.abs(self.overlap[:,:, 0])**2)
        ax[1].plot(self.t, np.abs(self.overlap[:,:, 1])**2)
        ax[2].plot(self.t, np.abs(self.overlap[:,:, 2])**2 )
        ax[3].plot(self.t, self.energies[:,:6])
        ax[0].set_ylim(0, 1.1)
        ax[1].set_ylim(0, 1.1)
        ax[2].set_ylim(0, 1.1)
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Population')
        ax[0].legend(['00', '01', '02', '03', '10', '11'])

        plt.show()
if __name__ == '__main__':
    # Need to optimize again I believe, as this is done with a different basis set.
    params1 = [89.05318891, 70.55927467, 46.69855885, 5., 66.12799976]
    params2 = [88.9432289, 70.22836355, 11.09930862, 11.17075134, 70.55100164]

    params1 = [ 95.70685722 , 76.66934364 , 54.2000149  , 10.        , 189.46694742]
    params2 = [77.13829616,  86.31735811,  34.57979308,  34.45182811, 188.08796131]

    ## Without fucked interactoin
    params1 = [73.40600785, 71.10039648 ,31.6125873,  26.57511632, 42.47007681]
    params2 = [73.44693037, 71.99175625 ,29.16144963 ,29.16767609, 42.79831711]
    ins = time_evolution(params1=params1, params2=params2, integrator='U', alpha=1.0)
    # C, e, n = ins.run_parameter_change()
    # steps = np.arange(n)
    # fig, ax = plt.subplots(4,1 )
    # ax[0].plot(steps, np.abs((ins.C0.conj().T @ C)[:,:,0]) ** 2)
    # ax[1].plot(steps, np.abs((ins.C0.conj().T @ C)[:,:,1]) ** 2)
    # ax[2].plot(steps, np.abs((ins.C0.conj().T @ C)[:,:,2]) ** 2)
    # ax[3].plot(steps, e[:,:6])
    # ax[0].set_ylim(0, 1.1)
    # ax[1].set_ylim(0, 1.1)
    # ax[2].set_ylim(0, 1.1)
    # ax[0].legend(['00', '01', '10', '11', '20', '02'])
    # plt.show()

    # breakpoint()
    # exit()
    ins._evolve()
    ins._plot_overlap()
    breakpoint()

