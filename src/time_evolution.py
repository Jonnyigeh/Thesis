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
from bipartite_hartree import BipartiteHartreeSolver as BHS






class TimeEvolution:
    def __init__(self,
                 params1,
                 params2,
                 l=25, 
                 num_func=4, 
                 n_particles=2,
                 grid_length=400, 
                 num_grid_points=4_001, 
                 a=0.25, 
                 alpha=1.0,
                 dt=0.01,
                 t_max=1.0,
                 ramp_time=0.25,
                 ramp='cosine',
                 integrator='Euler-Cromer',
                 hartree=False,
                 time_dependent_hamiltonian=False,
    ):
        self.params = params1
        self.config1 = params1
        self.config2 = params2
        self.l = l
        self.num_func = num_func
        self.n_particles = n_particles
        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.a = a
        self.alpha = alpha
        self.dt = dt
        self.t_max = t_max 
        self.ramp_time = ramp_time * t_max
        self.ramp = ramp
        self.integrator = integrator
        self.hartree = hartree
        self.t = np.arange(0, self.t_max, self.dt)
        # try:
        #     assert len(self.t) < 200_001 # Safety check to avoid memory overflow
        # except AssertionError:
        #     raise ValueError(f"Time vector is too long. Please choose a smaller timestep.")
        self.num_steps = len(self.t)
        # Find the initial state of the system, at t=0.
        self._set_initial_system()
        self.C0 = self.C.copy()#np.eye(*self.C.shape)
        self.H0 = self.H.copy()
        self.E0 = self.E.copy()
        self.time_dependent_hamiltonian = time_dependent_hamiltonian
        self.psi = self.C[:,1]
        order = np.arange(self.C0.shape[1])
        map = np.argmax(np.abs(self.C0), axis=0)
        new_order = np.zeros_like(order)
        for d in order:
            j = np.where(map == d)[0]
            if j.size > 1:
                print(f'Dupe found at {d}, j: {j}')
                continue
            if j.size == 0:
                print(f'basis function {d} not found in map')
                continue
            new_order[d] = j
        self.C0 = self.C0[:, new_order]
        self.C = self.C[:, new_order]
    
    
    def _evolve(self):
        """Evolve the system in time."""
        print(f"evolutionnn")
        self._integrate(integrator=self.integrator)
        
    
    def _set_initial_system(self, params=None):
        """Set the initial system for the time evolution.
        
        args:
            params (np.ndarray): The parameters for the initial system.
        """
        # params = [60.90228786 , 76.41079666  , 9.88551909 , 18.58238756 ,104.81157016] # Measurement config
        # params = [65.91466548, 44.08203839, 23.7505483, 24.73394647, 53.67108254]
        # params = [66.57505943, 42.6933204,  23.14104696, 13.03994431, 63.78797277] # Different measurement config
        params = [66.57505943, 42.6933204,  23.14104696, 13.03994431, 63.78797277]# no hartree measurement config
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
        )
        self.basis = qs.ODMorse(
            l=self.l,
            grid_length=self.grid_length,
            num_grid_points=self.num_grid_points,
            alpha=self.alpha,
            _a=self.a,
            potential=self.potential,
        )
        if self.hartree:
            self.bhs_solver = BHS(
                h_l=self.basis.h_l,
                h_r=self.basis.h_r,
                u_lr=self.basis._ulr,
                num_basis_l=self.num_func,
                num_basis_r=self.num_func,
            )

            self.e_l, self.c_l, self.e_r, self.c_r = self.bhs_solver.solve()
            self.h_l = self.c_l.conj().T @ self.basis.h_l @ self.c_l
            self.h_r = self.c_r.conj().T @ self.basis.h_r @ self.c_r
            self.u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', self.c_l.conj(), self.c_r.conj(), self.basis._ulr, self.c_l, self.c_r)
        else: 
            self.h_l = self.basis.h_l
            self.h_r = self.basis.h_r
            self.u_lr = self.basis._ulr
            self.e_l, self.c_l = np.linalg.eigh(self.h_l)
            self.e_r, self.c_r = np.linalg.eigh(self.h_r)
        # Construct, and diagonalize, the full Hamiltonian
        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        self.U =  self.u_lr.reshape(H.shape)
        self.H = H + self.U
        self.E, self.C = np.linalg.eigh(self.H)        

    def _find_density_matrix(self, C):
        """Find the density matrix of the system."""
        density_matrix = np.zeros(C.shape, dtype=np.complex128)
        for i in range(self.n_particles):
            density_matrix += np.outer(C[:, i], C[:, i].conj())
        return density_matrix

    def _groundstate_overlap(self, C):
        """Find the overlap between the current state and the ground state."""
        C0 = self.C0
        # overlap = np.abs(C[:,2][:,np.newaxis].conj().T @ C0[:, 2][:,np.newaxis])**2
        overlap = np.abs(C.conj().T @ C0[:,0])**2
        return overlap

    def _U_propagator_step(self, t=0):
        """Compute the time evolution of the system using the U-propagator."""
        U = scipy.linalg.expm(-1j * self.H * self.dt)
        UUdag = U @ U.conj().T
        UdagU = U.conj().T @ U
        assert np.allclose(UUdag, np.eye(*UUdag.shape)), "U is not unitary"
        assert np.allclose(UdagU, np.eye(*UdagU.shape)), "Udag is not unitary"
        self.C = U @ self.C
        self.psi = U @ self.psi

    def _euler_cromer_step(self, t=0):
        """Integrate the system using the Euler-Cromer method."""
        dC_dt = -1j * self.H @ self.C # Compute the time derivative of the wavefunction
        self.C += self.dt * dC_dt # Update the wavefunction
        # self.C /= np.linalg.norm(self.C) # Normalise the wavefunction

    def _evolve_hamiltonian(self, t):
        """Return the Hamiltonian of the system."""
        # self.H += np.ones((self.U.shape)) * self._sine_pulse(t)
        perturbation = np.zeros_like(self.H)
        perturbation[1,2] = perturbation[2,1] = self._sine_pulse(t)
        self.H += perturbation
        
    def _crank_nicolson_step(self, t=0):
        """Integrate the system using the Crank-Nicolson method."""
        I = np.eye(self.H.shape[0])
        # H = self.H + np.ones((self.H.shape), dtype=np.complex128) * self._sine_pulse(t)
        # U = np.linalg.inv(I + 1j * self.dt / 2 * H) @ (I - 1j * self.dt / 2 * H)
        # self.C = U @ self.C

        A = np.linalg.inv(I + 1j * self.dt / 2 * self.H) @ (I - 1j * self.dt / 2 * self.H)
        self.C = A @ self.C

    def _sine_pulse(self, t, V0=5.0, omega=12*np.pi):
        """Apply a perturbative pulse to the system."""
        pulse = V0 * np.sin(omega * t)
        
        return pulse

    def _transform_hamiltonian(self, t=0):
        """Update the Hamiltonian of the system by similarity transformation."""
        # self.H = self.C.conj().T @ self.H @ self.C
        # self.H += np.ones((self.H.shape), dtype=np.complex128) * self._sine_pulse(t)
        # self.H += np.eye(*self.H.shape, dtype=np.complex128) * self._sine_pulse(t)
        self.H = self.C.conj().T @ self.H @ self.C
        # pass
        
    def _find_VN_entropies(self, rho):
        """Find entropy from reduced density matrix"""
        eigs = np.linalg.eigvalsh(rho)
        return -np.sum(eigs * np.log2(eigs + 1e-15))    
    def _make_density_matrix(self, C):
        """Make the density matrix of the system."""
        rho = np.outer(C, np.conj(C))
        return rho
    def _find_svd_entropies(self, C):
        """Find entropy from SVD of coefficient matrix"""
        entropies = np.zeros(12)
        for i in range(12):
            vals = (scipy.linalg.svdvals(C[:,i].reshape(*self.h_l.shape))) **2
            entropies[i] = - np.sum(vals * np.log2(vals + 1e-15))
        
        return entropies


    def _integrate(self, integrator):
        """Integrate the system in time."""

        # Set the integrator
        if integrator == 'Euler-Cromer':
            self.step = self._euler_cromer_step
        elif integrator == 'Crank-Nicolson':
            self.step = self._crank_nicolson_step
        elif integrator == 'U-propagator':
            self.step = self._U_propagator_step
        else:
            raise ValueError(f"Integrator {integrator} not implemented.")
        # Decide on Hamiltonian update
        if not self.time_dependent_hamiltonian:
            self._evolve_hamiltonian = lambda t: self.H

        lmbda = self._lambda_t(self.t, self.ramp_time, self.t_max - self.ramp_time) # Time-dependent parameter
        lmbda[-1] = 0.0 # Ensure that the last value is zero, until we fix that it is not :) TODO: Fix this
        lmbda[0] = 0.0
        # lmbda = np.ones_like(self.t)
        # lmbda[0] = 0.0
        # Initialize arrays to store the evolution of the system
        self.state_evolution = np.zeros((self.num_steps, *self.C.shape), dtype=np.complex128)
        self.gs_overlap = np.zeros((self.num_steps, self.C.shape[0]))
        # Integrate the system in time, and store the evolution of the system.
        olap = self.C0[:,1].conj().T @ self.psi
        Smat = self.C0.conj().T @ self.C
        self.state_evolution[0] = olap.copy() #Smat.copy()
        self.gs_overlap[0] = self._groundstate_overlap(self.C)
        self.energies = np.zeros_like(self.gs_overlap)
        self.energies[0] = self.E
        # for i in tqdm(range(1, self.num_steps),
        #               desc='Integrating',
        #               total=self.num_steps - 1,
        #               unit='step',
        #               colour='green'):

        with tqdm(total=self.num_steps - 1,
                  position=0,
                  colour='green',
                  leave=True) as pbar:
            for i in range(1, self.num_steps):
                self._evolve_hamiltonian(self.t[i])
                # if lmbda[i-1] == 1.0 and lmbda[i] == 1.0:
                #     self.step()
                #     self.entropy = self._find_svd_entropies(self.C)
                #     # print('entropy:', self.entropy[:4])
                #     # print('energies L:', self.e_l[1] - self.e_l[0], 'energies R:', self.e_r[1] - self.e_r[0])
                #     self.Smat = self.C0.conj().T @ self.C
                #     # print(np.round(np.abs(self.Smat[:4,:4])**2, 5))
                #     # self.state_evolution[i] = self.Smat.copy()
                #     self.gs_overlap[i] = self._groundstate_overlap(self.C)
                #     self.energies[i] = self.E
                #     # print('energy: ', self.E)
                #     # print('hartree energy: ', self.e_l[1] - self.e_l[0], self.e_r[1] - self.e_r[0])
                #     # print(np.round(np.abs(self.H[:6,:6])**2, 3))
                #     pbar.set_description(
                #     rf'[Commutator = {np.linalg.norm(commutator):.3e}]'
                #      )
                #     pbar.update(1)
                #     continue
                Hprev = self.H.copy()
                # self.params = self._update_parameters(lmbda[i], self.config1, self.config2)
                # self._update_system(self.params, self.t[i], lmbda[i])
                self.step()
                self.entropy = self._find_svd_entropies(self.C)
                commutator = self.H @ Hprev - Hprev @ self.H
                self.Smat = self.C0.conj().T @ self.C
                # self.overlap = self.C0[:,1].conj().T @ self.psi
                # self.state_evolution[i] = self.overlap.copy()
                # breakpoint()
                self.state_evolution[i] = self.Smat.copy()
                self.gs_overlap[i] = self._groundstate_overlap(self.C)
                self.energies[i] = self.E
                # print('energy: ', self.E)
                # print('hartree energy: ', self.e_l[1] - self.e_l[0], self.e_r[1] - self.e_r[0])
                # print('hartree energy: ', self.e_l, self.e_r)
                # print(np.round(np.abs(self.H[:6,:6])**2, 3))

                pbar.set_description(
                    rf'[Commutator = {np.linalg.norm(commutator):.3e}]'
                )
                pbar.update(1)
                # print(f"Commutator: {np.linalg.norm(commutator)}")
                # print('entropy:', self.entropy[:4])
                # print('energies L:', self.e_l[1] - self.e_l[0], 'energies R:', self.e_r[1] - self.e_r[0])
                # print(np.round(np.abs(self.Smat[:4,:4])**2, 5))
                # if i == 4:
                #     print(np.round(np.abs(self.H[:5, :5])**2, 3))
                #     exit()  
                # norm = np.linalg.norm(self.H - self.old_H)
                # print(f"Norm of Hi+1 - Hi: {norm}")
                # print(np.round(np.abs(self.H[:4, :4])**2, 4))
                # print(self.E)
                # print(self.C)
                # print(self.H)
                # self.state_evolution[i] = self.C.copy()
                # print(f"C0 {self.C0[:,0]}")


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
        elif self.ramp=='linear':
            lmbda[mask1] = 0#t[mask1] / t_upramp
            lmbda[mask2] = 1
            lmbda[mask3] = 0#1 - (t[mask3] - t_downramp) / t_upramp
        elif self.ramp=='erf':
            lmbda[mask1] = 0.5 * (1 + erf(t[mask1] / t_upramp))
            lmbda[mask2] = 1
            lmbda[mask3] = 0.5 * (1 + erf((t[mask3] - t_downramp) / t_upramp))

        return lmbda


    def _visualize(self):
        """Visualize the time evolution of the system."""
        fig, ax = plt.subplots(5, 1)
        # for i in range(3):
        #     ax.plot(self.t, np.abs(self.evolution[ :,i,0])**2, label=f'Probability amplitude {i}')
        # ax[0].plot(self.t, self.gs_overlap, 'ko')
        # ax.plot(self.t, np.abs(self.state_evolution[:,0,0])**2, label='<Psi_0|00>')
        ax[0].plot(self.t, np.abs(self.state_evolution[:,:6,0])**2)
        ax[1].plot(self.t, np.abs(self.state_evolution[:,:6,1])**2)
        ax[2].plot(self.t, np.abs(self.state_evolution[:,:6,2])**2)
        ax[3].plot(self.t, np.abs(self.state_evolution[:,:6,3])**2)
        ax[4].plot(self.t, self.energies[:,:4])
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Probability amplitude')
        ax[0].set_ylim(0, 1.1)
        ax[1].set_ylim(0, 1.1)
        ax[2].set_ylim(0, 1.1)
        ax[3].set_ylim(0, 1.1)

        ax[0].legend()
        ax[1].legend(['|00>', '|01>', '|10>', '|11>',' |20>',' |02>'])
        plt.show()

    def _change_configs(self):
        """Change the configuration of the system."""
        lmbda = self._lambda_t(self.t, self.ramp_time, self.t_max - self.ramp_time) # Time-dependent parameter

        num_steps = 100#len(lmbda)
        g_state = np.zeros((num_steps, self.num_func**2, self.num_func**2), dtype=np.complex128)
        C_matrix = np.zeros((num_steps, *self.C.shape), dtype=np.complex128)
        C_matrix[0] = self.C.copy()
        g_state[0] = self.C[:,0].copy()
        energies = np.zeros((num_steps, self.num_func **2))
        energies[0] = self.E
        for i in range(1, num_steps):
            params = self._update_parameters(lmbda[i], self.config1, self.config2)
            self.params = params
            self._update_system(params)
            energies[i], self.C = np.linalg.eigh(self.H)
            self.S = np.zeros(self.num_func)
            for j in range(self.num_func):
                rho = self._make_density_matrix(self.C[:,j]) # Each column is the energy eigenstates
                rho = np.trace(rho.reshape(self.num_func, self.num_func, self.num_func, self.num_func), axis1=0, axis2=2)
                # and then entropy
                self.S[j] = self._find_VN_entropies(rho)
            rho = self._find_density_matrix(self.C)
            subrho = np.trace(rho.reshape(self.num_func, self.num_func, self.num_func, self.num_func), axis1=0, axis2=2)
            # print(f"entropy for both particles?: {self._find_VN_entropies(subrho)}")
            # print(f"Step {i+1}/{num_steps}")
            # # print(f"Energy: {self.E}")
            # print(f"params: {params}")
            # print(f"Entropies: {self.S}")
            g_state[i] = self.C[:,0].copy()
            C_matrix[i] = self.C.copy()
            
        
        return C_matrix, g_state, num_steps, energies

    def _update_parameters(self, lmbda=0, 
                           config_I=
                           [59.20710113, 59.44370983 , 32.42994617 ,45.31466205, 100.35488958], 
                           config_II=
                           [55.58210512, 73.97732761, 20.18436592, 19.75699802, 97.67955539]):
        """Update the parameters of the system, changing from config I into config II."""
        params = (1 - lmbda) * np.array(config_I) + lmbda * np.array(config_II)

        return params

    def _update_system_constbasis(self, params, *kwargs):
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
        )
        self.grid = np.linspace(
            -self.grid_length/2, self.grid_length/2, self.num_grid_points
        )
        dx = self.grid[1] - self.grid[0]
        # Find the eigenbasis for each well separately, left = A, right = B - accounting for Dirichlet BC, i.e the WF should go to zero at the end-points
        V_l = np.clip(self.potential.left_pot(self.grid[1:-1]),0, 100) # clip potential at 100, more natural, also prevents bugs with infinite potential wall.
        V_r = np.clip(self.potential.right_pot(self.grid[1:-1]), 0, 100)
        self.V = V_l + V_r - np.min(V_l + V_r)
        h_l_diag = 1 / (dx**2) + V_l
        h_l_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        h_r_diag = 1 / (dx**2) + V_r
        h_r_off_diag = - 1 / (2 * dx**2) * np.ones(self.num_grid_points - 3)
        eps_l, C_l = scipy.linalg.eigh_tridiagonal(h_l_diag, h_l_off_diag, select="i", select_range=(0, self.l - 1))
        eps_r, C_r = scipy.linalg.eigh_tridiagonal(h_r_diag, h_r_off_diag, select="i", select_range=(0, self.l - 1))
        self._h_l = np.diag(eps_l)
        self._h_r = np.diag(eps_r)
        self.h_l = self.c_l.conj().T @ self.basis.h_l @ self.c_l
        self.h_r = self.c_r.conj().T @ self.basis.h_r @ self.c_r
        self.e_l, _ = np.linalg.eigh(self.h_l)
        self.e_r, _ = np.linalg.eigh(self.h_r)
        print('ikje transformrt: ', self.e_l, self.e_r)
        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        U = self.u_lr.reshape(H.shape)
        new_H = H + U
        self.H = self.C0.conj().T @ new_H @ self.C0
        self.E, _ = np.linalg.eigh(self.H)


    def _update_system(self, params, t=0, lmbda=0):
        """Update the system with new parameters."""
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *params,
        )
        self.basis = qs.ODMorse(
            l=self.l,
            grid_length=self.grid_length,
            num_grid_points=self.num_grid_points,
            alpha=self.alpha,
            _a=self.a,
            potential=self.potential,
        )
        if self.hartree:
            self.bhs_solver = BHS(
                h_l=self.basis.h_l,
                h_r=self.basis.h_r,
                u_lr=self.basis._ulr,
                num_basis_l=self.num_func,
                num_basis_r=self.num_func,
            )
            self.e_l, self.c_l, self.e_r, self.c_r = self.bhs_solver.solve()
            self.h_l = self.c_l.conj().T @ self.basis.h_l @ self.c_l
            self.h_r = self.c_r.conj().T @ self.basis.h_r @ self.c_r
            self.u_lr = np.einsum('ia, jb, ijkl, kc, ld -> abcd', self.c_l.conj(), self.c_r.conj(), self.basis._ulr, self.c_l, self.c_r)
        else:
            self.h_l = self.basis.h_l
            self.h_r = self.basis.h_r
            self.u_lr = self.basis._ulr
            self.e_l, self.c_l = np.linalg.eigh(self.h_l)
            self.e_r, self.c_r = np.linalg.eigh(self.h_r)
        # Construct, and diagonalize, the full Hamiltonian
        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        U =  self.u_lr.reshape(H.shape)
        new_H = H + U
        Hprev = self.H.copy()
        self.H = self.C0.conj().T @ new_H @ self.C0
        self.E, _ = np.linalg.eigh(self.H)
        # Find the commutator of the Hamiltonian
        # commutator = self.H @ Hprev - Hprev @ self.H
        # print(f"Commutator: {np.linalgc.norm(commutator)}")

        # print(np.round(np.abs(self.H[:5, :5])**2, 3))
        


        # ------------------- TEST ----------------------
        # np.random.seed(42)
        # Create a diagonal matrix
        # E = np.random.randn(0, 100, len(self.E0))
        # breakpoint()
        # H_diag = np.diag(self.E0)

        # # Create an off-diagonal random matrix
        # H_offdiag = np.zeros_like(H_diag)

        # # Ensure it's Hermitian: H = (H + H.T) / 2
        # H_offdiag = (H_offdiag + np.conj(H_offdiag).T) / 2

        # # Explicitly set strong coupling between eigenstates 1 and 2
        # H_offdiag[1, 2] = H_offdiag[2, 1] = 2  # Strong coupling

        # # Construct the full Hamiltonian
        # self.H = H_diag + H_offdiag
        # self.E, _ = np.linalg.eigh(self.H)

        
        

if __name__ == '__main__':
    #### params = [45,  55,  35,  45, 100] # Measurement config (I) state parameters
    # params1 = [59.20710113, 59.44370983, 32.42994617, 45.31466205, 100.35488958] # Config I (Measurement config)
    params1 = [ 60.90228786 , 76.41079666  , 9.88551909 , 18.58238756 ,104.81157016] # Config I (Measurement config) from config II
    params = [55.58227515, 73.97731125, 20.18435972, 19.75699591, 99.9998964 ] # Config II

    params = [66.57505943, 42.6933204,  23.14104696, 13.03994431, 63.78797277] #config I
    param = [65.91466548, 44.08203839, 23.7505483, 24.73394647, 43.67108254] # intermediat params?
    params2 = [65.91466548, 44.08203839, 23.7505483, 24.73394647, 53.67108254] #config II

    # without hartree
    params2 = [99.99999899, 69.70563878, 25.72016208, 26.33180028, 38.6367954 ]
    params1= [66.57505943, 42.6933204,  23.14104696, 13.03994431, 63.78797277]

    # params1 = np.asarray(params2) + 0.1
    # params2 = [55.58210512, 73.97732761, 20.18436592, 19.75699802, 97.67955539]
    # params2= [55, 73, 20., 19, 97]
    ins = TimeEvolution(params1=params1, params2=params2, integrator='U-propagator', ramp='cosine', time_dependent_hamiltonian=True)
    # breakpoint()
    ins._evolve()
    ins._visualize()
    exit()
    Cmat, g_states, num_steps, energy = ins._change_configs()
    iters = np.arange(0, num_steps)
    plt.plot(iters, energy[:,:4])
    plt.show()
    exit()


    fig, ax = plt.subplots(1, 2)
    ax[0].plot(iters, np.abs(Cmat[:,:,1])**2)
    ax[1].plot(iters, np.abs(Cmat[:,:,2])**2)
    plt.show()
    exit()
    # ins._evolve()
    # ins._visualize()


    # some_H = np.array([[ 19.6  ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,  75.965,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,  75.965,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   , 163.957,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   , 165.154,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   , 169.114,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   , 278.815,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #         283.474,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   , 292.176,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   , 293.774,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   , 440.427,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   , 446.277,   0.   ,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   , 450.653,   0.   ,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   , 631.105,
    #           0.   ,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #         635.748,   0.   ],
    #        [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ,
    #           0.   , 847.232]])