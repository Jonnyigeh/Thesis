import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from tqdm import tqdm

# Local imports
import quantum_systems as qs
from bipartite_hartree import BipartiteHartreeSolver as BHS






class TimeEvolution:
    def __init__(self,
                 params1,
                 params2,
                 l=15, 
                 num_func=4, 
                 n_particles=2,
                 grid_length=400, 
                 num_grid_points=4_001, 
                 a=0.25, 
                 alpha=1.0,
                 dt=0.01,
                 t_max=1.0,
                 ramp_time=0.25,
                 integrator='Euler-Cromer',
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
        self.integrator = integrator
        self.t = np.arange(0, self.t_max, self.dt)
        # try:
        #     assert len(self.t) < 200_001 # Safety check to avoid memory overflow
        # except AssertionError:
        #     raise ValueError(f"Time vector is too long. Please choose a smaller timestep.")
        self.num_steps = len(self.t)
        # Find the initial state of the system, at t=0.
        self._set_initial_system(self.params)
        self.C0 = self.C.copy()
        self.time_dependent_hamiltonian = time_dependent_hamiltonian
    
    
    def _evolve(self):
        """Evolve the system in time."""
        print(f"evolutionnn")
        self._integrate(integrator=self.integrator)
        
    
    def _set_initial_system(self, params):
        """Set the initial system for the time evolution.
        
        args:
            params (np.ndarray): The parameters for the initial system.
        """
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
        # Construct, and diagonalize, the full Hamiltonian
        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        U =  self.u_lr.reshape(H.shape)
        self.H = H + U
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
        overlap = np.abs(C.conj().T @ C0[:, 2])**2
        return overlap

    def _U_propagator_step(self, t=0):
        """Compute the time evolution of the system using the U-propagator."""
        U = scipy.linalg.expm(-1j * self.H * self.dt)
        self.C = U @ self.C

    def _euler_cromer_step(self, t=0):
        """Integrate the system using the Euler-Cromer method."""
        dC_dt = -1j * self.H @ self.C # Compute the time derivative of the wavefunction
        self.C += self.dt * dC_dt # Update the wavefunction
        # self.C /= np.linalg.norm(self.C) # Normalise the wavefunction

    def _evolve_hamiltonian(self, t):
        """Return the Hamiltonian of the system."""
        self.H += np.ones((self.H.shape), dtype=np.complex128) * self._sine_pulse(t)

    def _crank_nicolson_step(self, t=0):
        """Integrate the system using the Crank-Nicolson method."""
        I = np.eye(self.H.shape[0])
        # H = self.H + np.ones((self.H.shape), dtype=np.complex128) * self._sine_pulse(t)
        # U = np.linalg.inv(I + 1j * self.dt / 2 * H) @ (I - 1j * self.dt / 2 * H)
        # self.C = U @ self.C

        A = np.linalg.inv(I + 1j * self.dt / 2 * self.H) @ (I - 1j * self.dt / 2 * self.H)
        self.C = A @ self.C


    def _sine_pulse(self, t, V0=0.0, omega=12*np.pi):
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
        rho = np.outer(C, np.conj(C).T)
        return rho

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
        # Initialize arrays to store the evolution of the system
        self.state_evolution = np.zeros((self.num_steps, self.num_func**2, self.num_func**2), dtype=np.complex128)
        self.gs_overlap = np.zeros((self.num_steps, self.num_func**2))
        # Integrate the system in time, and store the evolution of the system.
        self.state_evolution[0] = self.C.copy()
        self.gs_overlap[0] = self._groundstate_overlap(self.C)
        for i in tqdm(range(1, self.num_steps),
                      desc='Integrating',
                      total=self.num_steps - 1,
                      unit='step',
                      colour='green'):
            self.params = self._update_parameters(lmbda[i], self.config1, self.config2)
            self._update_system(self.params, self.t[i])
            # print(self.E)
            # print(self.C)
            # self._evolve_hamiltonian(self.t[i])
            # print(self.H)
            self.step()
            self.Smat = self.C0.conj().T @ self.C
            self.state_evolution[i] = self.Smat.copy()
            self.gs_overlap[i] = self._groundstate_overlap(self.C)
            # print(f"C0 {self.C0[:,0]}")


    def _lambda_t(self, t, t_upramp, t_downramp):
        """Return the time-dependent parameter lambda."""
        mask1 = t < t_upramp
        mask2 = (t >= t_upramp) & (t <= t_downramp)
        mask3 = t > t_downramp

        lmbda = np.zeros_like(t)
        lmbda[mask1] = 0.5 * (1 - np.cos(np.pi * t[mask1] / t_upramp)) # Up-ramp from config I -> config II
        lmbda[mask2] = 1 # Stay a while in config II
        lmbda[mask3] = 0.5 * (1 + np.cos(np.pi * (t[mask3] - t_downramp) / t_upramp)) # Down-ramp from config II -> config I
    
        return lmbda


    def _visualize(self):
        """Visualize the time evolution of the system."""
        fig, ax = plt.subplots(4, 1)
        # for i in range(3):
        #     ax.plot(self.t, np.abs(self.evolution[ :,i,0])**2, label=f'Probability amplitude {i}')
        ax[0].plot(self.t, self.gs_overlap, 'ko')
        # ax.plot(self.t, np.abs(self.state_evolution[:,0,0])**2, label='<Psi_0|00>')
        ax[1].plot(self.t, np.abs(self.state_evolution[:,:3,0])**2)
        ax[2].plot(self.t, np.abs(self.state_evolution[:,:3,1])**2)
        ax[3].plot(self.t, np.abs(self.state_evolution[:,:3,2])**2)


        
        
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Probability amplitude')
        ax[0].set_ylim(0, 1.1)
        ax[0].legend()
        ax[1].legend(['|00>', '|01>', '|10>', '|11>',' |20>',' |02>'])
        plt.show()

    def _change_configs(self):
        """Change the configuration of the system."""
        lmbda = 0
        num_steps = 30
        g_state = np.zeros((num_steps, self.num_func**2, self.num_func**2), dtype=np.complex128)
        C_matrix = np.zeros((num_steps, *self.C.shape), dtype=np.complex128)
        C_matrix[0] = self.C.copy()
        g_state[0] = self.C[:,0].copy()
        for i in range(1, num_steps):
            lmbda = (i+1) / num_steps
            params = self._update_parameters(lmbda, self.config1, self.config2)
            self.params = params
            self._update_system(params)
            self.S = np.zeros(self.num_func)
            for j in range(self.num_func):
                rho = self._make_density_matrix(self.C[:,j]) # Each column is the energy eigenstates
                rho = np.trace(rho.reshape(self.num_func, self.num_func, self.num_func, self.num_func), axis1=0, axis2=2)
                # and then entropy
                self.S[j] = self._find_VN_entropies(rho)
            rho = self._find_density_matrix(self.C)
            subrho = np.trace(rho.reshape(self.num_func, self.num_func, self.num_func, self.num_func), axis1=0, axis2=2)
            print(f"entropy for both particles?: {self._find_VN_entropies(subrho)}")
            print(f"Step {i+1}/{num_steps}")
            # print(f"Energy: {self.E}")
            print(f"params: {params}")
            print(f"Entropies: {self.S}")
            g_state[i] = self.C[:,0].copy()
            C_matrix[i] = self.C.copy()
            
        
        return C_matrix, g_state, num_steps

    def _update_parameters(self, lmbda=0, 
                           config_I=
                           [59.20710113, 59.44370983 , 32.42994617 ,45.31466205, 100.35488958], 
                           config_II=
                           [55.58210512, 73.97732761, 20.18436592, 19.75699802, 97.67955539]):
        """Update the parameters of the system, changing from config I into config II."""
        params = (1 - lmbda) * np.array(config_I) + lmbda * np.array(config_II)

        return params

    def _update_system(self, params, t=0):
        """Update the system with new parameters."""
        # self._set_initial_system(params)
        # self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
        #     *params,
        # )
        #  # Project new potential onto the fixed basis (keeping kinetic terms unchanged)
        # V_l_new = np.einsum('ia, ij, ja -> a', self.spf_l.conj(), self.potential.left_pot(self.grid), self.spf_l)
        # V_r_new = np.einsum('ia, ij, ja -> a', self.spf_r.conj(), self.potential.right_pot(self.grid), self.spf_r)

        # # Update single-particle Hamiltonians (keeping kinetic terms fixed)
        # self.h_l = np.diag(self.eigen_energies_l) + np.diag(V_l_new)
        # self.h_r = np.diag(self.eigen_energies_r) + np.diag(V_r_new)
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
        # Construct, and diagonalize, the full Hamiltonian
        H = np.kron(self.h_l, np.eye(*self.h_l.shape)) + np.kron(np.eye(*self.h_r.shape), self.h_r)
        U =  self.u_lr.reshape(H.shape)
        self.H = H + U
        self._evolve_hamiltonian(t)
        self.E, self.C = np.linalg.eigh(self.H)

        

        
        

if __name__ == '__main__':
    #### params = [45,  55,  35,  45, 100] # Measurement config (I) state parameters
    # params1 = [59.20710113, 59.44370983, 32.42994617, 45.31466205, 100.35488958] # Config I (Measurement config)
    params1 = [ 60.90228786 , 76.41079666  , 9.88551909 , 18.58238756 ,104.81157016] # Config I (Measurement config) from config II
    params2 = [55.58227515, 73.97731125, 20.18435972, 19.75699591, 99.9998964 ] # Config II
    params1 = np.asarray(params2) + 0.1
    # params2 = [55.58210512, 73.97732761, 20.18436592, 19.75699802, 97.67955539]
    # params2= [55, 73, 20., 19, 97]
    ins = TimeEvolution(params1=params1, params2=params2, integrator='Crank-Nicolson', time_dependent_hamiltonian=True)
    # breakpoint()
    ins._evolve()
    ins._visualize()
    exit()
    Cmat, g_states, num_steps = ins._change_configs()
    fig, ax = plt.subplots(1, 2)
    iters = np.arange(0, num_steps)
    ax[0].plot(iters, np.abs(Cmat[:,:,1])**2)
    ax[1].plot(iters, np.abs(Cmat[:,:,2])**2)
    plt.show()
    # ins._evolve()
    # ins._visualize()


        