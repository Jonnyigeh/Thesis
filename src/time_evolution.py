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
                 params,
                 l=15, 
                 num_func=4, 
                 grid_length=400, 
                 num_grid_points=4_001, 
                 a=0.25, 
                 alpha=1.0,
                 dt=0.001,
                 t_max=10,
                 integrator='Euler-Cromer',
                 time_dependent_hamiltonian=False,
    ):
        self.params = params
        self.l = l
        self.num_func = num_func
        self.grid_length = grid_length
        self.num_grid_points = num_grid_points
        self.a = a
        self.alpha = alpha
        self.dt = dt
        self.t_max = t_max
        self.integrator = integrator
        self.t = np.arange(0, t_max, dt)
        # try:
        #     assert len(self.t) < 200_001 # Safety check to avoid memory overflow
        # except AssertionError:
        #     raise ValueError(f"Time vector is too long. Please choose a smaller timestep.")
        self.num_steps = len(self.t)
        # Find the initial state of the system, at t=0.
        self._set_initial_system()
        self.psi = self.C.copy()
        self.time_dependent_hamiltonian = time_dependent_hamiltonian
    
    
    def _evolve(self):
        """Evolve the system in time."""
        print(f"evolutionnn")
        self._integrate(integrator=self.integrator)
        
    
    def _set_initial_system(self):
        """Set the initial system for the time evolution.
        
        args:
            params (np.ndarray): The parameters for the initial system.
        """
        self.potential = qs.quantum_dots.one_dim.one_dim_potentials.MorsePotentialDW(
            *self.params,
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

    def _U_propagator_step(self):
        """Compute the time evolution of the system using the U-propagator."""
        U = scipy.linalg.expm(-1j * self.H * self.dt)
        self.psi = U @ self.psi

    def _euler_cromer_step(self):
        """Integrate the system using the Euler-Cromer method."""
        dpsi_dt = -1j * self.H @ self.psi # Compute the time derivative of the wavefunction
        self.psi += self.dt * dpsi_dt # Update the wavefunction
        # self.psi /= np.linalg.norm(self.psi) # Normalise the wavefunction

    def _crank_nicolson_step(self):
        """Integrate the system using the Crank-Nicolson method."""
        I = np.eye(self.H.shape[0])
        A = np.linalg.inv(I + 1j * self.dt / 2 * self.H) @ (I - 1j * self.dt / 2 * self.H)
        self.psi = A @ self.psi


    def _update_hamiltonian(self):
        """Update the Hamiltonian of the system by similarity transformation."""
        # self.H = self.C.conj().T @ self.H @ self.C
        self.H = self.psi.conj().T @ self.H @ self.psi
        # pass
        

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
            self._update_hamiltonian = lambda: None


        self.state_evolution = np.zeros((self.num_steps, self.num_func**2, self.num_func**2), dtype=np.complex128)
        self.energy_evolution = np.zeros((self.num_steps, self.num_func**2))
        # Integrate the system in time, and store the evolution of the system.
        self.state_evolution[0] = self.psi.copy()
        for i in tqdm(range(1, self.num_steps),
                      desc='Integrating',
                      total=self.num_steps - 1,
                      unit='step',
                      colour='green'):
            self._update_hamiltonian()
            self.step()
            self.state_evolution[i] = self.psi.copy()
            # self.energy_evolution[i] = self.E.copy

        breakpoint()

    def _visualize(self):
        """Visualize the time evolution of the system."""
        fig, ax = plt.subplots()
        # for i in range(3):
        #     ax.plot(self.t, np.abs(self.evolution[ :,i,0])**2, label=f'Probability amplitude {i}')
        ax.plot(self.t, np.abs(self.state_evolution[:,0,0])**2, label='|00>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,1,0])**2, label='|01>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,2,0])**2, label='|10>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,3,0])**2, label='|11>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,4,0])**2, label='|20>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,5,0])**2, label='|02>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,6,0])**2, label='|21>')
        # ax.plot(self.t, np.abs(self.state_evolution[:,7,0])**2, label='|12>')

        
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability amplitude')
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.show()


if __name__ == '__main__':
    # params = [45,  55,  35,  45, 100] # Measurement config (I) state parameters
    params = [59.20710113, 59.44370983 , 32.42994617 ,45.31466205, 100.35488958]
    ins = TimeEvolution(params=params, integrator='Crank-Nicolson', time_dependent_hamiltonian=True)
    ins._evolve()
    ins._visualize()


        