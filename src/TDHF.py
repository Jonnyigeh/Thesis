import numpy as np
import quantum_systems as qs
import scipy
import scipy.integrate
import tqdm
import time

class TDHF_Solver:
    def __init__(
      self,
      system,
      potential,
      verbose=True      
    ):
        """Solve the time-dependent Hartree-Fock equations.
        
        Args:
            system (quantum_systems.GeneralOrbitalSystem): The system to solve.
            potential (quantum_system.potential): The potential function.
            verbose (bool): Print convergence information.
        """
    
        self.system = system
        self.potential = potential
        self.verbose = verbose
        try:
            self.omega = self.potential.omega
        except: 
            self.omega = 0.25
        self.h = self.system.h
        self.u = self.system.u
    
    def fill_fock_matrix(self, C, t=0):
        """Fill the Fock matrix.
        
        Computes the fock-operator matrix by usage of the one- and two-body matrix elements stored in quantum_system.GeneralOrbitalSystem,
        with the density matrix containing the eigenvectors found in the "C"-matrix.

        args:
            C  (np.ndarray): Matrix of (current) eigenvectors
        
        returns:
            fock (np.ndarray): Fock-operator matrix.
        """
        fock = np.zeros(self.system.h.shape, dtype=np.complex128)
        density_matrix = self.create_density_matrix(C)
        fock = np.einsum('ij, aibj->ab', density_matrix, self.system.u, dtype=np.complex128)        # Compute the two-body operator potential
        fock += self.system.h                                                                       # Add the one-body operator hamiltonian
        # fock += self.system.position[0] * self.laser_source(t)                                      # Add laser term
        
        return fock
    
    def laser_source(self, t=0, epsilon0=1.0):
        """Laser source term."""

        return epsilon0 * np.sin(8 * self.omega * t)
    
    def create_density_matrix(self, C):
        """Create the density matrix for the system.
        
        Computes the density matrix by summing the outer product of the eigenvectors found in the "C"-matrix.

        args:
            C (np.ndarray): Matrix of (current) eigenvectors.
        
        returns:
            density_matrix (np.ndarray): The density matrix.
        """
        density_matrix = np.zeros(C.shape, dtype=np.complex128)
        for i in range(self.system.n):
            density_matrix += np.outer(np.conj(C[:, i]), C[:, i])
        
        return density_matrix

    def find_obd(self, C):
        """Find the one-body density matrix.
        
        Computes the one-body density by summing the innerproduct of the single-particle wavefunctions with the density matrix.
        args:
            C (np.ndarray): Matrix of eigenvectors.
        
        returns:
            obd (np.ndarray): The one-body density.
        """
        density_matrix = self.create_density_matrix(C)
        obd = np.zeros(len(self.system.grid))
        obd = np.einsum('mi,mn,ni->i', np.conj(self.system.spf), density_matrix, self.system.spf, dtype=np.complex128)
        
        return obd

    def _solver_RHS(self, t, C):
        """Right-hand side in the differential equation for the TDHF C-matrix. Cdot(t) = -if(t)C(t)
        
        We reshape the result to fill our scipy.integrator scheme requirements.
        """
        C = np.reshape(C, self.system.h.shape)
        fock = self.fill_fock_matrix(C=C, t=t)
        C_dot = -1j * (fock @ C)
        C_dot = np.reshape(C_dot, len(C_dot)**2)

        return C_dot
    
    def _solve_TIHF(self,t0=0.0, max_iters=1000, epsilon=1e-18):
        """Solve the time-independent Hartree-Fock equations.
        
        Procedure:
        Set an initial guess for the eigenvectors. Here we've used the identity matrix, so the initial guess is the standard computational basis. [1, 0, 0, 0, ...] etc.
        Loop over the following steps:
        - Compute the Fock matrix using the initial guess, where a density matrix is created from the eigenvectors, and used to solve the one- and two-body matrix elements.
        - Diagonalize the Fock matrix to find the new eigenvectors.
        - Check for convergence, and if the convergence criterion is not met, repeat the procedure with the new eigenvectors. 

        args:
            t0          (float): Initial time.
            max_iters   (float): Maximum number of iterations.
            epsilon     (float): Convergence tolerance.

        returns: 
            energy (np.nparray): The energy of the system, i.e the eigenvalues of the system Hamiltonian.
            C (np.ndarray): The eigenvectors of the system Hamiltonian.
        """
        energy, C = scipy.linalg.eigh(np.eye(self.system.h.shape[0]))
        fock = self.fill_fock_matrix(C, t=t0)
        converged=False
        delta_E = 0.0
        e_list = []
        with tqdm.tqdm(total=max_iters,
                desc=rf"[Minimization progress, $\Delta E$ = {delta_E:.8f}]",
                position=0,
                colour="green",
                leave=True) as pbar:
            for i in range(max_iters):
                energy_new, C_ = scipy.linalg.eigh(fock)
                e_list.append(energy_new[0])
                delta_E = np.linalg.norm(energy_new - energy) / self.system.l
                pbar.set_description(
                    rf"[Optimization progress, $\Delta E$ = {delta_E:.8f}]"
                )
                pbar.update(1)
                if delta_E < epsilon:
                    if self.verbose:
                        print(f"Converged in {i} iterations.")
                    converged=True
                    break
                C = C_
                energy = energy_new
                fock = self.fill_fock_matrix(C, t=t0)

        if not converged:
            if self.verbose:
                raise RuntimeError(f"The solver failed to converged after maximum number (iters={max_iters}) of iterations was reached.")
            
        return energy, C
    
    def ground_state_overlap(self, C):
        """Find the ground state overlap with the current state."""
        C0 = self.C0
        overlap = np.abs(np.linalg.det(C[:,0:self.system.n].conj().T @ C0[:,0:self.system.n])) ** 2

        return overlap
    
    def solve(self, dt=0.001, t0=0.0, epsilon=1e-6):
        """Solve the time-dependent Hartree-Fock equations

        Solves the time-dependent Hartree-Fock equations by integrating the right-hand side of the differential equation for the C-matrix using Runge-Kutta 4 (5) (scipy.integrate.RK45).
        The integration is performed manually to allow for the reshaping of the C-matrix to fit the scipy.integrate.RK45 requirements.
        
        Args:
            dt (float): Time step.
            t0 (float): Initial time.
        """
        t_final = 2 * np.pi / self.omega
        if self.verbose:
            print(f"Finding the inital state..")
        energy, C0 = self._solve_TIHF(t0=t0, epsilon=epsilon)
        self.C0 = C0
        ground_state_overlap = []
        if self.verbose:
            print(f"Starting numeric integration to solve Time-dependent Hartree-Fock equations")
        t1 = time.time()
        n_iters = int((t_final - t0) / (2*dt))
        t = np.linspace(t0, t_final, n_iters)
        
        # Set very low tolerance to prevent early stoppage of time integration
        atol = 1e-12
        rtol = 1e-12
        integrator = scipy.integrate.RK45(self._solver_RHS, t0=t0, y0=np.reshape(C0, len(C0)**2), t_bound=t_final, atol=atol, rtol=rtol, max_step=dt)
        for i in tqdm.tqdm(range(n_iters),
                           colour='green'):
            integrator.step()
            C = integrator.y
            C = np.matrix(np.reshape(C, self.h.shape))
            ground_state_overlap.append(self.ground_state_overlap(C))
        
        onebody_density = self.find_obd(C)
        C_final = integrator.y
        t2 = time.time()
        if self.verbose:
            print(f'Time-dependent Hartree-Fock solver finished in {t2-t1:.7f} seconds!')

        return ground_state_overlap, onebody_density, C_final, t

    def visualize(self, obd, overlap, t, x=None):
        """Visualize the one-body density and the ground state overlap as a function of time.
        
        args:
            obd (np.ndarray): Array of one-body electron densities
            overlap (np.ndarray): Array of the evaluated ground-state overlap integrals (i.e <Psi(t)|Psi(0)>)
        """
        import matplotlib.pyplot as plt
        if x is None:
            x = np.linspace(-6, 6, len(obd))
        fig, ax = plt.subplots(ncols=1, nrows=2)
        ax[0].plot(x, obd)
        ax[0].set_title("One-body density")
        ax[0].set_xlabel("Position [a.u]")
        ax[1].plot(t, overlap)
        ax[1].set_title("Ground-state probability")
        ax[1].set_xlabel("Time [s]")
        plt.show()





if __name__ == "__main__":
    potential = qs.ODQD.HOPotential(omega=0.25)
    basis = qs.ODQD(l=10,  grid_length=10, num_grid_points=201, a=0.25, alpha=1.0, potential=potential)
    system = qs.GeneralOrbitalSystem(n=2, basis_set=basis, anti_symmetrize=True)
    tdhf_solver = TDHF_Solver(system, potential)
    overlap, obd, C, t = tdhf_solver.solve()
    tdhf_solver.visualize(obd=obd, overlap=overlap, t=t / (2 * np.pi))
    breakpoint()
