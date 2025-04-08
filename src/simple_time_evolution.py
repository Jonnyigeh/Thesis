import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

# Define parameters
Delta = 1.0  # Coupling strength
v = 2.0  # Sweeping rate
t_values = np.linspace(-5, 5, 1000)  # Time range
dt = t_values[1] - t_values[0]  # Time step

# Define the Hamiltonian function
def H(t):
    epsilon = v * t
    return np.array([[epsilon, Delta], [Delta, -epsilon]])
Hnon = lambda t:  np.array([[v * t , 0], [0, -v * t]])  
# Initialize state (start in |1>)
psi = np.array([1, 0], dtype=complex)

def analytical_eigenvalues(t):
    epsilon = v * t
    return np.array([epsilon - np.sqrt(epsilon**2 + Delta**2), epsilon + np.sqrt(epsilon**2 + Delta**2)])


# Time evolution
psi_t = []
energies = np.zeros((len(t_values), 2))
nonint_energies = np.zeros_like(energies)
E0, C0 = np.linalg.eigh(H(t_values[0]))
psi = C0[:, 0]
breakpoint()
for i, t in enumerate(t_values):
    U = scipy.linalg.expm(-1j * H(t) * dt)  # Time evolution operator
    psi = U @ psi
    psi_t.append(psi.copy())
    energies[i], _ = np.linalg.eigh(H(t))
    nonint_energies[i], _ = np.linalg.eigh(Hnon(t))

psi_t = np.array(psi_t)
# Compute probabilities
P1 = np.abs(psi_t[:, 0])**2
P2 = np.abs(psi_t[:, 1])**2
plt.plot(t_values, energies[:,0], 'r-', t_values, energies[:,1], 'b-')
plt.plot(t_values, nonint_energies[:,0], 'r--', t_values, nonint_energies[:,1], 'b--')
plt.plot(t_values[::50], analytical_eigenvalues(t_values[::50])[0,:], 'ro', t_values[::50], analytical_eigenvalues(t_values[::50])[1,:], 'bo')
# plt.scatter(t_values, energies[:,0], t_values, energies[:,1])
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Landau-Zener Transition')
plt.legend(['E1', 'E2'])
plt.show()
exit()


# Plot results
plt.plot(t_values, P1, label=r'$P_1(t)$')
plt.plot(t_values, P2, label=r'$P_2(t)$')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Landau-Zener Transition')
plt.show()
