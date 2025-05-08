import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
from tqdm import tqdm
from utils.visualization import find_figsize
def landau_zener(v, Delta):
    # Define parameters
    Delta =Delta  # Coupling strength
    v = v  # Sweeping rate
    t_values = np.linspace(-5, 5, 1000)  # Time range
    dt = t_values[1] - t_values[0]  # Time step
    E0 = 1.0
    # Define the Hamiltonian function
    def H(t):
        epsilon = v * t
        return np.array([[epsilon, Delta], [Delta, -epsilon]])
    Hnon = lambda t:  np.array([[v * t , 0], [0, -v * t]])  
    # Initialize state (start in |1>)
    psi = np.array([1, 0], dtype=np.complex128)

    def analytical_eigenvalues(t):
        epsilon = v * t
        return np.array([epsilon - np.sqrt(epsilon**2 + Delta**2), epsilon + np.sqrt(epsilon**2 + Delta**2)])


    # Time evolution
    psi_t = []
    energies = np.zeros((len(t_values), 2))
    nonint_energies = np.zeros_like(energies)
    E0, C0 = np.linalg.eigh(H(t_values[0]))
    psi = C0[:, 0]
    for i, t in enumerate(t_values):
        U = scipy.linalg.expm(-1j * H(t) * dt)  # Time evolution operator
        psi = U @ psi
        psi_t.append(psi.copy())
        energies[i], _ = np.linalg.eigh(H(t))
        nonint_energies[i], _ = np.linalg.eigh(Hnon(t))
    
    return t_values, np.array(psi_t), energies, nonint_energies

matplotlib.style.use('seaborn-v0_8')
colors = sns.color_palette()
b = colors[0]
r = colors[2]

# t_values, psi_t, energies, nonint_energies = landau_zener(2.0, 1.0)
# psi_t = np.array(psi_t)
# # Compute probabilities
# P1 = np.abs(psi_t[:, 0])**2
# P2 = np.abs(psi_t[:, 1])**2
# fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
# ax.plot(t_values, energies[:,0], linestyle='-', color=r, label=r'$E_1$ (coupled)')
# ax.plot(t_values, nonint_energies[:,0], color=r, linestyle='--', label=r'$E_1$')
# ax.plot(t_values, energies[:,1], linestyle='-', color=b, label=r'$E_2$ (coupled)')
# ax.plot(t_values, nonint_energies[:,1], linestyle='--', color=b, label=r'$E_2$')
# # ax.plot(t_values[::50], analytical_eigenvalues(t_values[::50])[0,:], 'ro', t_values[::50], analytical_eigenvalues(t_values[::50])[1,:], 'bo')
# # ax.scatter(t_values, energies[:,0], t_values, energies[:,1])
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Energy [a.u.]')
# ax.set_title('Landau-Zener Transition')
# ax.legend(loc='upper right', bbox_to_anchor=(1, 0.75))
# # plt.tight_layout()
# fig.subplots_adjust(left=0.15, right=0.87, top=0.9, bottom=0.15)
# plt.savefig('../doc/figs/avoided_crossing.pdf')
# plt.show()
# exit()

t1, psi1, e1, nonint_e1 = landau_zener(10.0, 1.0)
t2, psi2, e2, nonint_e2 = landau_zener(1.0, 1.0)
p11 = np.abs(psi1[:, 0])**2
p12 = np.abs(psi1[:, 1])**2
p21 = np.abs(psi2[:, 0])**2
p22 = np.abs(psi2[:, 1])**2
# Plot results
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4), sharex=True)
ax[0].plot(t1, p11, label=r'$\psi_0(t)$', color=b)
ax[0].plot(t1, p12, label=r'$\psi_1(t)$', color=r)
ax[1].plot(t2, p21, label=r'$\psi_0(t)$', color=b)
ax[1].plot(t2, p22, label=r'$\psi_1(t)$', color=r)
ax[0].set_title(r'$k = 10.0,\quad V=1.0$')
ax[1].set_title(r'$k = 1.0,\quad V=1.0$')
ax[0].set_xlabel('Time [s]')
ax[1].set_xlabel('Time [s]')
ax[0].set_ylabel('Population')
ax[1].set_ylabel('Population')
ax[1].legend(loc='upper right', bbox_to_anchor=(1, 0.75))
plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
fig.suptitle('Landau-Zener Transition')
plt.savefig('../doc/figs/landau_zener.pdf')
plt.show()