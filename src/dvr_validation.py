import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
from tqdm import tqdm
from utils.visualization import find_figsize

# Parameters for Morse
D = 1.0
a = 1.0
x0 = 0.0

# Grid
N = 200
x = np.linspace(-1, 2, N)
dx = x[1] - x[0]

def morse_potential(x, D, a, x0):
    """
    Morse potential function.
    """
    return D * (1 - np.exp(-a * (x - x0)))**2 - D

V_matrix = np.diag(morse_potential(x, D, a, x0))
T_dvr = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            T_dvr[i, j] = np.pi**2 / (6 * dx**2)
        else:
            T_dvr[i, j] = ((-1)**(i - j)) / (dx**2 * (i - j)**2)
T_true = np.zeros((N, N))
T_diag = 1 / dx**2 
T_off_diag = -1 / (2 * dx**2)
for i in range(N):
    for j in range(N):
        if i == j:
            T_true[i, j] = T_diag
        elif abs(i - j) == 1:
            T_true[i, j] = T_off_diag
        else:
            T_true[i, j] = 0


H_true = T_true + V_matrix
H_dvr = T_dvr + V_matrix
# Diagonalize the Hamiltonian
E_dvr, psi_dvr = np.linalg.eigh(H_dvr)
E_true, psi_true = np.linalg.eigh(H_true)
# Visualize the first 5 energy eigenlevels
matplotlib.style.use('seaborn-v0_8')
colors = sns.color_palette()
b = colors[0]
g = colors[1]
r = colors[2]
fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
n_levels = 25 # Number of levels to plot
xpos = 0.0    # x-location for energy ladders
width = 0.25   # half-width of ladder line

# Plot each energy level
for i in range(n_levels):
    E_t = E_true[i]
    E_d = E_dvr[i]
    
    # True energy: solid line
    ax.hlines(E_t, xpos - width, xpos + width, color=b, linewidth=2, label='True' if i == 0 else "")
    
    # DVR energy: dashed line
    ax.hlines(E_d, xpos - width, xpos + width, color=r, linewidth=2, linestyle='--', label='DVR' if i == 0 else "")
    
# Add E=0 reference line
ax.axhline(0, color='black', linestyle='dashed', linewidth=1)
# ax.text(xpos + width + 0.05, 0.5, '$E=0$', va='center', ha='left', fontsize=10)

# Axis and style
ax.set_xlim(xpos - width -0.2 , xpos + width + 0.2)
ax.set_xticks([])
ax.set_ylabel('Energy')
ax.set_title('Energy Spectrum: True vs DVR (First 12Levels)')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()