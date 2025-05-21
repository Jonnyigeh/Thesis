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








def landau_zener_benchmark(v, Delta):
    # Time and Hamiltonian setup
    t_values = np.linspace(-5, 5, 1000)
    dt = t_values[1] - t_values[0]
    H = lambda t: np.array([[v * t, Delta], [Delta, -v * t]])
    
    # Initial state: ground state at t = t0
    E0, C0 = np.linalg.eigh(H(t_values[0]))
    psi0 = C0[:, 0]  # Ground state

    # Storage
    pop_exp, pop_euler, pop_sym, pop_cn, pop_rk4 = [], [], [], [], []
    norm_exp, norm_euler, norm_sym, norm_cn, norm_rk4 = [], [], [], [], []
    psi_exp, psi_euler, psi_sym, psi_cn, psi_rk4 = psi0.copy(), psi0.copy(), psi0.copy(), psi0.copy(), psi0.copy()
    error_rk4, error_cn, error_euler = [], [], []
    for i, t in enumerate(t_values):
        H_current = H(t)
        # Matrix exponential (baseline)
        U_exp = scipy.linalg.expm(-1j * H(t) * dt)
        psi_exp = U_exp @ psi_exp
        pop_exp.append(np.abs(psi_exp[1])**2)
        norm_exp.append(np.linalg.norm(psi_exp)**2)

        # Euler-Cromer method (first order)
        psi_euler = (np.eye(2) - 1j * H(t) * dt) @ psi_euler
        pop_euler.append(np.abs(psi_euler[1])**2)
        norm_euler.append(np.linalg.norm(psi_euler)**2)


        # RK4 method (2nd order)
        k1 = -1j * H_current @ psi_rk4
        k2 = -1j * H(t + 0.5 * dt) @ (psi_rk4 + 0.5 * dt * k1)
        k3 = -1j * H(t + 0.5 * dt) @ (psi_rk4 + 0.5 * dt * k2)
        k4 = -1j * H(t + dt) @ (psi_rk4 + dt * k3)
        psi_rk4 = psi_rk4 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        pop_rk4.append(np.abs(psi_rk4[1])**2)
        norm_rk4.append(np.linalg.norm(psi_rk4)**2)
        
        
        # Symmetric 2nd order method
        # H_now = H(t)
        # U_sym = (np.eye(2) - 1j * H_now * dt / 2) @ (np.eye(2) - 1j * H_now * dt / 2)
        # psi_sym = U_sym @ psi_sym
        # pop_sym.append(np.abs(psi_sym[1])**2)
        # norm_sym.append(np.linalg.norm(psi_sym)**2)

        # Crank-Nicholson method
        # Crank-Nicolson update
        I = np.eye(2, dtype=complex)
        A = I + 1j * H_current * dt / 2
        B = I - 1j * H_current * dt / 2
        psi_cn = np.linalg.solve(A, B @ psi_cn)
        pop_cn.append(np.abs(psi_cn[1])**2)
        norm_cn.append(np.linalg.norm(psi_cn)**2)

        # Accuracy check
        error_rk4.append(np.linalg.norm(psi_rk4 - psi_exp))
        error_cn.append(np.linalg.norm(psi_cn - psi_exp))
        error_euler.append(np.linalg.norm(psi_euler - psi_exp))






    return t_values, pop_exp, pop_euler, pop_rk4, pop_cn, norm_exp, norm_euler, norm_rk4, norm_cn, error_rk4, error_cn, error_euler

def landau_zener_probability(v, Delta):
    return np.exp(-np.pi * Delta**2 / v)


matplotlib.style.use('seaborn-v0_8')
colors = sns.color_palette()
b = colors[0]
g = colors[1]
r = colors[2]

Delta = 1.0
v = 7.0
t, pop_exp, pop_euler, pop_rk4, pop_cn, norm_exp, norm_euler, norm_rk4, norm_cn, error_rk4, error_cn, error_euler = landau_zener_benchmark(v=v, Delta=Delta)
labels = ["Landau-Zener (Analytical)", "Matrix Exp.", "Crank-Nicholson", "RK4"]
P_LZ = np.exp(-2 * np.pi * Delta ** 2 / v)
P_exp = pop_exp[-1]
P_cn = pop_cn[-1]
P_rk4 = pop_rk4[-1]
# P_euler = pop_euler[-1]
values = [P_LZ, P_exp, P_cn, P_rk4,]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values, color=colors, edgecolor="k", alpha=0.8)
plt.ylabel("Transition Probability")
plt.title("Comparison of Final Transition Probabilities")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Optionally, annotate values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
             f"{height:.3f}", ha='center', va='bottom')

plt.show()
exit()

errors = [error_euler, error_rk4, error_cn]
labels = ['Euler-Cromer', 'RK4', 'Crank-Nicholson']
plt.figure(figsize=find_figsize(1.2, 0.4))
plt.plot(t, error_cn, label='Crank-Nicholson')
plt.plot(t, error_rk4, label='RK4')
plt.legend(loc='upper right')
# plt.plot(t, error_euler, label='Euler-Cromer', lw=2, color=r)
plt.ylabel(r'$||\Psi_\mathrm{method} - \Psi_\mathrm{expm}||$')
plt.title('Final-State Error vs. Matrix Exponential')
plt.tight_layout()
plt.show()
exit()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4))

# --- Excited state population ---
ax[0].plot(t, pop_exp, label='Matrix Exp.', lw=2, color=b)
# ax[0].plot(t, pop_euler, '--', label='Euler-Cromer', alpha=0.8)
ax[0].plot(t, pop_rk4, ':', label='RK4', alpha=0.8)
ax[0].plot(t, pop_cn, '-.', label='Crank-Nicholson', alpha=0.8)

ax[0].set_xlabel('Time')
ax[0].set_ylabel('Population in |1⟩')
ax[0].set_title('Population Transfer')
ax[0].legend(loc='upper left')

# --- Norm preservation ---
ax[1].plot(t, norm_exp, label='Matrix Exp.', lw=2, color=b)
# ax[1].plot(t, norm_euler, '--', label='Euler-Cromer', alpha=0.8)
ax[1].plot(t, norm_rk4, ':', label='RK4', alpha=0.8)
ax[1].plot(t, norm_cn, '-.', label='Crank-Nicholson', alpha=0.8)

ax[1].set_xlabel('Time')
ax[1].set_ylabel(r'Norm of $\Psi$')
ax[1].set_title('Norm Preservation')
ax[1].legend(loc='lower right')

# Layout adjustments
plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.3)

plt.show()
exit()




### PLOT IN THEORY: LANDAU ZENER COMPARISON OF EXPONENTIAL VS TAYLOR EXPENSAION
# t, pop_exp, pop_euler, pop_sym, pop_cn, norm_exp, norm_euler, norm_sym, norm_cn = landau_zener_benchmark(v=1.0, Delta=1.0)
if False:
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4))

    # # Excited state population

    # ax[0].plot(t, pop_exp, label='Matrix Exp.', lw=2, color=b)
    # ax[0].plot(t, pop_euler, '--', label='Euler-Cromer', alpha=0.8)
    # ax[0].plot(t, pop_sym, ':', label='Second order', alpha=0.8)
    # ax[0].plot(t, pop_cn, '-.', label='Crank-Nicholson', alpha=0.8)

    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Population')
    # ax[0].set_title('Population transfer')
    # ax[0].legend(loc='upper left', )

    # # Norms
    # ax[1].plot(t, norm_exp, label='Matrix Exp.', lw=2, color=b)
    # ax[1].plot(t, norm_euler, '--', label='Euler-Cromer', alpha=0.8)
    # ax[1].plot(t, norm_sym, ':', label='Second order', alpha=0.8)
    # ax[1].plot(t, norm_cn, '-.', label='Crank-Nicholson', alpha=0.8)
    # ax[1].set_xlabel('Time')
    # ax[1].set_ylabel(r'Norm of $\Psi$')
    # ax[1].set_title('Norm preservation')
    # ax[1].legend(loc='upper left', )

    # plt.tight_layout()
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.3)
    # plt.savefig('../doc/figs/landau_zener_numerical_methods.pdf')
    # plt.show()
    exit()







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

# t1, psi1, e1, nonint_e1 = landau_zener(10.0, 1.0)
# t2, psi2, e2, nonint_e2 = landau_zener(1.0, 1.0)
# p11 = np.abs(psi1[:, 0])**2
# p12 = np.abs(psi1[:, 1])**2
# p21 = np.abs(psi2[:, 0])**2
# p22 = np.abs(psi2[:, 1])**2
# # Plot results
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4), sharex=True)
# ax[0].plot(t1, p11, label=r'$\psi_0(t)$', color=b)
# ax[0].plot(t1, p12, label=r'$\psi_1(t)$', color=r)
# ax[1].plot(t2, p21, label=r'$\psi_0(t)$', color=b)
# ax[1].plot(t2, p22, label=r'$\psi_1(t)$', color=r)
# ax[0].set_title(r'$k = 10.0,\quad V=1.0$')
# ax[1].set_title(r'$k = 1.0,\quad V=1.0$')
# ax[0].set_xlabel('Time [s]')
# ax[1].set_xlabel('Time [s]')
# ax[0].set_ylabel('Population')
# ax[1].set_ylabel('Population')
# ax[1].legend(loc='upper right', bbox_to_anchor=(1, 0.75))
# plt.tight_layout()
# fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
# fig.suptitle('Landau-Zener Transition')
# plt.savefig('../doc/figs/landau_zener.pdf')
# plt.show()