import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import seaborn as sns
from time import perf_counter
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








def landau_zener_benchmark(v, Delta,
                        t_values = np.linspace(-5, 5, 1000)
                        , dt=None):
    # Time and Hamiltonian setup
    H = lambda t: np.array([[v * t, Delta], [Delta, -v * t]])
    if dt is None:
        dt = t_values[1] - t_values[0]
    # Initial state: ground state at t = t0
    E0, C0 = np.linalg.eigh(H(t_values[0]))
    # psi0 = C0[:, 0]  # Ground state
    psi0 = np.array([1, 0], dtype=np.complex128)  # Start in |0>
    psi1 = np.array([0, 1], dtype=np.complex128)  # Start in |1>

    # Storage
    pop_exp, pop_euler, pop_sym, pop_cn, pop_rk4 = [], [], [], [], []
    norm_exp, norm_euler, norm_sym, norm_cn, norm_rk4 = [], [], [], [], []
    psi_exp, psi_euler, psi_sym, psi_cn, psi_rk4 = psi0.copy(), psi0.copy(), psi0.copy(), psi0.copy(), psi0.copy()
    error_rk4, error_cn, error_euler = [], [], []
    for i, t in tqdm(enumerate(t_values)):
        H_current = H(t)
        # Matrix exponential (baseline)
        U_exp = scipy.linalg.expm(-1j * H_current * dt)
        psi_exp = U_exp @ psi_exp
        pop_exp.append(np.abs(psi_exp[1])**2)
        norm_exp.append(np.linalg.norm(psi_exp)**2)

        # # Euler-Cromer method (first order)
        psi_euler = (np.eye(2) - 1j * H_current * dt) @ psi_euler
        norm_euler.append(np.linalg.norm(psi_euler)**2)
        # psi_euler /= np.linalg.norm(psi_euler)  # Normalize
        pop_euler.append(np.abs(psi_euler[1])**2)


        # RK4 method (2nd order)
        k1 = -1j * H_current @ psi_rk4
        k2 = -1j * H(t + 0.5 * dt) @ (psi_rk4 + 0.5 * dt * k1)
        k3 = -1j * H(t + 0.5 * dt) @ (psi_rk4 + 0.5 * dt * k2)
        k4 = -1j * H(t + dt) @ (psi_rk4 + dt * k3)
        psi_rk4 = psi_rk4 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        norm_rk4.append(np.linalg.norm(psi_rk4)**2)
        # psi_rk4 /= np.linalg.norm(psi_rk4)  # Normalize
        pop_rk4.append(np.abs(psi_rk4[1])**2)
        
        
        # Symmetric 2nd order method
        H_now = H(t)
        U_sym = (np.eye(2) - 1j * H_now * dt / 2) @ (np.eye(2) - 1j * H_now * dt / 2)
        psi_sym = U_sym @ psi_sym
        pop_sym.append(np.abs(psi_sym[1])**2)
        norm_sym.append(np.linalg.norm(psi_sym)**2)

        # Crank-Nicholson method
        # Crank-Nicolson update
        I = np.eye(2, dtype=complex)
        A = I + 1j * H_current * dt / 2
        B = I - 1j * H_current * dt / 2
        psi_cn = np.linalg.solve(A, B @ psi_cn)
        norm_cn.append(np.linalg.norm(psi_cn)**2)
        pop_cn.append(np.abs(psi_cn[1])**2)

        # Accuracy check
        error_rk4.append(np.linalg.norm(psi_rk4 - psi_exp))
        error_cn.append(np.linalg.norm(psi_cn - psi_exp))
        error_euler.append(np.linalg.norm(psi_euler - psi_exp))

        # Calculate transitions 






    return t_values, pop_exp, pop_euler, pop_rk4, pop_cn, norm_exp, norm_euler, norm_rk4, norm_cn, error_rk4, error_cn, error_euler, pop_sym, norm_sym

def landau_zener_probability(v, Delta):
    return np.exp(-np.pi * Delta**2 / v)


def landau_zener_efficiency_benchmark(v, Delta, t_values=np.linspace(-5, 5, 1000), dt=None):
    H = lambda t: np.array([[v * t, Delta], [Delta, -v * t]], dtype=np.complex128)
    if dt is None:
        dt = t_values[1] - t_values[0]

    psi0 = np.array([1, 0], dtype=np.complex128)

    results = {}
    
    # ========== Matrix Exponential ==========
    psi = psi0.copy()
    start = perf_counter()
    for t in t_values:
        U = scipy.linalg.expm(-1j * H(t) * dt)
        psi = U @ psi
    end = perf_counter()
    results["Matrix Exponential"] = end - start

    # ========== Euler Method ==========
    psi = psi0.copy()
    start = perf_counter()
    for t in t_values:
        psi = (np.eye(2) - 1j * H(t) * dt) @ psi
    end = perf_counter()
    results["Euler"] = end - start

    # ========== RK4 ==========
    psi = psi0.copy()
    start = perf_counter()
    for t in t_values:
        H1 = H(t)
        k1 = -1j * H1 @ psi
        k2 = -1j * H(t + 0.5 * dt) @ (psi + 0.5 * dt * k1)
        k3 = -1j * H(t + 0.5 * dt) @ (psi + 0.5 * dt * k2)
        k4 = -1j * H(t + dt) @ (psi + dt * k3)
        psi = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    end = perf_counter()
    results["RK4"] = end - start

    # ========== Crank-Nicholson ==========
    psi = psi0.copy()
    start = perf_counter()
    for t in t_values:
        H_now = H(t)
        I = np.eye(2, dtype=np.complex128)
        A = I + 1j * H_now * dt / 2
        B = I - 1j * H_now * dt / 2
        psi = np.linalg.solve(A, B @ psi)
    end = perf_counter()
    results["Crank-Nicholson"] = end - start

    return results









# Set up the plotting style
matplotlib.style.use('seaborn-v0_8-deep')
colors = sns.color_palette()
b = colors[0]
g = colors[1]
r = colors[2]
benchmark=False
if benchmark:
    v = 7.0
    Delta = 1.0
    t_values = np.linspace(-5, 5, 100_000)
    # res = landau_zener_efficiency_benchmark(v=v, Delta=Delta, t_values=t_values)
    # import pickle
    # with open('data/lz_runtime_benchmark.pkl', 'wb') as f:
    #     pickle.dump(res, f)
    # breakpoint()
    # exit()
    # Convergence benchmark for Landau-Zener transition probability
    Delta = 1.0
    v = 7.0
    dt = 0.01
    t1 = np.linspace(-1, 1, 10_000)
    t2 = np.linspace(-5, 5 , 10_000)
    t3 = np.linspace(-10, 10, 10_000)
    t4 = np.linspace(-20, 20, 10_000)
    t5 = np.linspace(-50, 50, 10_000)
    t6 = np.linspace(-100, 100, 10_000)
    t7 = np.linspace(-300, 300, 10_000)
    P_LZ = np.exp(-2 * np.pi * Delta ** 2 / v)
    # # # Store intermediate results
    tags = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    i=0
    probs = {}
    errors = {}

    for t in tqdm([t1, t2, t3, t4, t5, t6, t7], desc="Running benchmarks"):
        t, pop_exp, pop_euler, pop_rk4, pop_cn, norm_exp, norm_euler, norm_rk4, norm_cn, error_rk4, error_cn, error_euler = landau_zener_benchmark(v=v, Delta=Delta, t_values=t)
        P_exp = pop_exp[-1]
        P_cn = pop_cn[-1]
        P_rk4 = pop_rk4[-1]
        P_euler = pop_euler[-1]

        # Store results
        probs[tags[i]] = {
            "P_exp": P_exp,
            "P_cn": P_cn,
            "P_rk4": P_rk4,
            "P_euler": P_euler
        }
        errors[tags[i]] = {
            "error_rk4": error_rk4,
            "error_cn": error_cn,
            "error_euler": error_euler
        }
        i += 1

    import pickle
    try:
        with open('data/lz_benchmark_results_constdt_1906.pkl', 'wb') as f:
            pickle.dump(probs, f)
        with open('data/lz_benchmark_errors_constdt_1906.pkl', 'wb') as f:
            pickle.dump(errors, f)
    except:
        print("Failed to save results to file.")
        breakpoint()
    # with open('data/lz_benchmark_results_constdt.pkl', 'rb') as f:
    #     probs = pickle.load(f)
    # with open('data/lz_benchmark_errors_constdt.pkl', 'rb') as f:
    #     errors = pickle.load(f)
    windows =[2.0, 10.0, 20.0, 40.0, 100.0, 200.0, 600.0] # seconds
    P_exp = [probs[tag]["P_exp"] for tag in tags]
    P_cn = [probs[tag]["P_cn"] for tag in tags]
    P_rk4 = [probs[tag]["P_rk4"] for tag in tags]
    P_euler = [probs[tag]["P_euler"] for tag in tags]

    fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
    ax.plot(windows, P_exp, 'o-', label='Matrix Exponential')
    ax.plot(windows, P_cn, 's-', label='Crank-Nicholson')
    ax.plot(windows, P_rk4, '^-', label='RK4')
    ax.plot(windows, P_euler, 'x--', label='Euler')
    ax.axhline(P_LZ, color='k', linestyle='--', label='Analytical $P_{LZ}$')
    ax.set_ylim([-0.2, 1])
    ax.set_xscale("log")
    ax.set_xlabel(r"Total simulation time window $2T$ [s]")
    ax.set_ylabel(r"Transition probability $P_{|0\rangle \to |1\rangle}$")
    ax.set_title("Numerical convergence toward Landau-Zener transition probability")
    ax.legend()
    ax.grid(True)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('../doc/figs/landau_zener_convergence_benchmark.pdf')
    plt.show()
    exit()


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4))

# # --- Excited state population ---
# ax[0].plot(t, pop_exp, label='Matrix Exp.', lw=2, color=b)
# # ax[0].plot(t, pop_euler, '--', label='Euler-Cromer', alpha=0.8)
# ax[0].plot(t, pop_rk4, ':', label='RK4', alpha=0.8)
# ax[0].plot(t, pop_cn, '-.', label='Crank-Nicholson', alpha=0.8)

# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Population in |1⟩')
# ax[0].set_title('Population Transfer')
# ax[0].legend(loc='upper left')

# # --- Norm preservation ---
# ax[1].plot(t, norm_exp, label='Matrix Exp.', lw=2, color=b)
# # ax[1].plot(t, norm_euler, '--', label='Euler-Cromer', alpha=0.8)
# ax[1].plot(t, norm_rk4, ':', label='RK4', alpha=0.8)
# ax[1].plot(t, norm_cn, '-.', label='Crank-Nicholson', alpha=0.8)

# ax[1].set_xlabel('Time')
# ax[1].set_ylabel(r'Norm of $\Psi$')
# ax[1].set_title('Norm Preservation')
# ax[1].legend(loc='lower right')

# # Layout adjustments
# plt.tight_layout()
# fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.3)
# plt.savefig('../doc/figs/landau_zener_numerical_methods.pdf')
# plt.show()
# exit()




### PLOT IN THEORY: LANDAU ZENER COMPARISON OF EXPONENTIAL VS TAYLOR EXPENSAION
# t, pop_exp, pop_euler, pop_rk4, pop_cn, norm_exp, norm_euler, norm_rk4, norm_cn, _,_,_, pop_sym, norm_sym = landau_zener_benchmark(v=1.0, Delta=1.0)
method_comparison = False
if method_comparison:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=find_figsize(1.2, 0.4))

    # Excited state population

    ax[0].plot(t, pop_exp, label='Matrix Exp.', lw=2, color=b)
    ax[0].plot(t, pop_euler, '--', label='Euler-Cromer', alpha=0.8)
    ax[0].plot(t, pop_sym, ':', label='Second order', alpha=0.8)
    ax[0].plot(t, pop_cn, '-.', label='Crank-Nicholson', alpha=0.8)

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Population')
    ax[0].set_title('Population transfer')
    ax[0].legend(loc='upper left', )

    # Norms
    ax[1].plot(t, norm_exp, label='Matrix Exp.', lw=2, color=b)
    ax[1].plot(t, norm_euler, '--', label='Euler-Cromer', alpha=0.8)
    ax[1].plot(t, norm_sym, ':', label='Second order', alpha=0.8)
    ax[1].plot(t, norm_cn, '-.', label='Crank-Nicholson', alpha=0.8)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel(r'Norm of $\Psi$')
    ax[1].set_title('Norm preservation')
    ax[1].legend(loc='upper left', )

    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.3)
    # plt.savefig('../doc/figs/landau_zener_numerical_methods.pdf')
    plt.show()
    exit()






crossing=False
if crossing:
    t_values, psi_t, energies, nonint_energies = landau_zener(2.0, 1.0)
    psi_t = np.array(psi_t)
    # Compute probabilities
    P1 = np.abs(psi_t[:, 0])**2
    P2 = np.abs(psi_t[:, 1])**2
    fig, ax = plt.subplots(figsize=find_figsize(1.2, 0.4))
    ax.plot(t_values, energies[:,0], linestyle='-', color=r, label=r'$E_1$ (coupled)')
    ax.plot(t_values, nonint_energies[:,0], color=r, linestyle='--', label=r'$E_1$')
    ax.plot(t_values, energies[:,1], linestyle='-', color=b, label=r'$E_2$ (coupled)')
    ax.plot(t_values, nonint_energies[:,1], linestyle='--', color=b, label=r'$E_2$')
    # ax.plot(t_values[::50], analytical_eigenvalues(t_values[::50])[0,:], 'ro', t_values[::50], analytical_eigenvalues(t_values[::50])[1,:], 'bo')
    # ax.scatter(t_values, energies[:,0], t_values, energies[:,1])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [a.u.]')
    ax.set_title('Landau-Zener Transition')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.75))
    # plt.tight_layout()
    fig.subplots_adjust(left=0.15, right=0.87, top=0.9, bottom=0.15)
    # plt.savefig('../doc/figs/avoided_crossing.pdf')
    plt.show()
    exit()
pop_transfer=False
if pop_transfer
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
    # plt.savefig('../doc/figs/landau_zener.pdf')
    plt.show()