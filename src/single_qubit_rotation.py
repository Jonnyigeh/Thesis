import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import numpy as np
from scipy.linalg import expm

from functools import partial
from scipy.optimize import minimize
 
def classical_fidelity(U_0, U_target):
    """Calculate classical fidelity between two unitary matrices U_0, U_target."""
    d = U_0.shape[0]
    P = np.abs(U_0)**2
    P_targt = np.abs(U_target)**2
    F_j = np.sum(np.sqrt(P * P_targt), axis=0)
    F_classical = np.mean(F_j)

    return F_classical

def average_fidelity(U_0, U_target):
    """Calculate aveage fidelity between two unitary matrices U_0, U_target."""
    d = U_0.shape[0]
    trace = np.trace(U_target.conj().T @ U_0)
    F = (np.abs(trace)**2 + d) / (d * (d + 1))

    return F.real
 
def SQ_rotation(alphas, U):
    a2, a3 = alphas
    P = np.diag(
        [
            np.exp(-0.5j * (a2 + a3)),
            np.exp(-0.5j * (a2 - a3)),
            np.exp(0.5j * (a2 - a3)),
            np.exp(0.5j * (a2 + a3)),
        ]
    )

    return P @ U

def phase_corr_angles(overlaps):
    P = np.angle(overlaps)
    a1 = 0.5 * (P[1, 1] + P[2, 2])
    a2 = P[2, 2] - P[0, 0]
    a3 = P[1, 1] - P[0, 0]

 
    return -a1, -a2, -a3

def fidelity_opt_correction(alphas, U, fidelity):

    return fidelity(SQ_rotation(alphas, U))

def opt_fidelity(U_0, U):
    """Optimize the average fidelity between a unitary matrix U_0 and a unitary matrix U"""
    a1, a2, a3 = phase_corr_angles(U)
    opt_res = minimize(
        lambda x: -fidelity_opt_correction(x, U, partial(average_fidelity, U_0)),
        [(a2) % (2 * np.pi), (a3) % (2 * np.pi)],
    )

    G = SQ_rotation(opt_res.x, U)
    G = np.exp(-1j * np.angle(G[0, 0])) * G

    return -opt_res.fun, opt_res.x, G
 





if __name__ == "__main__":
    import pickle
    sqrtswap = True  # Set to True if you want to load sqrtSWAP data
    if sqrtswap:
        with open("data/sqrtSWAP_gate_2306.pkl", "rb") as f:
            data = pickle.load(f)
        U_log = data["U_log"]
        C0 = data["C0"]
        psi00 = data["psi00"]
        psi01 = data["psi01"]
        psi10 = data["psi10"]
        psi11 = data["psi11"]

        U_ideal = np.array([ [1,0,0,0], [0,(1+1j)/2,(1-1j)/2,0], [0,(1-1j)/2,(1+1j)/2,0],[0,0,0,1]])
    print("sqrtSWAP gate optimization:")
    res, x, G = opt_fidelity(U_ideal, U_log)
    clas_fidel = classical_fidelity(U_ideal, G)
    old_fidel = average_fidelity(U_ideal, U_log)
    print(f"Old fidelity: {old_fidel}")
    print(f"Optimized fidelity: {res}"
          f"\nOptimized angles: {x}\nOptimized gate:\n{G}")
    print(f"Classical fidelity: {clas_fidel}")
    print("abs gate: ", np.abs(G)**2)
    swap = True  # Set to True if you want to load SWAP data
    if swap:
        with open("data/SWAP_gate_2306.pkl", "rb") as f:
            data = pickle.load(f)
        U_log = data["U_log"]
        C0 = data["C0"]
        psi00 = data["psi00"]
        psi01 = data["psi01"]
        psi10 = data["psi10"]
        psi11 = data["psi11"]

        U_ideal = np.array([ [1,0,0,0], [0,0,1,0], [0,1,0,0],[0,0,0,1]])

    print("SWAP gate optimization:")
    old_fidel = average_fidelity(U_ideal, U_log)
    res, x, G = opt_fidelity(U_ideal, U_log)
    print(f"Old fidelity: {old_fidel}")
    print(f"Optimized fidelity: {res}"
          f"\nOptimized angles: {x}\nOptimized gate:\n{G}")
    clas_fidel = classical_fidelity(U_ideal, G)
    print(f"Classical fidelity: {clas_fidel}")
    print("abs gate: ", np.abs(G)**2)