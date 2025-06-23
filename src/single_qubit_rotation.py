import numpy as np
from scipy.linalg import expm

from functools import partial
from scipy.optimize import minimize

 

def fidelity(U, U_0, d=4):
    """Calculate average fidelity between two unitary gate operation matrices U and U_0."""
    M = U_0.T.conj() @ U
    d = d if d is None else d

    return ((d + np.abs(np.trace(M)) ** 2) / (d * (d + 1))).real
    # return ((np.trace(M @ M.T.conj()) + np.abs(np.trace(M)) ** 2) / (d * (d + 1))).real
 
def classical_fidelity(U_log, U_ideal):
    P_log = np.abs(U_log)**2
    P_targt = np.abs(U_ideal)**2

    F_j = np.sum(np.sqrt(P_log * P_targt), axis=0)
    F_classical = np.mean(F_j)
    return F_classical
 

def opt_fidelity(U, U_0):
    a1, a2, a3 = phase_corr_angles(U)
    opt_res = minimize(
        lambda x: -fidelity_opt_correction(x, U, partial(fidelity, U_0= U_0)),
        [(a2) % (2 * np.pi), (a3) % (2 * np.pi)],
    )

    G = SQ_rotation(opt_res.x, U)
    G = np.exp(-1j * np.angle(G[0, 0])) * G

    return -opt_res.fun, opt_res.x, G
 

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



if __name__ == "__main__":
    import pickle
    sqrtswap = True  # Set to True if you want to load sqrtSWAP data
    if sqrtswap:
        with open("../data/sqrtSWAP_gate_2306.pkl", "rb") as f:
            data = pickle.load(f)
        U_log = data["U_log"]
        C0 = data["C0"]
        psi00 = data["psi00"]
        psi01 = data["psi01"]
        psi10 = data["psi10"]
        psi11 = data["psi11"]
    swap = True  # Set to True if you want to load SWAP data
    if swap:
        with open("../data/SWAP_gate_2306.pkl", "rb") as f:
            data = pickle.load(f)
        U_log = data["U_log"]
        C0 = data["C0"]
        psi00 = data["psi00"]
        psi01 = data["psi01"]
        psi10 = data["psi10"]
        psi11 = data["psi11"]

    breakpoint()
