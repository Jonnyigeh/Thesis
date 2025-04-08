import warnings
import numpy as np
import scipy.linalg


class BipartiteHartreeSolver:
    def __init__(self, h_l, h_r, u_lr, num_basis_l, num_basis_r):
        self.h_l = h_l
        self.h_r = h_r
        self.u_lr = u_lr
        self.num_basis_l = num_basis_l
        self.num_basis_r = num_basis_r

        # Set up initial guess
        # We might need different guesses, but lets start here
        self.eps_l, self.c_l, self.eps_r, self.c_r = self.diagonalize_fock_matrices(
            self.h_l, self.h_r
        )

    def diagonalize_fock_matrices(self, f_l, f_r):
        eps_l, c_l = scipy.linalg.eigh(f_l, subset_by_index=[0, self.num_basis_l - 1])
        eps_r, c_r = scipy.linalg.eigh(f_r, subset_by_index=[0, self.num_basis_r - 1])

        return eps_l, c_l, eps_r, c_r

    def construct_fock_matrices(self, h_l, h_r, u_lr, c_l, c_r):
        return (
            h_l + np.diag(np.einsum("c, ac -> a", np.abs(c_r[:, 0]) ** 2, u_lr)),
            h_r + np.diag(np.einsum("c, ca -> a", np.abs(c_l[:, 0]) ** 2, u_lr)),
        )

    def compute_energy(self, c_l, c_r):
        h_l = c_l[:, 0].conj() @ self.h_l @ c_l[:, 0]
        h_r = c_r[:, 0].conj() @ self.h_r @ c_r[:, 0]
        u = np.abs(c_l[:, 0]) ** 2 @ self.u_lr @ np.abs(c_r[:, 0]) ** 2
        return h_l + h_r + u

    def solve(self, max_iter=100, tol=1e-10):
        for i in range(max_iter):
            f_l, f_r = self.construct_fock_matrices(
                self.h_l, self.h_r, self.u_lr, self.c_l, self.c_r
            )
            eps_l, c_l, eps_r, c_r = self.diagonalize_fock_matrices(f_l, f_r)
            test_l = np.all(np.abs(eps_l - self.eps_l) < tol)
            test_r = np.all(np.abs(eps_r - self.eps_r) < tol)
            self.eps_l = eps_l
            self.eps_r = eps_r
            self.c_l = c_l
            self.c_r = c_r

            if test_l and test_r:
                break

        if i == max_iter - 1:
            warnings.warn("Maximum number of iterations reached")
