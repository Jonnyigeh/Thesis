import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import matplotlib.style
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import matplotlib
import seaborn as sns


import quantum_systems as qs        # Quantum systems library, will be swapped out with the new library at a later stage

def find_figsize(width_scale=1, height_scale=1):
    """Calculate the figsize based on the LaTeX scaling of \textwidth.

    Args:
        width_scale (int, optional): Scaling of \textwidth in width. Defaults to 1.
        height_scale (int, optional): Scaling of \textwidth in height. Defaults to 1.

    Returns:
        figsize 
    """
    textwidth_pt = 418.25368
    textwidth_in_inches = textwidth_pt * 0.01384
    textheight_pt = 674.33032
    textheight_in_inches = textheight_pt * 0.01384

    figure_width = textwidth_in_inches * width_scale
    figure_height = textheight_in_inches * height_scale

    figsize = (figure_width, figure_height)
    
    return figsize


def show_potential_with_spf():
    """Show the potential and the single-particle functions."""
    num_grid_points = 4_001
    grid = np.linspace(-1.5, 10.0, num_grid_points)
    D = 10
    a = 0.5
    # sns.set_theme()
    matplotlib.style.use('seaborn-v0_8')
    def compute_eigenenergies(c, D, l):
        hnu = 2 * c * np.sqrt(D / 2)
        E_n = np.zeros(l)
        for n in range(l):
            E_n[n] = hnu * (n + 0.5) - (c * hnu * (n + 0.5)**2) / np.sqrt(8 * D)

        return E_n
    def morse_spf(x, n, lmbda, x_e, c):
        z = 2 * lmbda * np.exp(-c * (x - x_e))
        return (
            (scipy.special.factorial(n) * (2 * lmbda - 2 * n - 1) * c / scipy.special.gamma(2 * lmbda - n))**0.5 * 
             z**(lmbda - n - 0.5) * np.exp(-z / 2) * scipy.special.genlaguerre(n, 2 * lmbda - 2 * n - 1)(z)
        )
    def morse_pot(x, D, a, x_e):
        return D * (1 - np.exp(-a * (x - x_e)))**2
    V = morse_pot(grid, D, a, 0)
    E_n = compute_eigenenergies(a, D, 5)
    # find crossing points
    e0 = np.where(np.abs(V - E_n[0]) < 1e-2)
    e0 = [324, 800]
    e1 = [215, 1084]
    e2 = [155, 1344]
    e3 = [115, 1618]
    e4 = [87, 1930]
    lmbda = np.sqrt(2 * D) / a
    spf = np.zeros((5, num_grid_points))
    for i in range(5):
        spf[i] = morse_spf(grid, i, lmbda, 0, a)
    # figsize=(10, 5)
    figsize = find_figsize(1.2, 0.4)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharex=True)
    ax[0].plot(grid, V, label="Potential")
    ax[0].set_title("Potential")
    ax[0].set_xlabel("Position [a.u]")
    ax[0].set_ylabel("Energy [a.u]")
    ax[0].hlines(E_n[0], grid[e0[0]], grid[e0[1]], linestyles="--")
    ax[0].hlines(E_n[1], grid[e1[0]], grid[e1[1]], linestyles="--")
    ax[0].hlines(E_n[2], grid[e2[0]], grid[e2[1]], linestyles="--")
    ax[0].hlines(E_n[3], grid[e3[0]], grid[e3[1]], linestyles="--")
    ax[0].hlines(E_n[4], grid[e4[0]], grid[e4[1]], linestyles="--")
    ax[0].axhline(D, linestyle="--", linewidth=1.5, color="black")
    ax[0].legend(["Potential", "Energy levels"])
    # ax[0].annotate(
    #     r"D",
    #     xy=(grid[-1], D),
    #     xytext=(grid[-1000], D - 2),
    #     arrowprops=dict(facecolor="black", arrowstyle="->"),
    # )
    ax[1].plot(grid, np.abs(spf[0])**2, grid, np.abs(spf[1])**2, grid, np.abs(spf[2])**2, grid, np.abs(spf[3])**2, grid, np.abs(spf[4])**2)
    ax[1].set_title("Single-particle functions")
    ax[1].set_xlabel("Position [a.u]")
    ax[1].set_ylabel("Amplitude")
    ax[1].legend(["n=0", "n=1", "n=2", "n=3", "n=4"])
    fig.tight_layout()
    plt.savefig('../doc/figs/potential_spf.pdf')
    plt.show()

if __name__ == "__main__":
    show_potential_with_spf()