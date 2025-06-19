import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Ignores a warnings due to scipy.special version dislike of current python3.10 installation
import matplotlib.style
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import matplotlib
import seaborn as sns

try:
    from potential import MorsePotentialDW
    from qd_system import ODMorse
    from sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_bhs
except:
    from utils.potential import MorsePotentialDW
    from utils.qd_system import ODMorse
    from utils.sinc_bipartite_hartree import BipartiteHartreeSolver as sinc_bhs


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
def visualize_ramp_protocol():
    def lambda_t():
        """Return the time-dependent parameter lambda."""
        chill_before = 3
        ramp_up = 3
        plateau = 5
        ramp_down = 3
        chill_after = 3
        t_max = (chill_before + ramp_up +
                    plateau   + ramp_down +
                    chill_after)
        t = np.linspace(0, t_max, 1000)
        t1 = chill_before
        t2 = t1 + ramp_up
        t3 = t2 + plateau
        t4 = t3 + ramp_down

        lmbda = np.zeros_like(t)
        # masks
        m0 = (t <=  t1)
        m1 = (t >  t1) & (t <= t2)
        m2 = (t >  t2) & (t <= t3)
        m3 = (t >  t3) & (t <= t4)
        m4 = (t >  t4)
        # assign segments
        lmbda[m0] = 0.0
        # up‐ramp: cosine from 0->1
        lmbda[m1] = 0.5*(1 - np.cos(np.pi*(t[m1]-t1)/ramp_up))
        # plateau
        lmbda[m2] = 1.0
        # down‐ramp: cosine from 1->0
        lmbda[m3] = 0.5*(1 + np.cos(np.pi*(t[m3]-t3)/ramp_down))
        # final chill
        lmbda[m4] = 0.0

        return lmbda, t, t1, t2, t3, t4, t_max
    lmbda, t, t1, t2, t3, t4, t_max = lambda_t()
    matplotlib.style.use('seaborn-v0_8-deep')
    color = sns.color_palette()
    b = color[0]
    figsize = find_figsize(1.2, 0.3)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, lmbda, lw=2, linestyle='--', color=b)

    # Draw vertical dashed lines at segment boundaries
    for xi in [t1, t2, t3, t4]:
        ax.axvline(xi, color='gray', linestyle='-', linewidth=1)

    # Annotate segments with horizontal labels at midpoints
    y_annot = 0.05  # place below x-axis
    ax.text((t1+t2)/2, y_annot, 'ramp up', ha='center', va='top')
    ax.text((t2+t3)/2, y_annot, 'hold', ha='center', va='top')
    ax.text((t3+t4)/2, y_annot, 'ramp down', ha='center', va='top')


    # Axis labels and limits
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel(r'$\lambda(t)$')
    ax.set_title('Ramping Protocol')

    fig.tight_layout(rect=[0,0,0.93,1])
    plt.savefig('../doc/figs/ramp_protocol.pdf')
    plt.show()




def show_potential_with_spf():
    """Show the potential and the single-particle functions."""
    num_grid_points = 4_001
    grid = np.linspace(-1.5, 10.0, num_grid_points)
    D = 10
    a = 0.52
    # sns.set_theme()
    matplotlib.style.use('seaborn-v0_8-deep')
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
    E_n = compute_eigenenergies(a, D, 6)
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
    figsize = find_figsize(1.2, 0.3)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharex=True)
    ax[0].plot(grid, V, lw=2)
    ax[0].set_title("Morse potential")
    ax[0].set_xlabel("Position [a.u]")
    ax[0].set_ylabel("Energy [a.u]")
    ax[0].hlines(E_n[0], grid[e0[0]], grid[e0[1]], linestyles="--")
    ax[0].hlines(E_n[1], grid[e1[0]], grid[e1[1]], linestyles="--")
    ax[0].hlines(E_n[2], grid[e2[0]], grid[e2[1]], linestyles="--")
    ax[0].hlines(E_n[3], grid[e3[0]], grid[e3[1]], linestyles="--")
    ax[0].hlines(E_n[4], grid[e4[0]], grid[e4[1]], linestyles="--")
    ax[0].axhline(D, linestyle="--", linewidth=1.5, color="black")
    ax[0].legend(["V(x)", "Energy levels"])
    # ax[0].annotate(
    #     r"D",
    #     xy=(grid[-1], D),
    #     xytext=(grid[-1000], D - 2),
    #     arrowprops=dict(facecolor="black", arrowstyle="->"),
    # )
    ax[1].plot(grid, np.abs(spf[0])**2, grid, np.abs(spf[1])**2, grid, np.abs(spf[2])**2, grid, np.abs(spf[3])**2, grid, np.abs(spf[4])**2)
    ax[1].set_title("Single-particle functions")
    ax[1].set_xlabel("Position [a.u]")
    ax[1].set_ylabel(r"Amplitude $|\phi_n(x)|^2$")
    ax[1].legend(["n=0", "n=1", "n=2", "n=3", "n=4"])
    ax[1].hlines(0, grid[0], grid[-1], color="black", linestyle="--", linewidth=1.5)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('../doc/figs/potential_spf.pdf')
    plt.show()


def show_double_well_pot():
    """Show the double well potential."""
    matplotlib.style.use('seaborn-v0_8-deep')
    colors = sns.color_palette()
    b = colors[0]
    g = colors[1]
    r = colors[2]
    p = 'k'
    # Define the double well potential
    pot = MorsePotentialDW(D_a=10, D_b=11, k_a=7.0, k_b=7.0, d=4)
    grid = np.linspace(-2.6, 2.685, 200)
    V = pot(grid[1:-1])
    V -= np.min(V)  # Shift the potential to have a minimum at zero
    def solve_sys():
        grid_length = 5.2
        num_grid_points = 200
        alpha=1.0
        potential = pot
        a = 0.1
        l = 25

        basis = ODMorse(
            l=l,
            grid_length=grid_length,
            num_grid_points=num_grid_points,
            _a=a,
            alpha=alpha,
            potential=potential,
        )
        num_l = 4
        num_r = 4
        h_l = basis._h_l
        h_r = basis._h_r
        eps_l, _ = scipy.linalg.eigh(h_l, subset_by_index=(0, num_l-1))
        eps_r, _ = scipy.linalg.eigh(h_r, subset_by_index=(0, num_r-1))

        return eps_l, eps_r
    # Find single-particle energies
    # eps_l, eps_r = solve_sys() 
    # el = eps_l - eps_l[0]
    # er = eps_r - eps_r[0]
    figsize = find_figsize(1.1, 0.25)
    fig, ax = plt.subplots(figsize=figsize)


    ax.plot(grid[1:-1], V, label="V(x)", color=b, lw=2)
    ax.set_xlabel("Position [a.u]")
    ax.set_ylabel("Potential energy [a.u]")
    ax.set_title("Example double well Morse potential")
    # axE = ax.twinx()
    # axE.set_ylabel("Single-particle energy [a.u]")
    # x0,x1 = grid[0], grid[-1]
    # for e in el:
    #     axE.hlines(e, x0, x1, colors=b, linestyles='-')
    # for e in er:
    #     axE.hlines(e, x0, x1, colors=r, linestyles='--')
    # labels = [r'$\epsilon_{L}$', r'$\epsilon_{R}$']
    # linesE = [plt.Line2D([],[],color=b,ls='-'),
    #       plt.Line2D([],[],color=r,ls='--')]
    # linesV, labelsV = ax.get_legend_handles_labels()
    # ax.legend(linesE + linesV, labels + labelsV, loc='upper right', fontsize=10)

    ax.legend()
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig('../doc/figs/double_well_potential.pdf')
    plt.show()

if __name__ == "__main__":
    exit()
    show_potential_with_spf()
    show_double_well_pot()
    visualize_ramp_protocol()