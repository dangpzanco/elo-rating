import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import model_utils as mutils
import plot_utils as lutils


# Figure 1
def plot_tau(
    num_beta=1000, M=15, v=2, hfa=0,
    lang='en', path='figures/discussion'
):
    """Time constant as a function of lr for different v, with M players"""

    name = 'time_constants'
    if 'pt' in lang:
        name = 'ptbr_' + name
        xylabel = dict(x='Passo de adaptação $\\beta$', y='Constante de tempo')
    else:
        xylabel = dict(x='Step size $\\beta$', y='Time constant')

    beta_max = 2
    beta = np.logspace(-2, np.log10(beta_max), num_beta)

    tau1 = mutils.time_constant(
        beta, v, M, hfa=hfa, approx=True, first_order=True)
    tau2 = mutils.time_constant(
        beta, v, M, hfa=hfa, approx=True, first_order=False)

    fig, ax = plt.subplots()

    ax.plot(beta, tau1, label=r"$\tau_1$")
    ax.plot(beta, tau2, label=r"$\tau_2$")
    ax.grid(True)

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xylabel['x'])
    ax.set_ylabel(xylabel['y'])
    ax.autoscale(tight=True, axis='x')
    fig.tight_layout()

    lutils.save_fig(fig, name=name, path=path, format='pdf', close=False)

    return fig, ax


def plot_lmin(num_points=1000, hfa_values=None, lang='en', path='figures/discussion'):

    name = 'loss_min'
    if 'pt' in lang:
        name = 'ptbr_' + name
        xylabel = dict(x='Variância $v$', y='Limite inferior do custo')
    else:
        xylabel = dict(x='Variance $v$', y='Loss lower-bound')

    if hfa_values is None:
        hfa_values = [0, 0.5, 1]

    v = np.logspace(-2, 1, num_points)
    # v = np.logspace(-2, np.log10(20), num_points)

    fig, ax = plt.subplots()

    # Colored lines
    for hfa in hfa_values:
        _, _, lmin = mutils.analytical_expectations(v, hfa)
        ax.plot(v, lmin, label=f'$\\eta = {hfa}$')

    # First assymptote
    _, _, lmin0 = mutils.analytical_expectations(0, 0)
    ax.axhline(lmin0, ls='--', color='k')

    # Second assymptote
    f_l = np.log(2)
    v_l = 4 * np.log(2)
    v_z = 2 * v
    lmin_inf = f_l / np.sqrt(v_z / v_l)
    # _, _, lmin_inf = mutils.analytical_expectations(v + 4 * np.log(2), 0)
    ax.plot(v, lmin_inf, ls='--', color='k')
    ax.grid(True)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axis([None, None, 0.2, 0.8])
    ax.autoscale(tight=True, axis='x')

    formatter = mticker.FuncFormatter(lambda x, pos: f'${x:.1f}$')
    ax.yaxis.set_minor_formatter(formatter)

    ax.legend(loc='lower left')
    ax.set_xlabel(xylabel['x'])
    ax.set_ylabel(xylabel['y'])
    fig.tight_layout()

    lutils.save_fig(fig, name=name, path=path, format='pdf', close=False)

    return fig, ax


def plot_lex(
    M=15, v=3, hfa=0, num_beta=1000, games=None, lang='en', path='figures/discussion'
):

    name = 'steady_state'
    if 'pt' in lang:
        name = 'ptbr_' + name
        xylabel = dict(x='Passo de adaptação $\\beta$', y='Custo de excesso')
    else:
        xylabel = dict(x='Step size $\\beta$', y='Excess loss')

    if games is None:
        games = np.array([50, 100, 200])

    beta = np.logspace(-2, np.log10(4), num_beta)
    h, _, _ = mutils.analytical_expectations(v, hfa)
    d_inf = mutils.var_expectation(beta, v, M, None, hfa=hfa)
    l_inf = d_inf * h / (M-1)

    fig, ax = plt.subplots()

    for K in games:
        d_k = mutils.var_expectation(beta, v, M, K, hfa=hfa)
        lex = d_k * h / (M-1)
        ax.plot(beta, lex, label=f'$k = {round(K)}$')
    ax.plot(beta, l_inf, label=r'$k \to \infty$', color='k')
    ax.grid(True)

    ax.set_xscale('log')
    ax.axis([None, None, 0, 0.6])
    ax.set_xlabel(xylabel['x'])
    ax.set_ylabel(xylabel['y'])
    ax.autoscale(tight=True, axis='x')
    ax.legend()
    fig.tight_layout()

    lutils.save_fig(fig, name=name, path=path, format='pdf', close=False)

    return fig, ax


def plot_improvement(num_v=1000, M=15, lang='en', path='figures/discussion'):

    name = 'improvement_over'
    if 'pt' in lang:
        name = 'ptbr_' + name
        xylabel = dict(x='Variância $v$', y='Limite superior do passo')
    else:
        xylabel = dict(x='Variance $v$', y='Step-size upper bound')

    v = np.logspace(-2, 1, num_v)

    fig, ax = plt.subplots()

    beta = mutils.improvement_condition(v, M, hfa=0)
    ax.plot(v, beta, color='k')
    ax.grid(True)

    ax.autoscale(tight=True)
    ax.axis([None, None, v.min(), 5])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xylabel['x'])
    ax.set_ylabel(xylabel['y'])

    lutils.save_fig(fig, name=name, path=path, format='pdf', close=False)

    return fig, ax


def plot_optbeta(M=15, v=None, hfa=0, lang='en', path='figures/discussion'):

    name = 'optimal_beta'
    if 'pt' in lang:
        name = 'ptbr_' + name
        xylabel = dict(x='Jogos $k$', y='Passo ótimo')
    else:
        xylabel = dict(x='Games $k$', y='Optimum step size')

    if v is None:
        v = np.array([0.03, 0.3, 3])

    K = M * (M - 1)

    fig, ax = plt.subplots()
    for i in range(v.size):
        label = f'$v = {v[i]}$'
        k = np.arange(K) + 1

        beta_theory = mutils.optimal_beta(
            v=v[i],
            num_players=M,
            num_games=K,
            hfa=hfa,
        )
        ax.plot(k, beta_theory, label=label)

        beta_approx = mutils.optimal_beta_k(v[i], M, k, hfa=hfa)
        ax.plot(k, beta_approx, color=lutils.color_list[i], ls='--')
    ax.grid(True)

    ax.legend(borderaxespad=1.2, loc='lower left')
    ax.autoscale(tight=True, axis='x')
    ax.set_yscale('log')
    ax.set_xlabel(xylabel['x'])
    ax.set_ylabel(xylabel['y'])
    fig.tight_layout()

    lutils.save_fig(fig, name=name, path=path, format='pdf', close=False)

    return fig, ax


def plot_appendix(M=15, v=None, hfa=0, lang='en', path='figures/discussion'):

    name = 'beta_appendix'
    if 'pt' in lang:
        name = 'ptbr_' + name
        xylabel = dict(x='Jogos $k$', y='Passo ótimo')
    else:
        xylabel = dict(x='Games $k$', y='Optimum step size')

    if v is None:
        v = np.array([0.03, 0.3, 3])

    K = M * (M - 1)

    fig, ax = plt.subplots()
    for i in range(v.size):
        label = f'$v = {v[i]}$'
        k = np.arange(K) + 1

        beta_theory = mutils.optimal_beta(
            v=v[i],
            num_players=M,
            num_games=K,
            hfa=hfa,
        )
        ax.plot(k, beta_theory, label=label)

        beta_approx = mutils.optimal_beta_k(
            v[i], M, k, hfa=hfa, method='taylor')
        ax.plot(k, beta_approx, color=lutils.color_list[i], ls='--')
    ax.grid(True)

    ax.legend(loc='upper right')
    ax.autoscale(tight=True, axis='x')
    ax.set_yscale('log')
    ax.set_xlabel(xylabel['x'])
    ax.set_ylabel(xylabel['y'])
    fig.tight_layout()

    lutils.save_fig(fig, name=name, path=path, format='pdf', close=False)

    return fig, ax


plot_tau(lang='en')
plot_tau(lang='pt')

plot_lmin(lang='en')
plot_lmin(lang='pt')

plot_lex(lang='en')
plot_lex(lang='pt')

plot_improvement(lang='en')
plot_improvement(lang='pt')

plot_optbeta(lang='en')
plot_optbeta(lang='pt')

plot_appendix(lang='en')
plot_appendix(lang='pt')
