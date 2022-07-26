import numpy as np
import numpy.random as rnd
import numba

import dask.array as da
from dask.diagnostics import ProgressBar

import tqdm
import matplotlib.pyplot as plt

import utils


@numba.njit(cache=True)
def simulate_expectations(num_players, num_games, num_samples, mu=0, v=1):
    c_1 = 0.0
    c_2 = 0.0
    Lmin = 0.0
    for i in numba.prange(num_samples):
        theta = utils.get_players(num_players, mu, v)
        x = utils.get_schedules(num_games, num_players)
        z = x @ theta
        y = utils.get_games(z)

        c = utils.hessian(z)
        loss_vec = utils.log_loss(y, z, eps=1e-15)

        c_1 += c.mean()
        c_2 += (c ** 2).mean()
        Lmin += loss_vec.mean()
    c_1 /= num_samples
    c_2 /= num_samples
    Lmin /= num_samples

    return c_1, c_2, Lmin


def simulate_expectations_sample(theta, num_games, num_samples, hfa=0):
    num_players = theta.size
    R = 0.0
    hX = 0.0
    h1 = 0.0
    h2 = 0.0
    Lmin = 0.0
    for i in tqdm.trange(num_samples, leave=False):
        x = utils.get_schedules(num_games, num_players)
        z = x @ theta + hfa
        y = utils.get_games(z)
        X = x[:, :, None] @ x[:, None, :]

        h = utils.hessian(z)
        loss_vec = utils.log_loss(y, z, eps=1e-15)

        R += X.mean(axis=0)
        hX += (h[:, None, None] * X).mean(axis=0)
        h1 += h.mean()
        h2 += (h ** 2).mean()
        Lmin += loss_vec.mean()
    R /= num_samples
    hX /= num_samples
    h1 /= num_samples
    h2 /= num_samples
    Lmin /= num_samples

    results = dict(
        R=R,
        hX=hX,
        h1=h1,
        h2=h2,
        Lmin=Lmin,
    )

    return results


def simulate_expectations_batch(hfa, v, num_players, num_games, num_samples):
    num_hfa = hfa.size
    num_v = v.size

    std = np.sqrt(v)
    h1 = np.zeros([num_hfa, num_v])
    h2 = np.zeros([num_hfa, num_v])
    lmin = np.zeros([num_hfa, num_v])
    for i in tqdm.trange(num_samples, leave=False):
        x = utils.get_schedules(num_games, num_players)

        theta = rnd.normal(0, std[:, None], (num_v, num_players))
        z = np.einsum('ij,kj->ki', x, theta)[None, ] + hfa[:, None, None]

        # y = utils.get_games(z)
        # loss = utils.log_loss(y, z, eps=1e-15)
        loss = utils.log_loss(z, z, eps=1e-15)

        h = utils.hessian(z)
        h1 += h.mean(axis=-1)
        h2 += (h ** 2).mean(axis=-1)
        lmin += loss.mean(axis=-1)
    h1 /= num_samples
    h2 /= num_samples
    lmin /= num_samples

    results = dict(
        h1=h1,
        h2=h2,
        lmin=lmin,
        hfa=hfa,
        v=v
    )

    return results



def analytical_expectations(v, hfa=0):

    v_z = 2 * v
    mu_z = hfa

    # h_1
    f_1 = 1 / 4
    v_1 = 2
    bias = np.exp(-0.5*mu_z**2 / (v_z + v_1))
    h_1 = f_1 / np.sqrt(1 + v_z / v_1) * bias

    # h_2
    f_2 = 1 / 16
    v_2 = 1
    bias = np.exp(-0.5*mu_z**2 / (v_z + v_2))
    h_2 = f_2 / np.sqrt(1 + v_z / v_2) * bias

    # L_{min}
    f_l = np.log(2)
    v_l = 4 * np.log(2)
    bias = np.exp(-0.5*mu_z**2 / (v_z + v_l))
    Lmin = f_l / np.sqrt(1 + v_z / v_l) * bias

    return h_1, h_2, Lmin


def time_constant(beta, v, num_players, hfa=0, approx=False, first_order=False):
    M = num_players
    h, h2, _ = analytical_expectations(v, hfa=hfa)

    if first_order:
        alpha = 1 - 2 / (M - 1) * beta * h
    else:
        alpha = 1 - 4 / (M - 1) * beta * (h - beta * h2)

    if approx:
        tau = 1 / (1 - alpha)
    else:
        tau = -1 / np.log(alpha)
    return tau


def stability_limit(v, hfa=0, num_players=None, first_order=False):
    h, h2, _ = analytical_expectations(v, hfa)
    if first_order:
        beta = (num_players - 1) / h
    else:
        beta = h / h2
    return beta


def stability_limit2(v, hfa=0):
    h, h2, _ = analytical_expectations(v, hfa)
    eta_max = 2*v * h / (h + 2*v * h2)
    return eta_max


def improvement_condition(v, M, hfa=0):
    h, h2, _ = analytical_expectations(v, hfa)
    limit1 = h/h2
    limit2 = 2*v/(1 - 1/M)
    improv = 1/(1/limit1 + 1/limit2)
    return improv


def corr_matrix(M):
    I = np.eye(M)
    ones = np.ones([M, M])
    R = 2 / (M - 1) * (I - ones/M)
    return R


def fisher_matrix(theta, hfa):
    M = theta.size
    factor = M * (M - 1)

    ones = np.ones(M)
    z = theta[:, None] - theta[None, ]
    h1 = utils.hessian(z + hfa)
    h2 = utils.hessian(-z + hfa)
    h = (h1 + h2) / factor

    H = np.diag(h @ ones) - h
    return H, h


def fisher_matrix_sample(theta, hfa):
    M = theta.size
    x = utils.get_uniform_schedules(M)
    z = x @ theta + hfa
    h = utils.hessian(z)

    K = M * (M - 1)
    H = x.T @ np.diag(h) @ x / K
    # H2 = x @ x.T @ np.diag(h**2) @ x @ x.T
    # # R = x.T @ x / K

    # X1 = np.einsum('im,in->imn', x, x)
    # R1 = X1.mean(axis=0)
    # X2 = np.einsum('imn,ink->imk', X1, X1)
    # R2 = X2.mean(axis=0)

    # h1 = np.einsum('im,i,in->imn', x, h, x)
    # H1 = h1.mean(axis=0)
    # h2 = np.einsum('imn,ink->imk', h1, h1)
    # H2 = h2.mean(axis=0)

    return H, h.mean()


def theta_expectation_exact(theta_star, theta0, eta, num_games, mu=0, hfa=0):
    num_players = theta_star.size

    I = np.eye(num_players)
    H, _ = fisher_matrix(theta_star, hfa)
    # _H, _, _ = fisher_matrix_sample(theta_star, hfa)
    A = I - eta * H

    # R = corr_matrix(num_players)
    # h, _, _ = analytical_expectations(theta_star.var(), hfa)

    theta = np.zeros([num_games + 1, num_players])
    theta[0, ] = theta0 - theta_star
    for i in range(1, num_games+1):
        theta[i, ] = A @ theta[i-1, ]
    theta += theta_star[None, ]
    return theta


# \E[\tilde\btheta_{k+1}] = (\bI - \eta \ov{h} \bR) E[\tilde\btheta_k].
def theta_expectation(theta_star, theta0, eta, num_games, mu=0, hfa=0):
    num_players = theta_star.size
    v = theta_star.var(ddof=1)
    h, _, _ = analytical_expectations(v, hfa=hfa)

    I = np.eye(num_players)
    R = corr_matrix(num_players)
    A = I - eta * h * R

    theta = np.zeros([num_games + 1, num_players])
    theta[0, ] = theta0 - theta_star
    for i in range(1, num_games+1):
        theta[i, ] = A @ theta[i-1, ]
    theta += theta_star[None, ]
    return theta


def mean_expectation(theta_star, beta, num_games, hfa=0):
    v = theta_star.var(ddof=1)
    h, _, _ = analytical_expectations(v, hfa=hfa)

    k = np.arange(num_games)
    M = theta_star.size
    alpha1 = 1 - 2/(M-1) * beta * h
    theta = (1 - alpha1**k[:, None]) * theta_star[None, ]

    return theta


def bias_expectation(beta, v, num_players, num_games, hfa=0):
    h, _, _ = analytical_expectations(v, hfa=hfa)
    M = num_players
    K = num_games

    bias0 = np.sqrt(M*v)
    alpha1 = 1 - 2/(M-1) * beta * h

    bias = bias0 * alpha1 ** K

    return bias


def loss_expectations(eta, v, num_players, num_games=None, theta0='zeros', mu=0, hfa=0):

    # `v` is the variance of \btheta^*, it can be estimated with:
    # v = theta_star.var(ddof=1)
    # where `ddof=1` means "1 degree of freedom",
    # which effectively is the sample variance with Bessel's correction

    c_1, c_2, Lmin = analytical_expectations(v, hfa=hfa)
    trHR_inf = eta * c_1 / (c_1 - eta * c_2)

    K = num_games
    if K is None:
        trHR = trHR_inf
    else:
        alpha = 1 - 4 / (num_players - 1) * eta * (c_1 - eta * c_2)
        # when θ₀ is μ, tr[H_0 R] = 2v
        if theta0 == 'zeros':
            trHR_0 = 2 * v
        elif theta0 == 'theta_star':
            trHR_0 = 0
        elif type(theta0) != str:
            ones = np.ones(num_players)
            I = np.eye(num_players)
            R = corr_matrix(num_players)
            A = np.outer(theta0, theta0) - mu * \
                (np.outer(theta0, ones) + np.outer(ones, theta0)) + v * I
            trHR_0 = np.trace(A @ R)
        trHR = alpha**K * (trHR_0 - trHR_inf) + trHR_inf

    Lex = 0.5 * c_1 * trHR

    return Lmin, Lex


def var_expectation(beta, v, num_players, num_games=None, hfa=0):

    # `v` is the variance of \btheta^*, it can be estimated with:
    # v = theta_star.var(ddof=1)
    # where `ddof=1` means "1 degree of freedom",
    # which effectively is the sample variance with Bessel's correction

    # v_k = E[(theta_k - theta_star)^T (theta_k - theta_star)]

    M = num_players
    K = num_games
    h, h2, _ = analytical_expectations(v, hfa=hfa)
    v_inf = 0.5 * (M - 1) * beta * h / (h - beta * h2)

    if K is None:
        v_k = v_inf
    else:
        alpha = 1 - 4 / (M - 1) * beta * (h - beta * h2)
        v_0 = M * v
        v_k = alpha**K * (v_0 - v_inf) + v_inf

    return v_k


def optimal_beta(v, num_players, num_games, hfa=0, num_beta=10000, beta_min=1e-5):
    h, h2, l_min = analytical_expectations(v, hfa)
    M = num_players

    num_games = np.array(num_games)
    if len(num_games.shape) == 0:
        games = np.arange(num_games) + 1
    else:
        games = num_games.copy()

    def func(beta, K, M, v, h, h2):
        v_0 = M * v
        factor = h - beta * h2
        v_inf = 0.5 * (M - 1) * beta * h / factor
        alpha = 1 - 4 / (M - 1) * beta * factor
        v_k = alpha**K * (v_0 - v_inf) + v_inf
        return v_k

    # beta0 = v * h / (h * (1 - 1/M) + 2*v * h2)
    # beta = np.logspace(-5, 1.1*np.log10(beta0), num_eta)
    # beta = np.logspace(-5, 5, num_eta)

    # beta_limit = stability_limit(v, hfa)
    beta_limit = stability_limit2(v, hfa)
    beta = np.logspace(np.log10(beta_min), np.log10(0.9*beta_limit), num_beta)
    # beta = np.hstack([beta, np.linspace(0.9*beta_limit, (1 - 1e-6)*beta_limit, num_eta)])

    beta = da.array(beta)
    games = da.array(games).rechunk((100,))
    v_k = func(beta[None, :], games[:, None], M, v, h, h2)

    with ProgressBar():
        beta_opt = beta[v_k.argmin(axis=-1)].compute()

    return beta_opt


def optimal_beta_k(v, num_players, num_games, hfa=0, method='best'):
    h, h2, _ = analytical_expectations(v, hfa)
    M = num_players
    K = num_games
    # beta_opt1 = 1 / ((1 - 1/M) / v + 2*h2/h)
    # beta_opt = 1 / ((1 - 1/M) / v + 2*h2/h + 4*h*(K-1)/(M-1))
    # beta_opt = 1 / (1/beta_opt1 + 4*h*(K-1)/(M-1))
    # beta_opt = 5 / (1/beta_opt + 4/beta_opt1)

    # Obtained via Taylor Series
    if method == 'taylor':
        beta_opt = 0.5 / ((1 - 1/M) / (2*v) + h2/h + 2*h*(K-1)/(M-1))

    # Obtained via proposed expression
    if method == 'best':
        beta_opt = 0.5 / ((1 - 1/M) / (2*v) + h2/h + 2*h2*(K-1)/(M-1))
        # beta_opt = 0.5 / ((1 - 1/M) / (2*v) + h2/h + 2*h**2*(K-1)/(M-1))

    return beta_opt


def loss_expectations_sample(theta_star, theta0, eta, num_games=None, hfa=0):

    num_players = theta_star.size
    v = theta_star.var(ddof=1)
    c_1, c_2, Lmin = analytical_expectations(v, hfa=hfa)
    trHR_inf = eta * c_1 / (c_1 - eta * c_2)

    K = num_games
    if K is None:
        trHR = trHR_inf
    else:
        theta_tilde = theta0 - theta_star
        H0 = np.outer(theta_tilde, theta_tilde)
        R = corr_matrix(num_players)
        trHR_0 = np.trace(H0 @ R)

        alpha = 1 - 4 / (num_players - 1) * eta * (c_1 - eta * c_2)
        trHR = alpha**K * (trHR_0 - trHR_inf) + trHR_inf

    Lex = 0.5 * c_1 * trHR
    return Lmin, Lex


def loss_expectations_inf(eta, v, mu=0, hfa=0):
    c_1, c_2, Lmin = analytical_expectations(v, hfa)
    trHR = eta * c_1 / (c_1 - eta * c_2)

    Lex = 0.5 * c_1 * trHR
    return Lmin, Lex


if __name__ == '__main__':

    beta = 0.5
    v = 5
    hfa = 0
    num_players = 10
    num_games = 1000
    k = np.arange(num_games)
    theta_star = utils.get_players(num_players, mu=0, v=v)
    v = theta_star.var(ddof=1)
    theta = mean_expectation(theta_star, beta, num_games, hfa)

    bias = np.linalg.norm(theta-theta_star, axis=-1)**2
    msd = var_expectation(beta, v, num_players, k, hfa)
    var = msd - bias

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(k, msd)
    ax.plot(k, var)
    ax.plot(k, bias)
    plt.show()

    print()
