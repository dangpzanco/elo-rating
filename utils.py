import numpy as np
import numpy.random as rnd
import numba
from scipy import signal

import pathlib
import zarr
import tqdm

import matplotlib.pyplot as plt

import model_utils as mutils


# numba.config.THREADING_LAYER = 'omp'
# numba.config.THREADING_LAYER = 'threadsafe'
# numba.config.THREADING_LAYER = 'forksafe'
# numba.config.THREADING_LAYER = 'workqueue'
numba.config.NUMBA_NUM_THREADS = 4


@numba.njit(cache=True)
def np_clip(x, min_val, max_val):
    return np.maximum(np.minimum(x, max_val), min_val)


@numba.njit(cache=True)
def sigmoid(z):
    h, _ = probs(z)
    return h


@numba.njit(cache=True)
def probs(z):
    a = np.exp(-z)
    h = 1.0 / (1.0 + a)

    return h, a * h


@numba.njit(cache=True)
def to_onehot(y):
    h = (y == 1)
    a = (y == 0)
    return h, a


@numba.njit(cache=True)
def hessian(z):
    h, a = probs(z)
    c = h * a
    return c


@numba.njit(cache=True)
def log_loss(y, z_hat, eps=1e-15):
    z_h, z_a = probs(z_hat)

    # Clip [equivalent to np.clip(z, eps, 1 - eps)]
    z_h = np_clip(z_h, eps, 1 - eps)
    z_a = np_clip(z_a, eps, 1 - eps)

    # Convert to one-hot
    h, a = to_onehot(y)

    loss = h * np.log(z_h) + a * np.log(z_a)
    return -loss


@numba.njit(cache=True)
def acc_metric(y, y_hat):
    acc = (y == (y_hat > 0.5)).mean()
    return acc


@numba.njit(cache=True)
def taylor_loss(x, y, z, c, e, theta_hat, theta_star):
    num_games = y.size
    loss = np.zeros(num_games)
    Lmin = log_loss(y, z)

    for k in range(num_games):
        grad = e[k] * x[k, :]
        theta = theta_hat[k, :] - theta_star
        loss[k] = Lmin[k] + theta @ grad + 0.5 * c[k] * (x[k, :] @ theta) ** 2

    return loss


@numba.njit(cache=True)
def get_players(num_players, mu=0.0, v=1.0):
    theta = rnd.normal(mu, np.sqrt(v), num_players)
    return theta


@numba.njit(cache=True)
def get_schedules(num_games, num_players):
    x = np.zeros((num_games, num_players))
    for m in range(num_games):
        i = j = 0
        while i == j:
            i = rnd.randint(num_players)
            j = rnd.randint(num_players)
        x[m, i] = 1
        x[m, j] = -1

    return x


@numba.njit(cache=True)
def get_uniform_schedules(num_players):
    num_games = num_players * (num_players - 1)
    x = np.zeros((num_games, num_players))
    k = 0
    for i in range(num_players):
        for j in range(num_players):
            if i == j:
                continue
            x[k, i] = 1
            x[k, j] = -1
            k += 1

    return x


@numba.njit(cache=True)
def get_games(z):
    rng = rnd.rand(*z.shape)
    p_h, p_a = probs(z)

    h = rng < p_h
    d = rng > p_a + p_h
    y = 1.0 * h + 0.5 * d
    return y


@numba.njit(cache=True)
def sgd_logistic(num_games, num_players, x, y, theta0, eta):
    theta = np.empty((num_games, num_players))
    theta[0, ] = theta0.copy()
    for k in range(num_games - 1):
        g = sigmoid(x[k, ] @ theta[k, ]) - y[k]
        dL = g * x[k, ]
        theta[k + 1, ] = theta[k, ] - eta * dL

    return theta


@numba.njit(cache=True)
def sgd_taylor(num_games, num_players, x, c, e, theta_star, theta0, eta):
    theta = np.empty((num_games, num_players))
    theta[0, ] = theta0.copy()
    for k in range(num_games - 1):
        g = c[k] * x[k, ] @ (theta[k, ] - theta_star) - e[k]
        dL = g * x[k, ]
        theta[k + 1, ] = theta[k, ] - eta * dL

    return theta


@numba.njit(parallel=True, cache=True)
def compute_expectations(
    num_games, num_players,
    theta0, eta,
    mu=0, v=1, num_samples=1000
):
    loss_logist = np.zeros(num_games)
    loss_taylor = np.zeros(num_games)
    metric_logist = np.zeros(num_games)
    metric_taylor = np.zeros(num_games)
    theta_logist = np.zeros((num_games, num_players))
    theta_taylor = np.zeros((num_games, num_players))
    theta_logist2 = np.zeros((num_games, num_players))
    theta_taylor2 = np.zeros((num_games, num_players))
    for i in numba.prange(num_samples):
        theta_star = get_players(num_players, mu, v)
        x = get_schedules(num_games, num_players)
        z = x @ theta_star
        p = sigmoid(z)
        c = hessian(z)
        y = get_games(z)
        e = y - p

        if theta0 == 'zeros':
            theta_init = mu + np.zeros(num_players)
        elif theta0 == 'theta_star':
            theta_init = theta_star.copy()
        elif theta0 == 'rand':
            theta_init = get_players(num_players, mu, v)

        # weights_logist, weights_taylor = sgd_full(
        #     num_games, num_players, theta_star, theta_init, x, y, c, e, eta)

        weights_logist = sgd_logistic(
            num_games, num_players, x, y, theta_init, eta)
        weights_taylor = sgd_taylor(
            num_games, num_players, x, c, e, theta_star, theta_init, eta)

        # E[theta]
        theta_logist += (weights_logist - theta_star.reshape(1, -1))
        theta_taylor += (weights_taylor - theta_star.reshape(1, -1))
        # theta_logist += weights_logist
        # theta_taylor += weights_taylor

        # E[theta^2]
        theta_logist2 += (weights_logist - theta_star.reshape(1, -1)) ** 2
        theta_taylor2 += (weights_taylor - theta_star.reshape(1, -1)) ** 2

        # E[L(theta)]
        z_hat = np.sum(weights_logist * x, axis=-1)
        loss_logist += log_loss(y, z_hat)
        # metric_logist += log_loss(p, z_hat)
        metric_logist += sigmoid(z_hat) - y

        z_hat = np.sum(weights_taylor * x, axis=-1)
        loss_taylor += taylor_loss(x, y, z, c, e,
                                   weights_taylor, theta_star)
        # metric_taylor += log_loss(p, z_hat)
        metric_taylor += c * z_hat - e

    loss_logist /= num_samples
    loss_taylor /= num_samples
    theta_logist /= num_samples
    theta_taylor /= num_samples
    theta_logist2 /= num_samples
    theta_taylor2 /= num_samples
    metric_logist /= num_samples
    metric_taylor /= num_samples

    results = (
        (theta_logist, theta_taylor),
        (theta_logist2, theta_taylor2),
        (loss_logist, loss_taylor),
        (metric_logist, metric_taylor),
    )

    return results


@numba.njit(parallel=False, cache=True)
def get_expectations(
    theta0, theta_star, num_games, eta, num_samples=100
):
    num_players = theta_star.size
    theta = np.zeros((num_games, num_players))
    theta2 = np.zeros((num_games, num_players))
    loss = np.zeros(num_games)
    for j in numba.prange(num_samples):
        # theta_star = get_players(num_players, mu, v)
        x = get_schedules(num_games, num_players)
        z = x @ theta_star
        y = get_games(z)
        # c = hessian(z)
        # p = sigmoid(z)
        # e = y - p

        weights = sgd_logistic(
            num_games, num_players, x, y, theta0, eta)

        # E[theta]
        theta += weights

        # E[theta^2]
        theta2 += (weights - theta_star.reshape(1, -1)) ** 2

        # E[L(theta)]
        z_hat = np.sum(weights * x, axis=-1)
        loss += log_loss(y, z_hat)

    theta /= num_samples
    loss /= num_samples
    theta2 /= num_samples

    return theta, loss, theta2


# @numba.njit(parallel=False, cache=True)
def beta_expectations(
    theta0, theta_star, num_games, beta, num_samples=100
):
    num_players = theta_star.size
    num_beta = beta.size
    # theta = np.zeros((num_beta, num_games, num_players))
    var = np.zeros((num_beta, num_games))
    loss = np.zeros((num_beta, num_games))
    # for i in numba.prange(num_beta):
    #     for j in numba.prange(num_samples):
    for i in tqdm.trange(num_beta):
        for j in tqdm.trange(num_samples, leave=False):
            # theta_star = get_players(num_players, mu, v)
            x = get_schedules(num_games, num_players)
            z = x @ theta_star
            y = get_games(z)

            weights = sgd_logistic(
                num_games, num_players, x, y, theta0, beta)

            # # E[theta]
            # theta[i, ] += weights

            # E[theta^T theta]
            # var += (weights - theta_star.reshape(1, -1)) ** 2
            theta_tilde = weights - theta_star.reshape(1, -1)
            var[i, ] += np.sum(theta_tilde ** 2, axis=-1)

            # E[L(theta)]
            z_hat = np.sum(weights * x, axis=-1)
            loss[i, ] += log_loss(y, z_hat)

    # theta /= num_samples
    loss /= num_samples
    var /= num_samples

    # return theta, loss, var
    return var, loss



@numba.njit(parallel=False, cache=True)
def learning_expectations(
    theta0, theta_star, num_games, eta, num_samples=100
):
    num_players = theta_star.size

    # theta_star = get_players(num_players, mu, v)
    # theta0 = mu + np.zeros(num_players)

    theta_logist = np.zeros((num_games, num_players))
    theta_taylor = np.zeros((num_games, num_players))
    theta2_logist = np.zeros((num_games, num_players))
    theta2_taylor = np.zeros((num_games, num_players))
    loss_logist = np.zeros(num_games)
    loss_taylor = np.zeros(num_games)
    for j in numba.prange(num_samples):
        # theta_star = get_players(num_players, mu, v)
        x = get_schedules(num_games, num_players)
        z = x @ theta_star
        p = sigmoid(z)
        c = hessian(z)
        y = get_games(z)
        e = y - p

        weights_logist = sgd_logistic(
            num_games, num_players, x, y, theta0, eta)
        weights_taylor = sgd_taylor(
            num_games, num_players, x, c, e, theta_star, theta0, eta)

        # E[theta]
        theta_logist += weights_logist
        theta_taylor += weights_taylor

        # E[theta^2]
        theta2_logist += (weights_logist - theta_star.reshape(1, -1)) ** 2
        theta2_taylor += (weights_taylor - theta_star.reshape(1, -1)) ** 2

        # E[L(theta)]
        z_hat = np.sum(weights_logist * x, axis=-1)
        loss_logist += log_loss(y, z_hat)
        loss_taylor += taylor_loss(x, y, z, c, e, weights_taylor, theta_star)

    theta_logist /= num_samples
    theta_taylor /= num_samples
    theta2_logist /= num_samples
    theta2_taylor /= num_samples
    loss_logist /= num_samples
    loss_taylor /= num_samples

    return (theta_logist, theta_taylor), (theta2_logist, theta2_taylor), (loss_logist, loss_taylor)


@numba.njit(parallel=False, cache=True)
def learning_expectations2(
    num_players, num_games, eta, mu=0, v=1, num_samples=100
):
    theta0 = mu * np.ones(num_players)

    theta_logist = np.zeros((num_games, num_players))
    theta_taylor = np.zeros((num_games, num_players))
    theta2_logist = np.zeros((num_games, num_players))
    theta2_taylor = np.zeros((num_games, num_players))
    loss_logist = np.zeros(num_games)
    loss_taylor = np.zeros(num_games)
    for j in numba.prange(num_samples):
        theta_star = get_players(num_players, mu, v)
        x = get_schedules(num_games, num_players)
        z = x @ theta_star
        p = sigmoid(z)
        c = hessian(z)
        y = get_games(z)
        e = y - p

        weights_logist = sgd_logistic(
            num_games, num_players, x, y, theta0, eta)
        weights_taylor = sgd_taylor(
            num_games, num_players, x, c, e, theta_star, theta0, eta)

        # E[theta]
        theta_logist += (weights_logist - theta_star.reshape(1, -1))
        theta_taylor += (weights_taylor - theta_star.reshape(1, -1))

        # E[theta^2]
        theta2_logist += (weights_logist - theta_star.reshape(1, -1)) ** 2
        theta2_taylor += (weights_taylor - theta_star.reshape(1, -1)) ** 2

        # E[L(theta)]
        z_hat = np.sum(weights_logist * x, axis=-1)
        loss_logist += log_loss(y, z_hat)
        loss_taylor += taylor_loss(x, y, z, c, e, weights_taylor, theta_star)

    theta_logist /= num_samples
    theta_taylor /= num_samples
    theta2_logist /= num_samples
    theta2_taylor /= num_samples
    loss_logist /= num_samples
    loss_taylor /= num_samples

    return (theta_logist, theta_taylor), (theta2_logist, theta2_taylor), (loss_logist, loss_taylor)



class Experiment():
    """Some Information about Experiment"""

    def __init__(self, config, readonly=True):
        super(Experiment, self).__init__()
        self.config = config
        self.readonly = readonly
        self.sgd_config = self.config['sgd']

        self.attrs = dict(**self.sgd_config)
        self.exp_name = self.config['name']

        self.out_folder = pathlib.Path(config['out_folder'])
        mode = 'r' if self.readonly else 'a'
        self.zarr_root = zarr.open_group(str(self.out_folder), mode=mode)
        self.exp_group = self.zarr_root.require_group(self.exp_name)

        self.num_games = self.sgd_config['num_games']
        self.num_players = self.sgd_config['num_players']

        self.result_keys = ['theta', 'theta2', 'loss']
        self.result_shapes = dict(
            theta=(self.num_games, self.num_players),
            theta2=(self.num_games, self.num_players),
            loss=(self.num_games, ),
        )

    def run(self):

        if self.readonly:
            raise Exception(
                "Argument `readonly` was set to True, \
                run() method is not allowed."
            )

        eta_values = eval(self.sgd_config['eta'])
        num_values = len(eta_values)

        self.exp_group.attrs.update(**self.attrs)
        self.exp_group.require_dataset(
            'eta',
            data=eta_values,
            shape=eta_values.shape
        )

        print(self.sgd_config)
        config = self.sgd_config.copy()
        if self.sgd_config['constant_theta']:
            theta_star = get_players(
                self.sgd_config['num_players'],
                self.sgd_config['mu'],
                self.sgd_config['v']
            )
            theta0 = self.sgd_config['mu'] * np.ones_like(theta_star)
            self.exp_group.require_dataset(
                'theta_star',
                data=theta_star,
                shape=theta_star.shape
            )
            self.exp_group.require_dataset(
                'theta0',
                data=theta0,
                shape=theta0.shape
            )
            config = dict(
                theta0=theta0,
                theta_star=theta_star,
                num_games=self.sgd_config['num_games'],
                eta=None,
                num_samples=self.sgd_config['num_samples']
            )
            expectations_func = learning_expectations
        else:
            config = dict(
                num_players=self.sgd_config['num_players'],
                num_games=self.sgd_config['num_games'],
                eta=None,
                mu=self.sgd_config['mu'],
                v=self.sgd_config['v'],
                num_samples=self.sgd_config['num_samples']
            )
            expectations_func = learning_expectations2

        # Chunks of 4 Megabytes
        chunk_memory = 4 * 2**20
        chunk_size = chunk_memory // (8 * self.num_games * self.num_players) + 1

        logistic_group = self.exp_group.require_group('logistic')
        for name, shape in self.result_shapes.items():
            logistic_group.require_dataset(
                name, (num_values, *shape),
                dtype='float64',
                chunks=(chunk_size, *shape)
            )

        taylor_group = self.exp_group.require_group('taylor')
        for name, shape in self.result_shapes.items():
            taylor_group.require_dataset(
                name, (num_values, *shape),
                dtype='float64',
                chunks=(chunk_size, *shape)
            )

        for i, eta in enumerate(tqdm.tqdm(eta_values)):
            config['eta'] = eta
            results = expectations_func(**config)

            # Save results in Zarr arrays
            for k, item in enumerate(results):
                key = self.result_keys[k]
                logistic_group[key][i, ] = item[0]
                taylor_group[key][i, ] = item[1]


class ExperimentPlotting():
    """Some Information about ExperimentPlotting"""

    def __init__(self, path, exp_name):
        super(ExperimentPlotting, self).__init__()
        self.path = pathlib.Path(path)
        self.exp_name = exp_name

        self.zarr_root = zarr.open_group(str(self.path), mode='r')
        self.exp_group = self.zarr_root[self.exp_name]
        self.attrs = dict(self.exp_group.attrs)

        self.label_dict = dict(
            theta=r'$E[\tilde\theta_k]$',
            theta2=r'$E[\tilde\theta_k^2]$',
            loss=r'$E[l_k(\theta)]$',
            metric='metric',
            eta=r'$\eta$'
        )

    def plot_eta(self, name='theta', player_mode='index-0', iter_index='all', ax=None):

        eta = self.exp_group.eta[:]
        logistic_data = self.exp_group.logistic[name][:]
        taylor_data = self.exp_group.taylor[name][:]
        shape = taylor_data.shape

        # How many iterations to consider?
        if iter_index == 'all':
            iter_index = np.arange(shape[1])
        elif 'last' in iter_index:
            num = int(iter_index.split('-')[-1])
            iter_index = np.arange(num) + (shape[1] - num)

        # Which players to consider?
        if len(shape) == 3:
            if 'index' in player_mode:
                index = int(player_mode.split('-')[-1])
                logistic_data = logistic_data[:, iter_index, index].mean(axis=-1)
                taylor_data = taylor_data[:, iter_index, index].mean(axis=-1)
            elif player_mode == 'all':
                logistic_data = logistic_data[:, iter_index, :].mean(axis=(-1, -2))
                taylor_data = taylor_data[:, iter_index, :].mean(axis=(-1, -2))
            elif isinstance(player_mode, np.ndarray):
                logistic_data = logistic_data[:, iter_index, player_mode].mean(axis=-1)
                taylor_data = taylor_data[:, iter_index, player_mode].mean(axis=-1)
        else:
            logistic_data = logistic_data[:, iter_index].mean(axis=-1)
            taylor_data = taylor_data[:, iter_index].mean(axis=-1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        logistic_data = self.filter_inf(logistic_data)
        taylor_data = self.filter_inf(taylor_data)

        ax.plot(eta, logistic_data, label='Original')
        ax.plot(eta, taylor_data, label='Taylor approximation')
        if name == 'loss':
            if self.exp_group.attrs['constant_theta']:
                theta_star = self.exp_group.theta_star[:]
                theta0 = self.exp_group.theta0[:]
                Lmin, Lex = mutils.loss_expectations_sample(
                    theta_star=theta_star,
                    theta0=theta0,
                    eta=eta,
                    num_games=self.attrs['num_games'],
                )
            else:
                Lmin, Lex = mutils.loss_expectations(
                    eta=eta,
                    num_players=self.attrs['num_players'],
                    num_games=self.attrs['num_games'],
                    mu=self.attrs['mu'],
                    v=self.attrs['v'],
                )
            analytical_data = Lmin + Lex
            ax.plot(eta, analytical_data, label='Analytical model')

            eta_max1 = mutils.stability_limit(self.attrs['v'])
            eta_max2 = mutils.stability_limit2(self.attrs['v'])
            ax.axvline(eta_max1, label='Stability Limit 1', ls='--', c='tab:red')
            ax.axvline(eta_max2, label='Stability Limit 2', ls='--', c='tab:purple')
        ax.axis([None, None, -1, 10])
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_xlabel(self.label_dict['eta'], fontsize=14)
        ax.set_ylabel(self.label_dict[name], fontsize=14)
        fig.tight_layout()

        return fig, ax

    def plot_iters(self, eta, name='theta', player_mode='index-0', ax=None):

        eta_values = self.exp_group.eta[:]
        eta_index = np.abs(eta_values - eta).argmin()
        eta = eta_values[eta_index]

        logistic_data = self.exp_group.logistic[name][:]
        taylor_data = self.exp_group.taylor[name][:]
        shape = taylor_data.shape

        # Which players to consider?
        if len(shape) == 3:
            if 'index' in player_mode:
                index = int(player_mode.split('-')[-1])
                logistic_data = logistic_data[eta_index, :, index]
                taylor_data = taylor_data[eta_index, :, index]
            elif player_mode == 'all':
                logistic_data = logistic_data[eta_index, :, :].mean(axis=-1)
                taylor_data = taylor_data[eta_index, :, :].mean(axis=-1)
            elif isinstance(player_mode, np.ndarray):
                logistic_data = logistic_data[eta_index, :, player_mode].mean(axis=-1)
                taylor_data = taylor_data[eta_index, :, player_mode].mean(axis=-1)
        else:
            logistic_data = logistic_data[eta_index, :]
            taylor_data = taylor_data[eta_index, :]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        num_games = shape[1]
        k = np.arange(num_games)

        logistic_data = self.filter_inf(logistic_data)
        taylor_data = self.filter_inf(taylor_data)

        ax.plot(k, logistic_data, label='Original', alpha=0.7)
        ax.plot(k, taylor_data, label='Taylor approximation', alpha=0.7)
        if name == 'loss':
            Lmin, Lex = mutils.loss_expectations(
                eta=eta,
                num_players=self.attrs['num_players'],
                num_games=k,
                theta0=self.attrs['theta0'],
                mu=self.attrs['mu'],
                v=self.attrs['v']
            )
            analytical_data = Lmin + Lex
            ax.plot(k, analytical_data, 'k', lw=4, label='Analytical model')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('$k$', fontsize=14)
        ax.set_ylabel(self.label_dict[name], fontsize=14)
        # ax.set_xscale('log')
        ax.set_title(f'$\\eta = ${eta:.3e}', fontsize=14)
        fig.tight_layout()

        return fig, ax

    def plot_etaloss(self, num_games=None, last_games=None, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        eta = self.exp_group.eta[:]
        logistic_loss = self.exp_group.logistic['loss'][:]
        taylor_loss = self.exp_group.taylor['loss'][:]
        shape = taylor_loss.shape

        if last_games is None and num_games is None:
            num_games = shape[1]
        elif num_games is None:
            num_games = np.arange(shape[1] - last_games, shape[1])

        logistic_loss = logistic_loss[:, num_games].mean(axis=-1)
        taylor_loss = taylor_loss[:, num_games].mean(axis=-1)

        logistic_loss = self.filter_inf(logistic_loss)
        taylor_loss = self.filter_inf(taylor_loss)

        ax.plot(eta, logistic_loss, label='Original')
        ax.plot(eta, taylor_loss, label='Taylor approximation')
        if self.exp_group.attrs['constant_theta']:
            theta_star = self.exp_group.theta_star[:]
            theta0 = self.exp_group.theta0[:]
            Lmin, Lex = mutils.loss_expectations_sample(
                theta_star=theta_star,
                theta0=theta0,
                eta=eta,
                num_games=self.attrs['num_games'],
            )
        else:
            Lmin, Lex = mutils.loss_expectations(
                eta=eta,
                num_players=self.attrs['num_players'],
                num_games=self.attrs['num_games'],
                mu=self.attrs['mu'],
                v=self.attrs['v'],
            )
        analytical_data = Lmin + Lex
        ax.plot(eta, analytical_data, label='Analytical model')

        eta_max1 = mutils.stability_limit(self.attrs['v'])
        eta_max2 = mutils.stability_limit2(self.attrs['v'])
        ax.axvline(eta_max1, label='Stability Limit 1',
                    ls='--', c='tab:red')
        ax.axvline(eta_max2, label='Stability Limit 2',
                    ls='--', c='tab:purple')
        ax.axis([None, None, -1, 10])
        ax.legend()
        ax.grid(True)
        ax.set_xscale('log')
        ax.set_xlabel(self.label_dict['eta'], fontsize=14)
        ax.set_ylabel(self.label_dict['loss'], fontsize=14)
        fig.tight_layout()

        return fig, ax

    def filter_inf(self, data, large_number=1e50):
        data = data.copy()
        data[np.abs(data) > large_number] = np.nan
        return data

    def mean_filter(self, x, p=10):
        b = np.ones(p) / p
        y = signal.lfilter(b, 1, x, axis=-1)
        y[:, :p] *= p / (1 + np.arange(p))
        return y

    def mean_filter2(self, x, p=10):
        b = np.ones(p) / p
        y = signal.lfilter(b, 1, x)
        y[:p, ] *= p / (1 + np.arange(p))
        return y

    def mean_filtfilt(self, x, p=10):
        b = np.ones(p) / p
        y = signal.filtfilt(b, 1, x, axis=-1)
        return y

if __name__ == '__main__':

    num_players = 10
    eta_min = 1e-4
    v = np.array([0.01, 0.1, 1.0])
    tau = mutils.time_constant(eta_min/2, v, num_players)

    # for i in range(v.size):
    #     num_games = int(5*tau[i])
    #     exp_config = dict(
    #         num_samples=100,
    #         num_games=num_games,
    #         num_players=num_players,
    #         eta='np.logspace(-4, 1, 100)',
    #         constant_theta=False,
    #         mu=0,
    #         v=v[i],
    #     )

    #     config = dict(
    #         sgd=exp_config,
    #         out_folder=r'D:\Datasets\simulations\saved_results2',
    #         name=f'eta-v={v[i]}'
    #     )

    #     exp_runner = Experiment(config, readonly=False)
    #     exp_runner.run()

