import gc

import numpy as np
import numpy.random as rnd
from scipy import signal

import torch
import torch.nn.functional as F
import typing

import pathlib
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
import tqdm

import matplotlib.pyplot as plt

import model_utils as mutils

torch.set_default_dtype(torch.float64)


# @torch.jit.script
def sigmoid(z):
    return probs(z)[0]


@torch.jit.script
def probs(z):
    a = torch.exp(-z)
    h = 1.0 / (1.0 + a)

    return h, a * h


@torch.jit.script
def to_onehot(y):
    h = (y == 1)
    a = (y == 0)
    return h, a


# @torch.jit.script
def hessian(z):
    h, a = probs(z)
    c = h * a
    return c


def log_loss(y, z_hat, eps=1e-15):
    # z_h = torch.sigmoid(z_hat)
    # # Inplace clip [equivalent to z = np.clip(z, eps, 1 - eps)]
    # z_h.clamp_(eps, 1 - eps)
    # # Convert to one-hot
    # h, a = to_onehot(y)
    # loss = h * torch.log(z_h) + a * torch.log(1 - z_h)

    loss = F.binary_cross_entropy_with_logits(z_hat, y)
    return loss


def acc_metric(y, y_hat):
    acc = (y == (y_hat > 0.5)).mean()
    return acc


def taylor_loss(x, y, z, c, e, theta_hat, theta_star):
    num_games = y.size
    loss = np.zeros(num_games)
    Lmin = log_loss(y, z)

    for k in range(num_games):
        grad = e[k] * x[k, :]
        theta = theta_hat[k, :] - theta_star
        loss[k] = Lmin[k] + theta @ grad + 0.5 * c[k] * (x[k, :] @ theta) ** 2

    return loss


@torch.jit.script
def get_players(num_players: int, mu: float = 0.0, v: float = 1.0):
    theta = torch.normal(mu, torch.sqrt(v) * torch.ones(num_players, device='cpu'))
    return theta


@torch.jit.script
def get_schedules(num_games: int, num_players: int):

    # Generate all pairings with repetition
    i = torch.randint(num_players, (num_games,), dtype=torch.int64, device='cpu')
    j = torch.randint(num_players, (num_games,), dtype=torch.int64, device='cpu')
    equal = i == j
    num_equal = int(equal.sum())

    # Re-generate indexes for repeated pairings until done
    while num_equal > 0:
        i[equal] = torch.randint(num_players, (num_equal,), dtype=torch.int64, device='cpu')
        j[equal] = torch.randint(num_players, (num_equal,), dtype=torch.int64, device='cpu')
        equal = i == j
        num_equal = int(equal.sum())

    # Turn each index into a one-hot vector and subtract
    # i = 3 -> i = [0,0,0,1,...,0,0,0]
    # j = 2 -> i = [0,0,1,0,...,0,0,0]
    # x = i - j = [0,0,-1,1,...,0,0,0]
    x = F.one_hot(i, num_players) - F.one_hot(j, num_players)

    # return x.type(torch.get_default_dtype())
    return x


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


@torch.jit.script
def get_games(z):
    rng = torch.rand_like(z)
    p_h, p_a = probs(z)

    h = rng < p_h
    d = rng > p_a + p_h
    y = 1.0 * h + 0.5 * d
    return y


@torch.jit.script
def sgd_logistic(num_games: int, num_players: int,
                 x: torch.Tensor, y: torch.Tensor,
                 theta0: torch.Tensor, beta: torch.Tensor
                 ) -> torch.Tensor:
    theta = torch.empty(num_games, num_players)
    theta[0, ] = theta0
    for k in range(num_games - 1):
        g = torch.sigmoid(x[k, ] @ theta[k, ]) - y[k]
        dL = g * x[k, ]
        theta[k + 1, ] = theta[k, ] - beta * dL

    return theta


@torch.jit.script
def sgd_logistic_beta(
    num_games: int, num_players: int,
    x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    num_beta = int(beta.shape[0])
    theta = torch.zeros(num_games, num_players, num_beta, device='cpu')
    for k in range(num_games - 1):
        z = torch.einsum('i,ij->j', x[k, ], theta[k, ])
        g = torch.sigmoid(z) - y[k]
        delta = torch.einsum('i,j,j->ij', x[k, ], g, beta)
        theta[k + 1, ] = theta[k, ] - delta

    return theta


@torch.jit.script
def sgd_logistic_beta_var(
    num_games: int, num_players: int,
    x: torch.Tensor, y: torch.Tensor,
    beta: torch.Tensor, star: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    num_beta = int(beta.shape[0])
    theta = torch.zeros(num_players, num_beta, device='cpu')
    var = torch.zeros(num_games, num_beta, device='cpu')
    loss = torch.zeros(num_games, num_beta, device='cpu')
    for k in range(num_games):
        z = torch.einsum('i,ij->j', x[k, ], theta)
        g = torch.sigmoid(z) - y[k]
        delta = torch.einsum('i,j,j->ij', x[k, ], g, beta)

        theta_tilde = theta - star[:, None]
        var[k, ] = torch.einsum('ij,ij->j', theta_tilde, theta_tilde)
        loss[k, ] = F.binary_cross_entropy_with_logits(z, y[k].repeat(num_beta), reduction='none')
        theta = theta - delta

    return var, loss


def sgd_taylor(num_games, num_players, x, c, e, theta_star, theta0, beta):
    theta = torch.empty(num_games, num_players)
    theta[0, ] = theta0
    for k in range(num_games - 1):
        g = c[k] * x[k, ] @ (theta[k, ] - theta_star) - e[k]
        dL = g * x[k, ]
        theta[k + 1, ] = theta[k, ] - beta * dL

    return theta



# @torch.jit.script
def beta_expectations(
    num_players: int, v: float, num_games: int,
    beta: torch.Tensor, hfa: float = 0.0, num_samples: int = 100
):
    num_beta = int(beta.shape[0])
    num_samples = int(num_samples)
    var = torch.zeros(num_beta, num_games, device='cpu')
    loss = torch.zeros(num_beta, num_games, device='cpu')
    for j in tqdm.trange(num_samples):
        theta_star = get_players(num_players, 0.0, v)
        x = get_schedules(num_games, num_players).type(torch.get_default_dtype())
        z = x @ theta_star + hfa
        y = get_games(z)

        # Run algorithm in parallel all beta in parallel
        var_tmp, loss_tmp = sgd_logistic_beta_var(
            num_games, num_players, x, y, beta, theta_star)
        var += var_tmp.T
        loss += loss_tmp.T

    loss /= num_samples
    var /= num_samples

    if loss.device == 'cuda':
        loss = loss.cpu()
        var = var.cpu()

    return var.numpy(), loss.numpy()






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

        self.result_keys = ['var', 'loss']
        self.result_shapes = dict(
            var=(self.num_games, ),
            loss=(self.num_games, ),
        )

    def run(self):

        if self.readonly:
            raise Exception(
                "Argument `readonly` was set to True, \
                run() method is not allowed."
            )

        lr_values = eval(self.sgd_config['beta'])
        num_lr = len(lr_values)

        self.exp_group.attrs.update(**self.attrs)
        self.exp_group.require_dataset(
            'lr',
            data=lr_values.numpy(),
            shape=lr_values.numpy().shape
        )

        # Chunks of 4 Megabytes
        chunk_memory = 4 * 2**20
        chunk_size = chunk_memory // (8 * self.num_games) + 1

        for name, shape in self.result_shapes.items():
            self.exp_group.require_dataset(
                name, (num_lr, *shape),
                dtype='float64',
                chunks=(chunk_size, *shape)
            )

        print(self.sgd_config)
        config = self.sgd_config.copy()
        config['beta'] = lr_values

        var, loss = beta_expectations(**config)
        self.exp_group['var'][:] = var
        self.exp_group['loss'][:] = loss


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
            var=r'$\overline{v}_k$',
            loss=r'$\overline\ell_k$',
            lr=r'$\beta$'
        )

    def get_optlr(self, name='var'):
        lr = da.from_zarr(self.exp_group.lr)
        data = da.from_zarr(self.exp_group[name])

        with ProgressBar():
            lr_sim = lr[data.argmin(axis=0)].compute()

        analytical_lr = mutils.optimal_beta(
            v=self.attrs['v'],
            num_players=self.attrs['num_players'],
            num_games=self.attrs['num_games'],
            hfa=self.attrs['hfa'],
        )

        return lr_sim, analytical_lr

    def plot_optlr(self, name='var', ax=None):

        lr_sim, analytical_lr = self.get_optlr(name)
        num_games = lr_sim.size
        k = np.arange(num_games)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        ax.plot(k[1:-1], lr_sim[1:-1], label='Simulation')
        ax.plot(k[1:-1], analytical_lr[1:-1], label='Analytical model')

        ax.axis([1, num_games, None, None])
        ax.legend()
        ax.grid(True)
        ax.set_title(self.exp_name)
        ax.set_yscale('log')
        ax.set_xlabel('$k$', fontsize=14)
        ax.set_ylabel(self.label_dict['lr'], fontsize=14)
        fig.tight_layout()

        return fig, ax

    def plot_iters(self, lr, name='var', ax=None):

        lr_values = self.exp_group.eta[:]
        lr_index = np.abs(lr_values - lr).argmin()
        lr = lr_values[lr_index]

        logistic_data = self.exp_group.logistic[name][:]
        taylor_data = self.exp_group.taylor[name][:]
        shape = taylor_data.shape

        logistic_data = logistic_data[lr_index, :]
        taylor_data = taylor_data[lr_index, :]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:
            fig = ax.figure

        num_games = shape[1]
        k = np.arange(num_games)

        logistic_data = self.filter_inf(logistic_data)
        taylor_data = self.filter_inf(taylor_data)

        ax.plot(k, logistic_data, label='Simulation', alpha=0.7)
        if name == 'var':
            analytical_data = mutils.var_expectation(
                beta=lr,
                v=self.attrs['v'],
                num_players=self.attrs['num_players'],
                num_games=k,
                hfa=self.attrs['hfa'],
            )
        elif name == 'loss':
            Lmin, Lex = mutils.loss_expectations(
                eta=lr,
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
        ax.set_title(f'$\\beta = ${lr:.3e}', fontsize=14)
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

if __name__ == '__main__':

    # num_players = 100
    # num_games = 500 * num_players
    # v = 1
    # num_beta = 2000
    # beta = torch.logspace(-3, np.log10(4), num_beta, device='cpu')

    # # func = torch.jit.script(beta_expectations)
    # var, loss = beta_expectations(
    #     num_players, v, num_games, beta, num_samples=100
    # )
    # beta = beta.numpy()
    # np.savez('exp3.npz', var=var, loss=loss,
    #          beta=beta, v=v,
    #          games=num_games, players=num_players)

    # M = num_players
    # beta_opt = mutils.optimal_beta(v, num_players, num_games, hfa=0)
    # games = np.arange(num_games - 1) + 1
    # plt.plot(games / M, beta[var.argmin(axis=0)][1:])
    # plt.plot(games / M, beta[loss.argmin(axis=0)][1:], alpha=0.3)
    # plt.plot(games / M, beta_opt[:-1], '--')
    # plt.yscale('log')
    # plt.show()


    num_players = 100
    v = np.array([0.01, 0.1, 1, 10])
    num_beta = 2000
    num_games = 500 * num_players
    hfa = 0.5

    for i in range(v.size):
        # num_games = int(5*tau[i])
        exp_config = dict(
            num_samples=100,
            num_games=num_games,
            num_players=num_players,
            beta=f"torch.logspace(-3, np.log10(4), {num_beta})",
            hfa=hfa,
            v=v[i],
        )

        config = dict(
            sgd=exp_config,
            out_folder=r'/home/daniel/data/Datasets/simulations/results_torch',
            name=f'v={v[i]},hfa={hfa}'
        )

        exp_runner = Experiment(config, readonly=False)
        exp_runner.run()


    print()
