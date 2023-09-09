import warnings
import numpy as np
import numpy.random as rnd
import pandas as pd
import pathlib

import numba
import tqdm

import utils
import model_utils as mutils

import zarr

import sklearn.preprocessing as skprep
import sklearn.model_selection as skselect
import sklearn.linear_model as skmodel
import sklearn.metrics as skmetrics
from warnings import simplefilter

# ignore all future warnings (for sklearn)
simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')


def fill_group(group, **out_dict):
    for key, item in out_dict.items():
        if key in group.keys():
            group[key][:] = item
        else:
            group.require_dataset(
                key,
                data=item,
                shape=item.shape,
                dtype=item.dtype
            )


def sparse2onehot(home, away, num_players='auto'):
    if num_players != 'auto':
        num_players = np.arange(num_players)
    enc = skprep.OneHotEncoder(categories=num_players)
    home_ = enc.fit_transform(home[:, None])
    away_ = enc.fit_transform(away[:, None])
    X = (home_ - away_).todense()

    return X


@numba.njit()
def player_stats(home, away, results, num_players=None):
    if num_players is None:
        num_players = np.unique(np.hstack((home, away))).size
    home_wins = np.zeros(num_players, dtype='int64')
    away_wins = np.zeros(num_players, dtype='int64')
    draws = np.zeros(num_players, dtype='int64')
    games = np.zeros(num_players, dtype='int64')
    for i in range(results.size):
        h = home[i]
        a = away[i]
        y = results[i]

        games[h] += 1
        games[a] += 1

        h_result = 1 * (y == 1)
        a_result = 1 * (y == 0)
        d_result = 1 * (y == 0.5)

        home_wins[h] += h_result
        away_wins[a] += a_result
        draws[h] += d_result
        draws[a] += d_result

    return home_wins, away_wins, draws, games


@numba.njit()
def schedule_matrix(home, away, num_players=None):
    if num_players is None:
        num_players = np.unique(np.hstack((home, away))).size
    num_games = home.size
    x = np.zeros((num_games, num_players), dtype='float64')
    for i in range(num_games):
        x[i, home[i]] = 1.0
        x[i, away[i]] = -1.0
    return x


@numba.njit()
def corr_matrix(home, away, num_players=None):
    if num_players is None:
        num_players = np.unique(np.hstack((home, away))).size
    num_games = home.size
    R = np.zeros((num_players, num_players), dtype='float64')
    for i in range(num_games):
        h = home[i]
        a = away[i]
        R[h, h] += 1.0
        R[a, a] += 1.0
        R[h, a] += -1.0
        R[a, h] += -1.0

    return R / num_games


def convert_dataset(zarr_path='superlega', season=2018, root_folder=None, overwrite=False):

    if root_folder is None:
        root_folder = r'D:\Datasets\SuperLega'
    root_folder = pathlib.Path(root_folder)
    filename = f'SuperLega-{season}-{season+1}.csv'
    path = root_folder / filename

    df = pd.read_csv(path)
    # df = df.dropna()
    # date,home_team,home_score,away_team,away_score
    date = pd.to_datetime(df['date']).values.astype('datetime64[D]')
    home_team = df['home_team'].values.astype('str')
    away_team = df['away_team'].values.astype('str')

    # Team pairs
    team_names = np.array(sorted(df['home_team'].unique()))
    team_dict = dict(zip(team_names, np.arange(team_names.size)))
    home = df['home_team'].replace(team_dict).values
    away = df['away_team'].replace(team_dict).values

    # Game results
    home_goals = df['home_score'].values
    away_goals = df['away_score'].values
    y = 1.0*(home_goals > away_goals)
    y[home_goals == away_goals] = 0.5
    print(season, np.unique(y))

    data_dict = dict(
        team_names=team_names,
        date=date,
        home_goals=home_goals,
        away_goals=away_goals,
        home_team=home_team,
        away_team=away_team,
        home=home,
        away=away,
        y=y,
    )

    zarr_root = zarr.open_group(str(zarr_path), mode='a')
    group = zarr_root.require_group(f'season-{season}', overwrite=overwrite)
    for key, item in data_dict.items():
        group.create_dataset(
            key,
            data=item,
            shape=item.shape,
            dtype=item.dtype,
            overwrite=True
        )


class SuperLegaRatings():
    """Some Information about SuperLegaRatings"""

    def __init__(self, zarr_path='superlega', season=2018):
        super(SuperLegaRatings, self).__init__()
        self.zarr_path = pathlib.Path(zarr_path)
        self.root_group = zarr.open_group(str(self.zarr_path), mode='a')
        self.group = self.root_group[f'season-{season}']

        try:
            self.group.theta_star
        except Exception:
            self.fit_star()

    def _fit_star(self, eta=1e-3, num_epochs=10000, theta0=None):

        if theta0 is None:
            theta0 = np.zeros(self.group.team_names.size)

        theta_batch, _ = batch_descent(
            self.group.home[:],
            self.group.away[:],
            self.group.y[:],
            theta0=theta0,
            hfa=self.hfa,
            eta=eta,
            num_iters=num_epochs
        )
        theta_star = theta_batch[-1, ].copy()

        self.group.attrs['v'] = theta_star.var(ddof=1)
        self.group.attrs['hfa'] = self.hfa

        fill_group(
            self.group,
            theta_star=theta_star
        )

    def fit_star(self):

        home = self.group.home[:]
        away = self.group.away[:]
        X = sparse2onehot(home, away)
        y = self.group.y[:]

        lreg = skmodel.SGDClassifier(
            loss='log', penalty='none', fit_intercept=True,
            learning_rate='constant', eta0=1e-3,
            max_iter=10000, n_iter_no_change=10000,
            tol=1e-10, n_jobs=-1
        )
        lreg.fit(X, y)

        theta_star = lreg.coef_.ravel()
        v = theta_star.var(ddof=1)
        hfa = lreg.intercept_[0]

        self.group.attrs['v'] = v
        self.group.attrs['hfa'] = hfa

        fill_group(
            self.group,
            theta_star=theta_star
        )

    def fit(self, eta, theta0=None, use_hfa=True):
        theta_star = self.group.theta_star[:]
        v = theta_star.var(ddof=1)
        num_players = theta_star.size

        if theta0 is None:
            theta0 = np.zeros(self.group.team_names.size)

        num_games = self.group.y.size
        games = np.arange(num_games)
        hfa = self.hfa if use_hfa else 0

        # Simulate a season (with estimated hfa)
        exp_theta, exp_loss = sgd_shuffle(
            self.group.home[:],
            self.group.away[:],
            self.group.y[:],
            theta0=theta0,
            eta=eta,
            hfa=hfa,
            shuffle_vec=games
        )
        theta_star = self.group.theta_star[:]
        # theta_error = (exp_theta - theta_star[None, ])
        # exp_msd = theta_error.var(ddof=1, axis=-1) * (num_players - 1)

        # Make sure that the first element is theta0
        exp_theta = np.vstack((theta0, exp_theta[:-1,]))
        theta_error = (exp_theta - theta_star[None, ])
        exp_msd = theta_error.var(ddof=1, axis=-1) * num_players

        # Generate learning curve from stochastic model (with estimated hfa)
        model_theta = mutils.theta_expectation(theta_star, theta0, eta, num_games, hfa=hfa)
        model_msd = mutils.var_expectation(
            eta, v, num_players, num_games=games, hfa=hfa)
        Lmin, Lex = mutils.loss_expectations(
            eta, v, num_players, num_games=games, hfa=hfa)
        model_loss = Lmin + Lex

        tau1 = mutils.time_constant(
            eta, v, num_players, hfa, approx=True, first_order=True)
        tau2 = mutils.time_constant(
            eta, v, num_players, hfa, approx=True, first_order=False)

        exp = dict(
            theta=exp_theta,
            # msd=exp_msd / (num_players - 1),
            msd=exp_msd,
            loss=exp_loss,
        )
        model = dict(
            theta=model_theta[1:, ],
            # msd=model_msd / (num_players - 1),
            msd=model_msd,
            loss=model_loss,
            tau1=tau1,
            tau2=tau2,
        )

        return exp, model

######################################################### DEBUG

    def fit_shuffle(self, eta, theta0=None, use_hfa=True, num_shuffles=100):
        theta_star = self.group.theta_star[:]
        mu = theta_star.mean()
        v = theta_star.var(ddof=1)
        num_players = theta_star.size

        if eta is None:
            eta_max = mutils.stability_limit(v)
            eta = np.logspace(-5, 0.99*np.log10(eta_max), 100)

        if theta0 is None:
            theta0 = np.zeros(self.group.team_names.size)

        num_games = self.group.y.size
        # shuffle_vec = np.arange(num_games)
        hfa = self.hfa if use_hfa else 0

        # Simulate a season (with unkown hfa)
        exp_loss = np.zeros(num_games)
        exp_theta = np.zeros([num_games, num_players])
        for i in tqdm.trange(num_shuffles, leave=False):
            shuffle_vec = rnd.permutation(num_games)
            # shuffle_vec = rnd.randint(0, num_games, num_games)
            temp_theta, temp_loss = sgd_shuffle(
                self.group.home[:],
                self.group.away[:],
                self.group.y[:],
                theta0=theta0,
                eta=eta,
                hfa=0,
                shuffle_vec=shuffle_vec
            )
            exp_loss += temp_loss
            exp_theta += temp_theta
        exp_loss /= num_shuffles
        exp_theta /= num_shuffles

        # Generate learning curve from stochastic model (with estimated hfa)
        Lmin, Lex = mutils.loss_expectations(
            eta, v, num_players, num_games=np.arange(num_games), mu=mu, hfa=hfa)
        model_loss = Lmin + Lex

        model_theta = mutils.theta_expectation(
            theta_star, theta0, eta, num_games, hfa=hfa)

        return (exp_loss, exp_theta), (model_loss, model_theta[1:, ])

    def _loss_eta(self, eta=None, theta0=None, use_hfa=True):
        theta_star = self.group.theta_star[:]
        mu = theta_star.mean()
        v = theta_star.var(ddof=1)
        num_players = theta_star.size

        if eta is None:
            eta_max = mutils.stability_limit(v)
            eta = np.logspace(-5, 0.99*np.log10(eta_max), 100)

        if np.array(eta).ndim == 0:
            eta = np.array(eta)[None, ]

        if theta0 is None:
            theta0 = np.zeros(self.group.team_names.size)

        num_games = self.group.y.size
        shuffle_vec = np.arange(num_games)
        hfa = self.hfa if use_hfa else 0

        exp_loss = np.empty([eta.size, num_games])
        for i in tqdm.trange(eta.size):
            _, exp_loss[i, ] = sgd_shuffle(
                self.group.home[:],
                self.group.away[:],
                self.group.y[:],
                theta0=theta0,
                eta=eta[i],
                hfa=0,
                shuffle_vec=shuffle_vec
            )

        Lmin, Lex = mutils.loss_expectations(
            eta[:, None], v, num_players, num_games=shuffle_vec[None, :], mu=mu, hfa=hfa)
        model_loss = Lmin + Lex

        return eta, exp_loss, model_loss

    def loss_lr(self, lr=None, theta0=None, use_hfa=True):
        theta_star = self.group.theta_star[:]
        v = theta_star.var(ddof=1)
        num_players = theta_star.size

        if lr is None:
            lr_max = mutils.stability_limit(v)
            lr = np.logspace(-4, 0.99*np.log10(lr_max), 100)

        if np.array(lr).ndim == 0:
            lr = np.array(lr)[None, ]

        if theta0 is None:
            theta0 = np.zeros(self.group.team_names.size)

        num_games = self.group.y.size
        games = np.arange(num_games)
        hfa = self.hfa if use_hfa else 0

        exp_loss = np.empty([lr.size, num_games])
        exp_msd = np.empty([lr.size, num_games])
        for i in tqdm.trange(lr.size):
            theta_temp, exp_loss[i, ] = sgd_shuffle(
                self.group.home[:],
                self.group.away[:],
                self.group.y[:],
                theta0=theta0,
                eta=lr[i],
                hfa=hfa,
                shuffle_vec=games
            )
            theta_temp = np.vstack((theta0, theta_temp[:-1,]))
            theta_error = (theta_temp - theta_star[None, ])
            exp_msd[i, ] = theta_error.var(ddof=1, axis=-1) * num_players

        model_msd = mutils.var_expectation(
            lr[:, None], v, num_players, num_games=games[None, :], hfa=hfa)
        Lmin, Lex = mutils.loss_expectations(
            lr[:, None], v, num_players, num_games=games[None, :], hfa=hfa)
        model_loss = Lmin + Lex

        exp = dict(
            # msd=exp_msd / (num_players - 1),
            msd=exp_msd,
            loss=exp_loss,
        )
        model = dict(
            # msd=model_msd / (num_players - 1),
            msd=model_msd,
            loss=model_loss,
        )

        return lr, exp, model

    def test(self, **kwargs):
        return kwargs

    @property
    def R(self):
        if 'R' not in self.group.keys():
            R = corr_matrix(
                home=self.group.home[:],
                away=self.group.away[:],
                num_players=self.group.team_names.size
            )
            fill_group(self.group, R=R)
            return R
        else:
            return self.group.R[:]

    @property
    def hfa(self):
        return self.group.attrs['hfa']
        y = self.group.y[:]
        p_a = (y == 0).mean()
        p_h = (y == 1).mean()
        return np.log(p_h / p_a)


class SuperLegaSummary():
    """Some Information about SuperLegaSummary"""

    def __init__(self, zarr_path='superlega'):
        super(SuperLegaSummary, self).__init__()
        self.zarr_path = pathlib.Path(zarr_path)
        self.root_group = zarr.open_group(str(self.zarr_path), mode='a')
        self.num_seasons = len(self.root_group.items())
        self.season_list = [int(name.split('-')[-1])
                            for name, _ in self.root_group.items()]
        self.datasets = [SuperLegaRatings(self.zarr_path, season)
                         for season in self.season_list]
        self.df = self.summary()

    def summary(self):
        columns = ['season', 'players', 'games', 'hfa', 'mu', 'v', 'h', 'h2', 'lmin']
        df = pd.DataFrame(
            columns=columns,
            data=np.zeros((self.num_seasons, len(columns)))
        )
        for i, season in enumerate(self.season_list):
            dataset = self.datasets[i]
            group = dataset.group
            theta_star = group.theta_star[:]
            home = group.home[:]
            away = group.away[:]
            # hfa = dataset.hfa
            # v = theta_star.var(ddof=1)

            v = group.attrs['v']
            hfa = group.attrs['hfa']

            z = theta_star[home] - theta_star[away] + hfa
            mu = theta_star.mean()

            h_1 = utils.hessian(z).mean()
            h_2 = (utils.hessian(z) ** 2).mean()

            h, a = utils.probs(z)
            lmin = -(h * np.log(h) + a * np.log(a)).mean()

            df['season'][i] = season
            df['players'][i] = theta_star.size
            df['games'][i] = home.size
            df['hfa'][i] = hfa
            df['h'][i] = h_1
            df['h2'][i] = h_2
            df['lmin'][i] = lmin
            df['mu'][i] = mu
            df['v'][i] = v

        df.season = df.season.astype('int64')
        df.players = df.players.astype('int64')
        df.games = df.games.astype('int64')

        return df

    def get_full_data(self):
        home = np.zeros(0, dtype='int64')
        away = np.zeros(0, dtype='int64')
        y = np.zeros(0, dtype='int64')
        players = 0
        for i, season in enumerate(self.season_list):
            dataset = self.datasets[i]
            group = dataset.group
            players += group.team_names.size

            home = np.append(home, group.home[:] + players)
            away = np.append(away, group.away[:] + players)
            y = np.append(y, group.y[:])

        X = sparse2onehot(home, away)

        return X, y


    def grid_search(self, season='all'):
        if season == 'all':
            X, y = self.get_full_data()
        else:
            dataset = self.datasets[self.season_list.index(season)]
            group = dataset.group
            home = group.home[:]
            away = group.away[:]
            X = sparse2onehot(home, away)
            y = group.y[:]
        # X = X.astype('float64')
        # y = y.astype('float64')

        lreg = LogisticReg(fit_intercept=False, penalty='l2')
        # lreg = skmodel.LogisticRegression(penalty='l2', fit_intercept=True)

        parameters = {
            # 'C': np.logspace(-2, 2, 10),
            # 'C': np.logspace(np.log10(4.97702356), np.log10(5.72236766), 10),
            'C': np.logspace(np.log10(0.5), np.log10(5), 10),
            'intercept': np.linspace(0, 2, 10)
        }
        # {'C': 5.295484774737589, 'intercept': 0.5656565656565657}

        cv_splitter = skselect.LeaveOneOut()
        # cv_splitter = skselect.KFold(n_splits=100)

        LogLoss = skmetrics.make_scorer(
            skmetrics.log_loss,
            greater_is_better=False,
            needs_proba=True,
            eps=1e-7, labels=[0, 1]
        )
        grid = skselect.GridSearchCV(
            lreg, parameters, cv=cv_splitter,
            scoring=LogLoss, n_jobs=-1, error_score='raise'
        )
        grid.fit(X, y)

        theta = grid.best_estimator_.coef_.ravel()
        # theta_star = group.theta_star[:]

        # print(grid.best_params_)
        # params = pd.DataFrame(grid.cv_results_['params'])
        # score = grid.cv_results_['mean_test_score']
        # ind = np.where(params.intercept.values == grid.best_params_['intercept'])[0]
        # import matplotlib.pyplot as plt
        # plt.plot(params.C[ind], score[ind], 'o-')
        # plt.xscale('log')

        return grid.best_params_, theta


    def _learning_rate(self, steady_games=100, num_eta=100, eta_min=1e-4, use_hfa=True):
        eta_max = mutils.stability_limit(1e6)
        eta = np.logspace(np.log10(eta_min), 0.99*np.log10(eta_max), num_eta)

        exp_loss = np.empty([self.num_seasons, eta.size])
        model_loss = np.empty([self.num_seasons, eta.size])
        for i, season in enumerate(self.season_list):
            dataset = SuperLegaRatings(season=season)
            theta0 = np.zeros(dataset.group.team_names.size)
            _, exp_temp, model_temp = dataset.loss_eta(
                eta=eta, theta0=theta0, use_hfa=use_hfa)
            exp_loss[i, ] = exp_temp[:, steady_games:].mean(axis=-1)
            model_loss[i, ] = model_temp[:, steady_games:].mean(axis=-1)

        return eta, exp_loss, model_loss

    def learning_rate(self, steady_games=None, num_lr=100, lr_min=1e-3, lr_max=None, use_hfa=True):
        if lr_max is None:
            lr_max = mutils.stability_limit(1e6)
        lr = np.logspace(np.log10(lr_min), 0.99*np.log10(lr_max), num_lr)

        if steady_games is None:
            steady_games = (100, None)

        shape = (self.num_seasons, lr.size)
        exp = dict(
            msd=np.empty(shape),
            loss=np.empty(shape),
        )
        model = dict(
            msd=np.empty(shape),
            loss=np.empty(shape),
        )
        for i, season in enumerate(self.season_list):
            dataset = SuperLegaRatings(season=season)
            num_games = self.df.games[i]
            games = np.arange(num_games)[steady_games[0]:steady_games[1]]

            theta0 = np.zeros(dataset.group.team_names.size)
            _, exp_temp, model_temp = dataset.loss_lr(lr=lr, theta0=theta0, use_hfa=use_hfa)
            exp['msd'][i, ] = exp_temp['msd'][:, games].mean(axis=-1)
            exp['loss'][i, ] = exp_temp['loss'][:, games].mean(axis=-1)
            model['msd'][i, ] = model_temp['msd'][:, games].mean(axis=-1)
            model['loss'][i, ] = model_temp['loss'][:, games].mean(axis=-1)

        return lr, exp, model

    def learning_full(self, num_eta=100, eta_min=1e-4, use_hfa=True):
        eta_max = mutils.stability_limit(1e6)
        eta_vec = np.logspace(np.log10(eta_min), 0.99 *
                              np.log10(eta_max), num_eta)

        max_games = self.df.games.max()
        exp_loss = np.full([num_eta, self.num_seasons, max_games], np.nan)
        model_loss = np.full([num_eta, self.num_seasons, max_games], np.nan)
        for i, eta in enumerate(tqdm.tqdm(eta_vec)):
            result = self.learning_curve(eta, use_hfa=use_hfa)
            (exp_loss[i, ], _), (model_loss[i, ], _) = result

        return eta_vec, exp_loss, model_loss

    def learning_shuffle(self, num_eta=100, eta_min=1e-4, use_hfa=True, num_shuffles=100):
        eta_max = mutils.stability_limit(1e6)
        eta_vec = np.logspace(np.log10(eta_min), 0.99 *
                              np.log10(eta_max), num_eta)

        max_games = self.df.games.max()
        exp_loss = np.full([num_eta, self.num_seasons, max_games], np.nan)
        model_loss = np.full([num_eta, self.num_seasons, max_games], np.nan)
        for i, eta in enumerate(tqdm.tqdm(eta_vec)):
            result = self.learning_curve_shuffle(eta, use_hfa=use_hfa, num_shuffles=num_shuffles)
            (exp_loss[i, ], _), (model_loss[i, ], _) = result

        return eta_vec, exp_loss, model_loss

    def _learning_curve(self, eta, use_hfa=True):
        max_players = self.df.players.max()
        max_games = self.df.games.max()

        exp_loss = np.full([self.num_seasons, max_games], np.nan)
        exp_theta = np.full([self.num_seasons, max_games, max_players], np.nan)
        model_loss = np.full([self.num_seasons, max_games], np.nan)
        model_theta = np.full(
            [self.num_seasons, max_games, max_players], np.nan)
        for i, season in enumerate(self.season_list):
            dataset = self.datasets[i]
            theta0 = np.zeros(dataset.group.team_names.size)
            _result = dataset.fit(eta, theta0=theta0, use_hfa=use_hfa)
            (_exp_loss, _exp_theta), (_model_loss, _model_theta) = _result

            num_games = dataset.group.y.size
            num_players = dataset.group.team_names.size

            exp_loss[i, :num_games] = _exp_loss
            exp_theta[i, :num_games, :num_players] = _exp_theta
            model_loss[i, :num_games] = _model_loss
            model_theta[i, :num_games, :num_players] = _model_theta

        return (exp_loss, exp_theta), (model_loss, model_theta)

    def learning_curve(self, lr, use_hfa=True):
        max_players = self.df.players.max()
        max_games = self.df.games.max()
        shape = (self.num_seasons, max_games, max_players)

        if lr == 'best':
            hfa = self.df.hfa.values
            v = self.df.v.values
            # lr = mutils.stability_limit2(v, hfa) / 2

            players = self.df.players.values
            games = self.df.games.values
            lr = mutils.optimal_beta_k(v, players, games // 4, hfa)
            # lr = mutils.optimal_beta_k(v, players, games // 2, hfa)
            # lr = mutils.optimal_beta_k(v, players, games, hfa)
        else:
            lr = np.array(lr)
            if len(lr.shape) == 0:
                lr = np.repeat(lr, self.num_seasons)
        assert lr.size == self.num_seasons

        exp = dict(
            theta=np.full(shape, np.nan),
            msd=np.full(shape[:-1], np.nan),
            loss=np.full(shape[:-1], np.nan),
        )
        model = dict(
            theta=np.full(shape, np.nan),
            msd=np.full(shape[:-1], np.nan),
            loss=np.full(shape[:-1], np.nan),
            tau1=np.empty(shape[0]),
            tau2=np.empty(shape[0]),
        )
        for i in range(self.num_seasons):
            dataset = self.datasets[i]
            _exp, _model = dataset.fit(lr[i], use_hfa=use_hfa)

            num_games = dataset.group.y.size
            num_players = dataset.group.team_names.size

            exp['theta'][i, :num_games, :num_players] = _exp['theta']
            exp['loss'][i, :num_games] = _exp['loss']
            exp['msd'][i, :num_games] = _exp['msd']

            model['theta'][i, :num_games, :num_players] = _model['theta']
            model['loss'][i, :num_games] = _model['loss']
            model['msd'][i, :num_games] = _model['msd']
            model['tau1'][i] = _model['tau1']
            model['tau2'][i] = _model['tau2']

        return exp, model

    def learning_curve_shuffle(self, eta, use_hfa=True, num_shuffles=100):
        max_players = self.df.players.max()
        max_games = self.df.games.max()

        exp_loss = np.full([self.num_seasons, max_games], np.nan)
        exp_theta = np.full([self.num_seasons, max_games, max_players], np.nan)
        model_loss = np.full([self.num_seasons, max_games], np.nan)
        model_theta = np.full(
            [self.num_seasons, max_games, max_players], np.nan)
        for i, season in enumerate(tqdm.tqdm(self.season_list, leave=False)):
            dataset = self.datasets[i]
            theta0 = np.zeros(dataset.group.team_names.size)
            _result = dataset.fit_shuffle(
                eta, theta0=theta0, use_hfa=use_hfa, num_shuffles=num_shuffles)
            (_exp_loss, _exp_theta), (_model_loss, _model_theta) = _result

            num_games = dataset.group.y.size
            num_players = dataset.group.team_names.size

            exp_loss[i, :num_games] = _exp_loss
            exp_theta[i, :num_games, :num_players] = _exp_theta
            model_loss[i, :num_games] = _model_loss
            model_theta[i, :num_games, :num_players] = _model_theta

        return (exp_loss, exp_theta), (model_loss, model_theta)


@numba.njit()
def log_loss(y, y_hat, eps=1e-15):
    if y_hat > 1 - eps:
        y_hat = 1 - eps
    elif y_hat < eps:
        y_hat = eps
    loss = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    return -loss


@numba.njit(cache=True)
def sgd_epochs(home, away, y, theta0, eta, hfa, num_epochs=1, log_step=10, shuffle=False):
    theta = theta0.copy()
    num_players = theta.size
    num_games = y.size
    loss = np.zeros((num_epochs, num_games))
    num_logs = num_games // log_step + 1
    theta_hat = np.zeros((num_epochs, num_logs, num_players))
    for k in range(num_epochs):
        if shuffle:
            games_index = rnd.permutation(num_games)
        else:
            games_index = np.arange(num_games)
        for i, g in enumerate(games_index):
            # # Input vector
            # x = np.zeros(num_players)
            # x[home[g]] = 1.0
            # x[away[g]] = -1.0
            h = home[g]
            a = away[g]

            # Predicted output
            z_hat = theta[h] - theta[a] + hfa
            y_hat = utils.sigmoid(z_hat)

            # Learning step
            delta = eta * (y[g] - y_hat)

            # Parameter update
            # theta = theta + eta * e * x
            theta[h] = theta[h] + delta
            theta[a] = theta[a] - delta

            # Logging
            loss[k, i] = utils.log_loss(y[g], z_hat)
            if i % log_step == 0:
                theta_hat[k, i // log_step, ] = theta

    return theta_hat, loss


@numba.njit(cache=True)
def sgd_shuffle(home, away, y, theta0, eta, hfa, shuffle_vec):
    theta = theta0.copy()
    num_players = theta.size
    num_games = y.size
    loss = np.zeros(num_games)
    theta_hat = np.zeros((num_games, num_players))
    for i, g in enumerate(shuffle_vec):
        # Input vector indexes
        # x = np.zeros(num_players)
        # x[home[g]] = 1.0
        # x[away[g]] = -1.0
        h = home[g]
        a = away[g]

        # Predicted output
        # z_hat = x @ theta + hfa
        z_hat = theta[h] - theta[a] + hfa
        y_hat = utils.sigmoid(z_hat)

        # Learning step
        # dL = x * (y_hat - y[g])
        delta = eta * (y[g] - y_hat)

        # Parameter update
        # theta = theta - eta * dL
        theta[h] = theta[h] + delta
        theta[a] = theta[a] - delta

        # Logging
        loss[i] = utils.log_loss(y[g], z_hat)
        theta_hat[i, ] = theta

    return theta_hat, loss


@numba.njit(cache=True)
def batch_descent(home, away, y, theta0, eta, hfa, num_iters):
    theta = theta0.copy()
    num_players = theta.size
    num_games = y.size
    loss = np.zeros(num_iters)
    theta_hat = np.zeros((num_iters, num_players))
    x = np.zeros((num_games, num_players))
    for i in range(num_games):
        x[i, home[i]] = 1.0
        x[i, away[i]] = -1.0

    for i in range(num_iters):
        # Predicted output
        z_hat = theta[home] - theta[away] + hfa
        y_hat = utils.sigmoid(z_hat)

        # Learning step
        g = y_hat - y
        dL = x.T @ g

        # Parameter update
        # for k in range(num_games):
        #     theta[home[k]] = theta[home[k]] + delta[k]
        #     theta[away[k]] = theta[away[k]] - delta[k]
        theta = theta - eta * dL

        # Logging
        loss[i] = utils.log_loss(y, z_hat).mean()
        theta_hat[i, ] = theta

    return theta_hat, loss


@numba.njit()
def sgd_logistic(num_games, num_players, x, y, theta0, eta):
    theta = np.empty((num_games, num_players))
    theta[0, ] = theta0.copy()
    for k in range(num_games - 1):
        g = utils.sigmoid(x[k, ] @ theta[k, ]) - y[k]
        dL = g * x[k, ]
        theta[k + 1, ] = theta[k, ] - eta * dL

    return theta


def main():
    import matplotlib.pyplot as plt

    # Convert from CSV to Zarr
    # root_folder = '~/data/Datasets/SuperLega'
    root_folder = 'dataset_csv'
    for i in tqdm.trange(2009,2019):
        convert_dataset(season=i, root_folder=root_folder)

        dataset = SuperLegaRatings(season=i)
        dataset.fit_star()

    # eta = 1e-3
    # R = dataset.R
    # hfa = dataset.hfa

    # group = dataset.group
    # theta_star = group.theta_star[:]
    # home = group.home[:]
    # away = group.away[:]
    # y = group.y[:]
    # loss = group.loss[:]

    # fig, ax = plt.subplots(figsize=(10,7))
    # ax.hist(theta_star, bins=100)

    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.plot(loss.mean(axis=-1))

    # plt.show()
    print()


if __name__ == '__main__':
    main()
    # summary = SuperLegaSummary()
    # summary.grid_search(season='all')

    # for i in range(2009, 2020):
    #     params, theta = summary.grid_search(season=i)



    print()
