import numpy as np
import matplotlib.pyplot as plt
# from adjustText import adjust_text

import data_utils as dutils
import model_utils as mutils
import plot_utils as lutils

import matplotlib.pyplot as plt
from scipy import signal


def mean_filter(x, p=10):
    b = np.ones(p) / p
    y = signal.lfilter(b, 1, x)
    y[:p, ] *= p / (1 + np.arange(p))
    return y


def get_optimal_beta_k(steady_games: tuple):
    summary = dutils.SuperLegaSummary()
    df = summary.df

    opt_beta = np.empty(summary.num_seasons)
    for i in range(opt_beta.size):
        games = np.arange(df.games.values[i])[steady_games[0]:steady_games[1]]
        opt_beta[i] = mutils.optimal_beta_k(
            v=df.v.values[i],
            num_players=df.players.values[i],
            num_games=games,
            hfa=df.hfa.values[i]
        ).mean()

    return opt_beta


def plot_season_teams(
    season_index=None, use_hfa=True, num_teams=4, num_tau=3,
    lang='en', path='figures/experiments/seasons', close=False
):

    figname = 'skills-season'
    if 'pt' in lang:
        lang_str = 'ptbr_'
        xylabel = dict(x='Jogos $k$', y='Habilidades')
    else:
        lang_str = ''
        xylabel = dict(x='Games $k$', y='Skills')

    summary = dutils.SuperLegaSummary()
    # min_games = np.min([item.group.y.size for item in summary.datasets])

    exp, model = summary.learning_curve(lr='best', use_hfa=use_hfa)

    if season_index is None:
        season_index = range(summary.num_seasons)

    fig_list = []
    for i, season in enumerate(season_index):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Sample teams uniformly
        M = summary.df.players[season]
        K = summary.df.games[season]
        sample_players = np.linspace(0, M-1, num_teams).astype(int)
        sorted_ind = np.argsort(summary.datasets[season].group.theta_star[:])
        sample_players = sorted_ind[sample_players]

        # Print team names
        team_names = summary.datasets[season].group.team_names[:][sample_players]
        print(season, team_names)

        # Select lines to plot
        exp_plot = exp['theta'][season, :, sample_players]
        model_plot = model['theta'][season, :, sample_players]
        # tau_plot = round(num_tau*model['tau1'][season])
        # print(summary.season_list[season], f'{num_tau}*tau = {tau_plot}', team_names)

        for k in range(sample_players.size):
            ax.plot(np.hstack([0, exp_plot[k, ]]),
                    color=lutils.color_list[k], ls='-')
            ax.plot(np.hstack([0, model_plot[k, ]]),
                    color=lutils.color_list[k], ls='--')
        # ax.axvline(tau_plot, color='k', ls='--')
        ax.set_xlabel(xylabel['x'])
        ax.set_ylabel(xylabel['y'])
        ax.grid(True)

        fig.tight_layout()

        # Add extra ticks
        ax.autoscale(tight=True)
        xticks = ax.get_xticks()
        extra_ticks = np.array([K])
        new_ticks = []
        for tick in xticks:
            if (np.abs(extra_ticks - tick) > 10).all():
                new_ticks.append(tick)
        new_ticks.extend(extra_ticks)
        ax.set_xticks(new_ticks)

        # texts = []
        # for k in range(sample_players.size):
        #     text_x = (k+1) * tau_plot // (sample_players.size - 1)
        #     text_y = model_plot[k, text_x-1]
        #     texts.append(ax.text(text_x, text_y, team_names[k]))

        # adjust_text(texts)

        plot_name = f"{lang_str}{figname}{df.season[season]}"
        lutils.save_fig(fig, name=plot_name, path=path,
                        format='pdf', close=close)

        fig_list.append(fig)

    print(summary.df.iloc[season_index].to_string())

    return fig_list


def plot_season_metric(
    name='loss', season_index=None, use_hfa=True, num_tau=3,
    lang='en', path='figures/experiments/seasons/msd', close=False
):

    figname = 'msd-season'
    if 'pt' in lang:
        ylabel_dict = dict(
            msd='MSD',
            loss='Custo',
        )
        lang_str = 'ptbr_'
        xylabel = dict(x='Jogos $k$', y=ylabel_dict[name])
    else:
        ylabel_dict = dict(
            msd='MSD',
            loss='Loss',
        )
        lang_str = ''
        xylabel = dict(x='Games $k$', y=ylabel_dict[name])

    summary = dutils.SuperLegaSummary()
    exp, model = summary.learning_curve(lr='best', use_hfa=use_hfa)

    if season_index is None:
        season_index = range(summary.num_seasons)

    fig_list = []
    for i, season in enumerate(season_index):

        K = summary.df.games[season]
        # tau_plot = round(num_tau*model['tau2'][season])
        exp_plot = exp[name][season, ]
        model_plot = model[name][season, ]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(exp_plot, color='gray', ls='-')
        ax.plot(model_plot, color='k', ls='--')
        # ax.axvline(tau_plot, color='k', ls='--')
        ax.set_xlabel(xylabel['x'])
        ax.set_ylabel(xylabel['y'])
        ax.grid(True)

        fig.tight_layout()

        # Add extra ticks
        ax.autoscale(tight=True, axis='x')
        if False:
            xticks = ax.get_xticks()
            extra_ticks = np.array([K])
            new_ticks = []
            for tick in xticks:
                if (np.abs(extra_ticks - tick) > 10).all():
                    new_ticks.append(tick)
            new_ticks.extend(extra_ticks)
            ax.set_xticks(new_ticks)

        plot_name = f'{lang_str}{figname}{df.season[season]}'
        lutils.save_fig(fig, name=plot_name, path=path,
                        format='pdf', close=close)

        fig_list.append(fig)

    # print(summary.df.iloc[season_index].to_string())

    return fig_list


def plot_metrics(
    lr, name='loss', use_hfa=True,
    lang='en', path='figures/experiments/metrics', close=False
):

    figname = f'{name}_beta'
    if 'pt' in lang:
        ylabel_dict = dict(
            msd='MSD',
            loss='Custo',
        )
        lang_str = 'ptbr_'
        xylabel = dict(x='Jogos $k$', y=ylabel_dict[name])
    else:
        ylabel_dict = dict(
            msd='MSD',
            loss='Loss',
        )
        lang_str = ''
        xylabel = dict(x='Games $k$', y=ylabel_dict[name])

    lr = np.array(lr)
    if len(lr.shape) == 0:
        lr = lr[None, ]

    summary = dutils.SuperLegaSummary()
    min_games = np.min([item.group.y.size for item in summary.datasets])

    fig_list = []
    for i in range(lr.size):
        exp, model = summary.learning_curve(lr=lr[i], use_hfa=use_hfa)

        exp_plot = np.nanmean(exp[name], axis=0)[:min_games]
        model_plot = np.nanmean(model[name], axis=0)[:min_games]

        if name == 'loss':
            if exp_plot.max() > 2:
                axis_limits = [None, None, 0, 2.3]
            else:
                axis_limits = [None, None, 0.16, 1.35]
        elif name == 'msd':
            axis_limits = [None, None, 0, 40]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(exp_plot, color='gray', ls='-')
        ax.plot(model_plot, color='k', ls='--')
        ax.grid(True)

        # if name == 'loss':
        #     filter_data = mean_filter(exp_plot, p=10)
        #     ax.plot(filter_data, color='tab:blue', ls='-')

        ax.set_xlabel(xylabel['x'])
        ax.set_ylabel(xylabel['y'])
        ax.axis(axis_limits)
        fig.tight_layout()

        plot_name = f'{lang_str}{figname}{i}'
        lutils.save_fig(fig, name=plot_name, path=path,
                        format='pdf', close=close)

        fig_list.append(fig)

    return fig_list


def plot_steady(
    lr_list, steady_games=None, use_hfa=True, num_lr=100,
    lang='en', path='figures/experiments/steady', close=False
):

    figname = 'steady_state'
    if 'pt' in lang:
        lang_str = 'ptbr_'
        xlabel = 'Passo de adaptação $\\beta$'
        ylabel = dict(
            msd='MSD',
            loss='Custo',
        )
    else:
        lang_str = ''
        xlabel = 'Step size $\\beta$'
        ylabel = dict(
            msd='MSD',
            loss='Loss',
        )

    lr_list = np.array(lr_list)
    if len(lr_list.shape) == 0:
        lr_list = lr_list[None, ]

    summary = dutils.SuperLegaSummary()

    if steady_games is None:
        steady_games = (100, None)

    lr, exp, model = summary.learning_rate(
        steady_games=steady_games, use_hfa=use_hfa, num_lr=num_lr, lr_min=1e-2, lr_max=4)

    fig_dict = {}
    for i, name in enumerate(exp.keys()):
        if name == 'loss':
            axis_limits = [None, None, 0.4, 1.2]
        elif name == 'msd':
            axis_limits = [None, None, 2, 100]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(lr, exp[name].mean(axis=0), color='gray')
        ax.plot(lr, model[name].mean(axis=0), ls='--', color='k')
        ax.grid(True)

        for j in range(lr_list.size):
            # ax.axvline(lr_list[j], color=lutils.color_list[j], ls='--')
            ax.axvline(lr_list[j], color='k', ls=lutils.linestyle_list[j])

        if name == 'msd':
            ax.set_yscale('log')

        ax.set_xscale('log')
        ax.axis(axis_limits)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel[name])
        fig.tight_layout()

        plot_name = f'{lang_str}{figname}-{name}'
        lutils.save_fig(fig, name=plot_name, path=path,
                        format='pdf', close=close)

        fig_dict[name] = fig

    return fig_dict


summary = dutils.SuperLegaSummary()
df = summary.summary()
print(df.to_string())
beta_limit = mutils.stability_limit(df.v, hfa=df.hfa).values
beta_improv = mutils.improvement_condition(df.v, df.players, hfa=df.hfa).values
beta_o1 = beta_improv / 2
beta_ok = mutils.optimal_beta_k(
    df.v, df.players, df.games // 4, hfa=df.hfa).values
print('Optimal β (k = K//4):', beta_ok)

# summary.datasets[0].fit_star()

use_hfa = True

################################################################################################

# plot_seasons = np.array([1, 2, 7, 10]) - 1
plot_seasons = np.array([1, 9, 7, 10]) - 1

print('Considered seasons:', df.index[plot_seasons])
print('Optimal β (k = K//4) [per season]:', beta_ok[plot_seasons])

plot_season_teams(season_index=plot_seasons, lang='en')
plot_season_metric(name='msd', season_index=plot_seasons, lang='en')

plot_season_teams(season_index=plot_seasons, lang='pt')
plot_season_metric(name='msd', season_index=plot_seasons, lang='pt')


################################################################################################

# FIFA
K_elo = 15
s_elo = 600
beta_elo = K_elo / (s_elo / np.log(10))


v = df.v.values
beta0 = 2 * (beta_o1 * v).sum() / v.sum()
beta_ok = mutils.optimal_beta_k(
    df.v, df.players, df.games // 4, hfa=df.hfa).values

beta_elo = 0.1
beta = [beta_elo, beta_ok.mean(), beta_improv.mean()]

print('β values:', beta)

plot_metrics(beta, name='loss', lang='en')
plot_metrics(beta, name='msd', lang='en')

plot_metrics(beta, name='loss', lang='pt')
plot_metrics(beta, name='msd', lang='pt')


################################################################################################

# # FIDE
# K_elo = 40
# s_elo = 400

# FIFA
K_elo = 15
s_elo = 600

beta_elo = K_elo / (s_elo / np.log(10))

min_games = np.min([item.group.y.size for item in summary.datasets])
# vertical lines (color order): blue, orange, green, red, purple

K = 12*11
steady_games = (K-10, K)
opt_beta = get_optimal_beta_k(steady_games)
opt_beta = mutils.optimal_beta_k(df.v, df.players, df.games, hfa=df.hfa).values

print(opt_beta.mean(), opt_beta)
betas = [beta_improv.mean(), opt_beta.mean()]

plot_steady(betas, steady_games, lang='en')
plot_steady(betas, steady_games, lang='pt')
