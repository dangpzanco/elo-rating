### ps backend params
#ps.papersize      : letter   ## auto, letter, legal, ledger, A0-A10, B0-B10
#ps.useafm         : False    ## use of afm fonts, results in small files
#ps.usedistiller   : False    ## can be: None, ghostscript or xpdf
## Experimental: may produce smaller files.
## xpdf intended for production of publication quality files,
## but requires ghostscript, xpdf and ps2eps
#ps.distiller.res  : 6000      ## dpi
#ps.fonttype       : 3         ## Output Type 3 (Type3) or Type 42 (TrueType)
## See https://matplotlib.org/users/customizing.html#the-matplotlibrc-file
## for more details on the paths which are checked for the configuration file.


import pathlib
from matplotlib.figure import Figure
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfont
import matplotlib.colors as mcolors

mpl.rcParams['ps.papersize'] = 'auto'
mpl.rcParams['ps.useafm'] = False
mpl.rcParams['ps.usedistiller'] = 'ghostscript'
mpl.rcParams['ps.distiller.res'] = 6000
mpl.rcParams['ps.fonttype'] = 42

# Easy 10-color list
color_list = list(mcolors.TABLEAU_COLORS)

# Requires a distribution of LaTeX installed (tested with MiKTeX on Windows 10)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['pgf.texsystem'] = "xelatex"
preamble = "\n".join(
    [r"\usepackage{amsmath}"])
mpl.rcParams['text.latex.preamble'] = preamble
mpl.rcParams['pgf.preamble'] = preamble


# figsize defaults:
# 9cm x 4.5cm (paper)
# 16cm x 10cm (screen)
def format_figure(
        fig: Figure, figsize='paper', times='Times New Roman', arial='Arial',
        tight_scale='x', custom=None
):

    # Assumes a single axis in figure (no support for plt.subplots)
    ax = fig.axes[0]

    # Default fonts
    times_path = mfont.findfont(times)
    arial_path = mfont.findfont(arial)

    arial_dict = dict(
        fname=arial_path,
        family='sans-serif'
    )

    times_dict = dict(
        fname=times_path,
        family='serif'
    )

    # Define font properties
    ticks_font = mfont.FontProperties(size=8, **arial_dict)
    labels_font = mfont.FontProperties(size=10, **times_dict)
    legend_font = mfont.FontProperties(size=8, **times_dict)

    # Set figure sizes
    if figsize == 'paper':
        figsize = np.array([9, 4.5])
    elif figsize == 'screen':
        figsize = np.array([16, 10])
        ticks_font.set_size(10)
        labels_font.set_size(12)
        legend_font.set_size(10)

    # Convert from cm to inch
    figsize = np.array(figsize) / 2.54

    # Adjust figure size
    fig.set_size_inches(figsize)
    # fig.set_dpi(200)

    # Tight axis scaling
    if tight_scale in ['x', 'y', 'both']:
        ax.autoscale(enable=True, axis=tight_scale, tight=True)

    # Adjust the font of x and y labels
    ax.set_xlabel(ax.get_xlabel(), font=labels_font)
    ax.set_ylabel(ax.get_ylabel(), font=labels_font)

    # Legend font config
    leg = ax.get_legend()
    if leg is not None:
        ax.legend(
            prop=legend_font,
            loc=leg._get_loc(),
            borderaxespad=leg.borderaxespad
        )

    # Set the font settings for axis tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(ticks_font)

    for tick in ax.get_yticklabels():
        tick.set_fontproperties(ticks_font)

    xscale = ax.get_xscale()
    yscale = ax.get_yscale()

    if xscale == 'log':
        for tick in ax.get_xminorticklabels():
            tick.set_fontproperties(ticks_font)

    if yscale == 'log':
        for tick in ax.get_yminorticklabels():
            tick.set_fontproperties(ticks_font)

    # Add scientific notation ticks for axis in the (10 ** scilimits) interval
    ticklabel_options = dict(
        style='scientific', scilimits=(-2, 3),
        useMathText=True, useOffset=True
    )

    if xscale == 'linear':
        ax.ticklabel_format(axis='x', **ticklabel_options)

    if yscale == 'linear':
        ax.ticklabel_format(axis='y', **ticklabel_options)

    # Custom code you edit in here
    if custom:
        pass

    # Set the font settings for axis offset text (a.k.a. scientific notation)
    ax.xaxis.offsetText.set_font(ticks_font)
    ax.yaxis.offsetText.set_font(ticks_font)

    # Really tight layout (default pad is ~1.0)
    fig.tight_layout(pad=0.1, h_pad=None, w_pad=None, rect=None)

    return fig


def save_fig(fig, name, path=None, format=None, dpi=600, close=False, usetex=True, **kwargs):
    # Make sure to save on a folder that exists
    if path is None:
        path = 'figures'
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = ['pdf', 'eps', 'pgf', 'png']

    fig = format_figure(fig, **kwargs)
    backend = 'pgf' if usetex else None
    if 'pdf' in format:
        # fig.savefig(path / f'{name}.pdf', format='pdf', backend=backend)
        fig.savefig(path / f'{name}.pdf', format='pdf', transparent=True)
    if 'eps' in format: # no support for pgf backend in eps
        fig.savefig(path / f'{name}.eps', format='eps', transparent=True)
    if usetex and ('pgf' in format):
        fig.savefig(path / f'{name}.pgf', format='pgf', transparent=True)
    if 'png' in format:
        fig.savefig(path / f'{name}.png', format='png', transparent=True, dpi=dpi, backend=backend)

    if close:
        plt.close(fig)

    return fig
