import seaborn as sb
from matplotlib import rcParams
import matplotlib


def set_style():
    sb.set(context="paper", style="ticks", font_scale=2, font="Arial", rc={
        "lines.linewidth": 2.5
    })

    matplotlib.rc('axes', edgecolor='black')

    rcParams['axes.linewidth'] = 3.0

    sb.set_palette("dark")

    matplotlib.rcParams['xtick.major.size'] = 20
    matplotlib.rcParams['xtick.major.width'] = 4

    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 2

    matplotlib.rcParams['ytick.major.size'] = 20
    matplotlib.rcParams['ytick.major.width'] = 4

    matplotlib.rcParams['ytick.minor.size'] = 10
    matplotlib.rcParams['ytick.minor.width'] = 2