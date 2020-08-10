import os
from typing import Tuple

import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.plotting.help_func import MidpointNormalize, OOMFormatter, get_scientific_order


def plotly_mnist_image(image):
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.Cividis, zmin=-0.15, zmax=0.15)
    fig.show()


def plot_rel_prop(image: np.ndarray, correct_label: str, relevances: Tuple, persist_string: str, show: bool = False):
    num_pics = len(relevances)
    n_col = int((num_pics + 1)/2 + 0.5) +1

    # plt.suptitle(f'Erklärung für die Klassifizierung: {correct_label}')

    plt.subplot(1, num_pics+1, 1)
    plt.title('Input')
    fig = plt.imshow(np.array(image, dtype=np.dtype(int)))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    # vals = np.array([val[1] for val in relevances])
    # rel_max = vals.max()
    # rel_min = vals.min()

    for i in range(0, num_pics):
        plt.subplot(1, num_pics+1, i+2)
        plt.title(relevances[i][0], fontsize=12)
        relevance = relevances[i][1][0]
        ax = plt.gca()
        fig = ax.imshow(relevance, cmap='seismic',
                         norm=MidpointNormalize(midpoint=0, vmin=relevance.min(), vmax=relevance.max()))
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        order = get_scientific_order(vmin=relevance.min(), vmax=relevance.max())

        # plt.colorbar(fig, cax=cax, format=OOMFormatter(order, mathText=False))
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        # if i+2 > n_col:
        #     plt.xlabel(relevances[i][0], fontsize=12)
        # else:
            # plt.title(relevances[i][0], fontsize=12)

        fig.axes.get_yaxis().set_visible(False)
        # fig.axes.get_xaxis().set_visible(False)

    if not show:
        fig_path = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', persist_string)

        plt.savefig(fig_path)
        plt.cla()
        plt.clf()
    else:
        plt.subplots_adjust(left=0.125, wspace=0.25)
        plt.show()


def plot_R_evo(evolutions_of_R: tuple, persist_string: str, show: bool, y_min: int = 0, y_max: int = 1.1):
    x = np.arange(len(evolutions_of_R[0][1]), 0, -1)

    num_pics = len(evolutions_of_R)

    plt.suptitle(f'Relative Entwicklung der Summe über alle Relevanzwerte')
    plt.gca().invert_xaxis()
    for i in range(0, num_pics):
        # plt.subplot(2, n_col, i+1)
        sum_over_R = evolutions_of_R[i][1]
        if i < num_pics-1:
            label = evolutions_of_R[i][0] + '\n'
        else:
            label = evolutions_of_R[i][0]

        fig = plt.plot(x, sum_over_R, label=label)
        plt.ylim(y_min, y_max)
        plt.xlabel('Nummer des Layers\nKlassifizierung -> Pixelinput')
        plt.ylabel(r'$\frac{Netzwerkoutput}{\sum_{Layer} Relevanzwert}$')

    if not show:
        fig_path = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'plots', persist_string + '_R_plots')
        plt.legend()
        plt.savefig(fig_path)
        plt.cla()
        plt.clf()
    else:
        plt.legend()
        plt.show()

