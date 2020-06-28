import os
from typing import Any, Tuple

import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

from src.rel_prop.help_func import MidpointNormalize


def plotly_mnist_image(image):
    fig = px.imshow(image, color_continuous_scale=px.colors.sequential.Cividis, zmin=-0.15, zmax=0.15)
    fig.show()


def plot_rel_prop(image: np.ndarray, relevances: Tuple, persist_string: str, show: bool = False):
    num_pics = len(relevances)
    n_col = int((num_pics + 1)/2 + 0.5)

    plt.subplot(2, n_col, 1)
    plt.title('Input')
    fig = plt.imshow(np.array(image, dtype=np.dtype(int)))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    for i in range(0, num_pics):
        plt.subplot(2, n_col, i+2)
        relevance = relevances[i][1][0]
        fig = plt.imshow(relevance, cmap='seismic',
                         norm=MidpointNormalize(midpoint=0, vmin=relevance.min(), vmax=relevance.max()))

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        if i+2 > n_col:
            plt.xlabel(relevances[i][0], fontsize=12)
        else:
            fig.axes.get_xaxis().set_label('')
            plt.title(relevances[i][0], fontsize=12)
        fig.axes.get_yaxis().set_visible(False)

    if not show:
        fig_path = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', persist_string)

        plt.savefig(fig_path)
        plt.cla()
        plt.clf()
    else:
        plt.show()
