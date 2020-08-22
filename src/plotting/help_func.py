import numpy as np
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

    Quelle:
    https://stackoverflow.com/questions/48598291/how-to-obtain-transparency-for-masked-values-in-customised-colormap-matplotlib
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        if np.abs(vmin) > np.abs(vmax):
            vmax = np.abs(vmin)
        else:
            vmin = -vmax
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class OOMFormatter(ScalarFormatter):
    """

    Quelle:
    https://stackoverflow.com/questions/43324152/python-matplotlib-colorbar-scientific-notation-base
    """
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format


def get_scientific_order(vmin: float, vmax: float) -> int:
    max_abs = np.maximum(np.abs(vmin), np.abs(vmax))
    for i in range(-10, 10):
        if int(max_abs * 10**i) > 0:
            return -i
