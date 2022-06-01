"""
Utility functions
"""

import io

import numpy as np
import pandas as pd

from missionbio.mosaic.plotting import mpimg, plt, require_seaborn, sns


@require_seaborn
def static_fig(fig, figsize=(7, 7), scale=3, ax=None):
    """
    Convert plotly figure to matplotlib.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    figsize : 2-tuple
        The size of the matplotlib figure in case
        no axis is provided.
    scale : float (default: 3)
        The scale factor used when exporting the figure.
        A value greater than 1 increases the resolution
        of the plotly figure and a value less than 1
        reduces the resolution of the plotly figure.
    ax : matplotlib.pyplot.axis
        The axis to show the image on.
    """

    i = fig.to_image(format='png', scale=scale)
    i = io.BytesIO(i)
    i = mpimg.imread(i, format='png')

    if ax is None:
        sns.set(style='white')
        plt.figure(figsize=figsize)
        plt.imshow(i)
        plt.axis('off')
        ax = plt.gca()
    else:
        ax.imshow(i)

    return ax


def get_indexes(from_list, find_list):
    """
    Find the intersection of arrays.

    Parameters
    ----------
    from_list : list-like
        The list for which indexes are to be found.

    find_list : list-like
        The list containing values which are to be
        matched in `from_list`.

    Returns
    -------
    indexes
        Returns the indexes in from_list for
        values that are found in find_list.

    Notes
    -----
    get_indexes is much faster than `np.where(np.isin(from_list, find_list))`,
    but it uses more memory since it duplicates the data.
    """

    df_find = pd.DataFrame(find_list, columns=['value'])
    df_from = pd.DataFrame(list(zip(from_list, np.arange(len(from_list)))), columns=['value', 'index'])
    indexes = pd.merge(df_from, df_find, on='value', how='inner')['index'].values
    return indexes


def clipped_values(values):
    """
    Cut at the 1st and 99th percentiles.

    Parameters
    ----------
    values : list-like
        List of float or int values.

    Returns
    -------
    2-tuple
        It contains the values corresponding to the
        1st percentile and 99th percentile of the given
        values.
    """
    values = np.sort(values)
    val_99 = values[int(0.99 * len(values))]
    val_1 = values[int(0.01 * len(values))]
    return val_1, val_99


def extend_docs(cls):
    """
    Extend the superclass documentation to subclass.

    Parameters
    ----------
    f : function
        function to decorate

    Raises
    ------
    ValueError
        Raise exception if seaborn could not be imported.

    Also
    ----
    Can be used as a decorator.
    """

    import types

    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            print(func, 'needs doc')
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, types.FunctionType):
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ += parfunc.__doc__
                    break

    return cls
