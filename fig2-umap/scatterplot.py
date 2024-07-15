import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

from natsort import natsorted


def scatter(
    x=None,
    y=None,
    hue=None,
    marker="o",
    data=None,
    hue_order=None,
    force_categorical_hue=False,
    force_continuous_hue=False,
    palette=None,
    color=None,
    cmap=None,
    size=20,
    alpha=0.6,
    highlight_index=None,
    highlight_x=None,
    highlight_y=None,
    highlight_marker="x",
    highlight_size=90,
    highlight_color="k",
    highlight_name=None,
    highlight_alpha=0.9,
    plot_kwargs=None,
    legend_kwargs=None,
    hide_legend=False,
    xlabel=None,
    ylabel=None,
    title=None,
    title_fontsize=20,
    title_fontweight="normal",
    title_loc="center",
    title_pad=None,
    show_title=False,
    xlabel_fontsize=16,
    ylabel_fontsize=16,
    xtick_labelsize=14,
    ytick_labelsize=14,
    xtick_labelrotation=0,
    ytick_labelrotation=0,
    tiny_axis=True,
    tiny_axis_xoffset=None,
    tiny_axis_yoffset=None,
    cbar_width=0.35,
    cbar_height=0.05,
    cbar_loc="lower right",
    cbar_orientation="horizontal",
    cbar_bbox_to_anchor=None,
    cbar_flip_ticks=False,
    cbar_title=None,
    cbar_title_fontsize=12,
    hide_cbar=False,
    equal_axes=True,
    ax=None,
    show=False,
    figsize=None,
    figfile=None,
):
    """
    Produces a scatter plot.

    Parameters
    ----------

    x : str or iterable object  
        Name of a column in `data` or an iterable of values to be plotted on the 
        x-axis. Required.

    y : str or iterable object  
        Name of a column in `data` or an iterable of values to be plotted on the 
        y-axis..

    hue : str or iterable object, optional  
        Name of a column in `data` or an iterable of hue categories to be used to 
        group data into stacked bars. If not provided, an un-stacked bar plot is created.  
        
    marker : str, dict or iterable object, optional  
        Marker style for the scatter plot. Accepts any of the following:
          * a `matplotlib marker`_ string
          * a ``dict`` mapping `hue` categories to a `matplotlib marker`_ string
          * a ``list`` of `matplotlib marker`_ strings, which should be the same 
              length as `x` and `y`.

    data : pandas.DataFrame, optional  
        A ``DataFrame`` object containing the input data. If provided, `x` and/or `y` should 
        be column names in `data`.   
        
    hue_order : iterable object, optional  
        List of `hue` categories in the order they should be plotted. If `hue_order` contains a 
        subset of all categories found in `hue`, only the supplied categories will be plotted.  
        If not provided, `hue` categories will be plotted in ``natsort.natsorted()`` order.  
        
    force_categorical_hue : bool, default=False  
        If ``True``, `hue` data will be treated as categorical, even if the data appear to 
        be continuous. This results in `color` being used to color the points rather than `cmap`.   

    force_continuous_hue : bool, default=False  
        If ``True``, `hue` data will be treated as continuous, even if the data appear to 
        be categorical. This results in `cmap` being used to color the points rather than `color`.  
        This may produce unexpected results and/or errors if used on non-numerical data.
        
    palette : dict, optional  
        Dictionary mapping `hue`, `x` or `y` names to colors. If if keys in `palette` match 
        more than one category, `hue` categories take priority. If `palette` is not provided, 
        bars are colored using `color` (if `hue` is ``None``) or a palette is generated 
        automatically using ``sns.hls_palette()``.  

    color : str or iterable object, optional. 
        Color for the plot markers. Can be any of the following:
          * a ``list`` or ``tuple`` containing RGB or RGBA values
          * a color string, either from `Matplotlib's set of named colors`_ or a hex color value
          * the name of a column in `data` that contains color values
          * a ``list`` of colors (either as strings or RGB tuples) to be used for `hue` categories. 
            If `colors` is shorter than the list of hue categories, colors will be reused.  
        
        .. tip::
            If a single RGB or RGBA ``list`` or ``tuple`` is provided and `hue` is also supplied, 
            there may be unexpected results as ``scatter()`` will attempt to map each of the 
            individual RGB(A) values to a hue category.   
        
        Only used if `hue` contains categorical data (`cmap` is used for continuous data). If not 
        provided, the `default Seaborn color palette`_ will be used. 
        
    cmap : str or matplotlib.color.Colormap, default='flare'   
        Colormap to be used for continuous `hue` data.  
        
    size : str or float or iterable object, default=20  
        Size of the scatter points. Either a   

    alpha : float, default=0.6  
        Alpha of the scatter points.  
        
    highlight_index : iterable object, optional  
        An iterable of index names (present in `data.index`) of points to be highlighted on 
        the plot. If provided, `highlight_x` and `highlight_y` are ignored.

    highlight_x : iterable object, optional  
        An iterable of x-values for highlighted points. Also requires `highlight_y`.
        
    highlight_y : iterable object, optional  
        An iterable of y-values for highlighted points. Also requires `highlight_x`.
        
    highlight_marker : str, default='x'  
        Marker style to be used for highlight points. Accepts any `matplotlib marker`_. 

    highlight_size : int, default=90  
        Size of the highlight marker.

    highlight_color : string or list of color values, default='k'
        Color of the highlight points.

    highlight_name : str, optional  
        Name of the highlights, to be used in the plot legend. If not supplied,
        highlight points will not be included in the legend.
        
    highlight_alpha : float, default=0.9  
        Alpha of the highlight points. 

    plot_kwargs : dict, optional  
        Dictionary containing keyword arguments that will be passed to ``pyplot.scatter()``.

    legend_kwargs : dict, optional  
        Dictionary containing keyword arguments that will be passed to ``ax.legend()``.

    hide_legend : bool, default=False  
        By default, a plot legend will be shown if multiple batches are plotted. If ``True``, 
        the legend will not be shown.  
        
    xlabel : str, optional  
        Text for the X-axis label. 

    ylabel : str, optional  
        Text for the Y-axis label.  
        
    xlabel_fontsize : int or float, default=16  
        Fontsize for the X-axis label text.

    ylabel_fontsize : int or float, default=16  
        Fontsize for the Y-axis label text.

    xtick_labelsize : int or float, default=14  
        Fontsize for the X-axis tick labels.  

    ytick_labelsize : int or float, default=14  
        Fontsize for the Y-axis tick labels.  

    xtick_labelrotation : int or float, default=0  
        Rotation of the X-axis tick labels.  
    
    ytick_labelrotation : int or float, default=0  
        Rotation of the Y-axis tick labels. 

    cbar_width : int, default=35  
        Width of the colorbar. Only used for continuous `hue` types.  

    cbar_height : int, default=5  
        Height of the colorbar. Only used for continuous `hue` types.  

    cbar_loc : str or iterable object, default='lower right'  
        Location of the colorbar. Accepts `any valid inset_axes() location`_.

    cbar_orientation : str, default='horizontal'  
        Orientation of the colorbar. Options are ``'horizontal'`` and ``'vertical'``.

    cbar_bbox_to_anchor : list or tuple, optional
        bbox_to_anchor for the colorbar. Used in combination with `cbar_loc` to provide 
        fine-grained positioning of the colorbar.

    cbar_flip_ticks : bool, default=False  
        Flips the position of colorbar ticks. Ticks default to the bottom if `cbar_orientation` 
        is  ``'horizontal'`` and the left if  `cbar_orientation` is ``'vertical'``.  

    cbar_title : str, optional  
        Colorbar title. If not provided, `hue` is used.  

    cbar_title_fontsize : int or float, default=12  
        Fontsize for the colorbar title.  
        
    hide_cbar : bool, default=False. 
        If ``True``, the color bar will be hidden on plots with continuous `hue` values.  
        
    equal_axes : bool, default=True
        If ```True```, the the limits of the x- and y-axis will be equal.

    ax : mpl.axes.Axes, default=None
        Pre-existing axes for the plot. If not provided, a new axes will be created.

    show :bool, default=False  
        If ``True``, plot is shown and the plot ``Axes`` object is not returned. Default
        is ``False``, which does not call ``pyplot.show()`` and returns the ``Axes`` object.

    figsize : iterable object, default=[6, 4]  
        List containing the figure size (as ``[x-dimension, y-dimension]``) in inches. 

    figfile : str, optional  
        Path at which to save the figure file. If not provided, the figure is not saved
        and is either shown (if `show` is ``True``) or the ``Axes`` object is returned.  
        
        
    .. _matplotlib marker: 
        https://matplotlib.org/stable/api/markers_api.html  

    .. _Matplotlib's set of named colors:
        https://matplotlib.org/stable/gallery/color/named_colors.html
        
    .. _default Seaborn color palette: 
        https://seaborn.pydata.org/generated/seaborn.color_palette.html  

    .. _any valid inset_axes() location: 
        https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.inset_axes.html
        
    """
    # process input data
    if data is None:
        _data = {}
        _data["x"] = x
        x = "x"
        _data["y"] = y
        y = "y"
        if hue is not None:
            _data["hue"] = hue
            hue = "hue"
        df = pd.DataFrame(_data)
    else:
        df = data.copy()

    # figure size
    if figsize is None:
        figsize = [6, 6]

    # hue and color
    continuous_hue = False
    if hue is not None:
        if force_continuous_hue:
            continuous_hue = True
        elif all([isinstance(h, float) for h in df[hue]]) and not force_categorical_hue:
            continuous_hue = True
        if continuous_hue:
            continuous_hue = True
            hue_order = []
            if cmap is None:
                cmap = sns.color_palette("flare", as_cmap=True)
            else:
                cmap = plt.get_cmap(cmap)
            min_hue = max(0, df[hue].min())
            max_hue = np.ceil(df[hue].max())
            df["color"] = [cmap((h - min_hue) / (max_hue - min_hue)) for h in df[hue]]
        else:
            if hue_order is None:
                hue_order = natsorted(list(set(df[hue])))
            if palette is not None:
                missing_color = color if color is not None else "lightgrey"
                hue_dict = {h: palette.get(h, missing_color) for h in hue_order}
            else:
                n_colors = max(1, len(hue_order))
                if color is None:
                    color = sns.color_palette(n_colors=n_colors)
                if len(color) < n_colors:
                    color = itertools.cycle(color)
                hue_dict = {h: c for h, c in zip(hue_order, color)}
            df["color"] = [hue_dict[h] for h in df[hue]]
    else:
        hue_order = []
        if isinstance(color, str) and color in df.columns:
            df["color"] = df[color]
        elif color is not None:
            df["color"] = [color] * df.shape[0]
        else:
            df["color"] = [sns.color_palette()[0]] * df.shape[0]

    # plot kwargs
    default_plot_kwargs = {"linewidths": 0}
    if plot_kwargs is not None:
        default_plot_kwargs.update(plot_kwargs)
    plot_kwargs = default_plot_kwargs

    # scatterplot
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    if hue_order:
        for h in hue_order[::-1]:
            d = df[df[hue] == h]
            ax.scatter(
                d[x],
                d[y],
                c=d["color"],
                s=size,
                marker=marker,
                alpha=alpha,
                label=h,
                **plot_kwargs,
            )
    else:
        ax.scatter(
            df[x],
            df[y],
            c=df["color"],
            s=size,
            marker=marker,
            alpha=alpha,
            **plot_kwargs,
        )

    # highlighted points
    highlight = any(
        [
            highlight_index is not None,
            all([highlight_x is not None, highlight_y is not None]),
        ]
    )
    if highlight:
        if highlight_index is not None:
            hi_index = [h for h in highlight_index if h in df.index.values]
            hi_df = df.loc[hi_index]
            highlight_x = hi_df[x]
            highlight_y = hi_df[y]
        plt.scatter(
            highlight_x,
            highlight_y,
            zorder=10,
            s=highlight_size,
            c=highlight_color,
            alpha=highlight_alpha,
            marker=highlight_marker,
            label=highlight_name,
        )

    # legend
    if not continuous_hue:
        if hue is not None and not hide_legend:
            default_legend_kwargs = {"frameon": True, "loc": "best", "fontsize": 12}
            if legend_kwargs is not None:
                default_legend_kwargs.update(legend_kwargs)
            legend_kwargs = default_legend_kwargs
            ax.legend(**legend_kwargs)

    # colorbar
    elif not hide_cbar:
        if cbar_orientation == "horizontal":
            width = max([cbar_width, cbar_height])
            height = min([cbar_width, cbar_height])
        else:
            width = min([cbar_width, cbar_height])
            height = max([cbar_width, cbar_height])
        cbar_bounds = get_inset_axes_bounds(
            cbar_loc, cbar_bbox_to_anchor, width, height
        )
        cbax = ax.inset_axes(cbar_bounds)

        max_hue = np.ceil(df[hue].max())
        min_hue = max(0, df[hue].min())
        norm = mpl.colors.Normalize(vmin=min_hue, vmax=max_hue)

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbax,
            orientation=cbar_orientation,
        )
        if cbar_orientation == "horizontal":
            ticks_position = "bottom" if cbar_flip_ticks else "top"
            cbax.xaxis.set_ticks_position(ticks_position)
            cbax.xaxis.set_label_position(ticks_position)
            cbar.ax.set_xlabel(
                cbar_title, fontsize=cbar_title_fontsize,
            )
        else:
            ticks_position = "left" if cbar_flip_ticks else "right"
            cbax.yaxis.set_ticks_position(ticks_position)
            cbax.yaxis.set_label_position(ticks_position)
            cbar.ax.set_ylabel(
                cbar_title, fontsize=cbar_title_fontsize,
            )

    # style the plot
    ax.set_xlabel(xlabel if xlabel is not None else x, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else y, fontsize=ylabel_fontsize)
    ax.tick_params(
        axis="x", labelsize=xtick_labelsize, labelrotation=xtick_labelrotation
    )
    ax.tick_params(
        axis="y", labelsize=ytick_labelsize, labelrotation=ytick_labelrotation
    )

    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position(("outward", 10))

    if equal_axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        axlim = [min([xlim[0], ylim[0]]), max([xlim[1], ylim[1]])]
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)

    if show_title and title is not None:
        ax.set_title(
            title,
            loc=title_loc,
            pad=title_pad,
            fontsize=title_fontsize,
            fontweight=title_fontweight,
        )
        
    if tiny_axis:
        # get coords for the UMAP-specific axes
        if tiny_axis_xoffset is None:
            tiny_axis_xoffset = 0
        if tiny_axis_yoffset is None:
            tiny_axis_yoffset = 0
        xmin = df["x"].min()
        xmax = df["x"].max()
        ymin = df["y"].min()
        ymax = df["y"].max()
        x_range = abs(xmax - xmin)
        y_range = abs(ymax - ymin)
        x_offset = x_range * tiny_axis_xoffset
        y_offset = y_range * tiny_axis_yoffset
        x_start = xmin + x_offset
        y_start = ymin + x_offset
        x_end = xmin + (x_range / 5) + x_offset
        x_center = x_start + ((x_end - x_start) / 2)
        y_end = ymin + (y_range / 5) + y_offset
        y_center = y_start + ((y_end - y_start) / 2)
        
        # draw the new "mini" axis lines
        ax.hlines(y_start, x_start, x_end, "k", lw=2)
        ax.vlines(x_start, y_start, y_end, "k", lw=2)
        ax.annotate(
            xlabel,
            xy=(x_center, ymin),
            xytext=(0, -5),
            textcoords="offset points",
            fontsize=xlabel_fontsize,
            ha="center",
            va="top",
        )
        ax.annotate(
            ylabel,
            xy=(xmin, y_center),
            xytext=(-5, 0),
            textcoords="offset points",
            fontsize=ylabel_fontsize,
            rotation="vertical",
            ha="right",
            va="center",
        )

        # remove the normal axis lines
        ax.set_xlabel("", fontsize=0)
        ax.set_ylabel("", fontsize=0)
        for s in ["left", "right", "top", "bottom"]:
            ax.spines[s].set_visible(False)
   
    # hide the ticks
    ax.set_xticks([])
    ax.set_yticks([])
        
    # save, show or return the ax
    if figfile is not None:
        plt.tight_layout()
        plt.savefig(figfile)
    elif show:
        plt.show()
    else:
        return ax


def get_inset_axes_bounds(loc, bbox_to_anchor, width, height):
    if bbox_to_anchor is None:
        loc_dict = {
            "upper left": [0, 1 - height],
            "upper center": [0.5 - width / 2, 1 - height],
            "upper right": [1 - width, 1 - height],
            "center left": [0, 0.5 - height / 2],
            "center": [0.5 - width / 2, 0.5 - height / 2],
            "center right": [1 - width, 0.5 - height / 2],
            "lower left": [0, 0],
            "lower center": [0.5 - width / 2, 0],
            "lower right": [1 - width, 0],
        }
        x0, y0 = loc_dict.get(loc, [0, 0])
    else:
        x, y = bbox_to_anchor[:2]
        loc_dict = {
            "upper left": [x, y - height],
            "upper center": [x - width / 2, y - height],
            "upper right": [x - width, y - height],
            "center left": [x, y - height / 2],
            "center": [x - width / 2, y - height / 2],
            "center right": [x - width, y - height / 2],
            "lower left": [x, y],
            "lower center": [x - width / 2, y],
            "lower right": [x - width, y],
        }
        x0, y0 = loc_dict.get(loc, [0, 0])
    return [x0, y0, width, height]