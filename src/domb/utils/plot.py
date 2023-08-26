"""
Functions for plotting, image creation, cmaps

"""

import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats


class CMaps():
    def __init__(self):
        """
        Class for custom colormaps storage.
        To use colormaps with `matplotlib.imshow` just call `plot.CMaps().cmap_name`.

        Attributes
        ----------
        cmap_cyan: matplotlib cmap
            cyan colormap
        cmap_yellow: matplotlib cmap
            yellow colormap
        cmap_red_green: matplotlib cmap
            red-green colormap (low-green, zero-black, high-red),
            especially suitable for differential images

        """
        # CFP cmap
        dict_cyan = {'red':(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),
                    'blue':(
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
                    'green':(
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0))}
        self.cmap_cyan = LinearSegmentedColormap('Cyan', dict_cyan)

        # YFP cmap
        dict_yellow = {'red':(
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
                    'blue':(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),
                    'green':(
                    (0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0))}
        self.cmap_yellow = LinearSegmentedColormap('Yellow', dict_yellow)

        # red-green cmap creation
        dict_red_green = {'red':(
                        (0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.0),
                        (0.55, 0.3, 0.7),
                        (1.0, 1.0, 1.0)),
                        'blue':(
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),
                        'green':(
                        (0.0, 1.0, 1.0),
                        (0.45, 0.7, 0.3),
                        (0.5, 0.0, 0.0),
                        (1.0, 0.0, 0.0))}
        self.cmap_red_green = LinearSegmentedColormap('RedGreen', dict_red_green)


        @staticmethod
        def plot_linearmap(cdict):
            newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
            rgba = newcmp(np.linspace(0, 1, 256))
            fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
            col = ['r', 'g', 'b']
            for xx in [0.25, 0.5, 0.75]:
                ax.axvline(xx, color='0.7', linestyle='--')
            for i in range(3):
                ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
            ax.set_xlabel('index')
            ax.set_ylabel('RGB')
            plt.show()


def toRGB(r_img:np.ndarray, g_img:np.ndarray, b_img:np.ndarray):
    """ Convert three 2d arrays to RGB (input arrays must have same size).

    Parameters
    ----------
    r_img: ndarray [x,y]
        2D array for red chennel
    g_img: ndarray [x,y]
        2D array for green chennel
    b_img: ndarray [x,y]
        2D array for blue chennel

    Returns
    -------
    rgb_img: ndarray [x,y]
        RGB image

    """
    r_norm_img = np.asarray(r_img, dtype='float')
    r_norm_img = (r_norm_img - np.min(r_norm_img)) / (np.max(r_norm_img) - np.min(r_norm_img))
    g_norm_img = np.asarray(g_img, dtype='float')
    g_norm_img = (g_norm_img - np.min(g_norm_img)) / (np.max(g_norm_img) - np.min(g_norm_img))
    b_norm_img = np.asarray(b_img, dtype='float')
    b_norm_img = (b_norm_img - np.min(b_norm_img)) / (np.max(b_norm_img) - np.min(b_norm_img))

    rgb_img = np.stack([r_norm_img, g_norm_img, b_norm_img], axis=-1) 
    return rgb_img


def arr_cascade_plot(input_arr:np.ndarray, y_shift:float=0.1):
    plt.figure(figsize=(20, 8))
    
    shift = 0
    for num_ROI in range(input_arr.shape[0]):
        prof_ROI = input_arr[num_ROI]
        plt.plot(prof_ROI+shift, alpha=.5, label=f'ROI {num_ROI}')
        shift += y_shift

    # for line_name in line_dict:
    #     line_lim = line_dict[line_name]
    #     plt.plot(line_lim, [-0.4] * len(line_lim), label=line_name, linewidth=4)

    # plt.vlines(x=[-20], ymin=[-0.1], ymax=[0.0], linewidth=3, color='k')
    # plt.text(x=-30, y=-0.2, s="0.1", size=15, rotation=90.)

    # plt.hlines(y=[-0.75], xmin=[-2], xmax=[8], linewidth=3, color='k')
    # plt.text(x=30, y=-1.15, s="10 s", size=15)

    plt.axis('off')
    plt.legend(loc=2)
    plt.show()


def stat_line_plot(arr_list: list,
                   lab_list:list,
                   stat_method:str='se',
                   stim_t:int=12, show_stim:bool=True, t_scale:int=2,
                   figsize:tuple=(15,10), x_lab:str='NA', y_lab:str='NA', plot_title:str='NA'):
    """ Line plot with variance wiskers for set of arrays `[t,val],
    could take as input results of `util.masking.label_prof_arr()` function

    Parameters
    ----------
    arr_list: list
        list of 2D arrays with profiles `[t, val]`
    lab_list: list
        list of labels for lines, should be same len with arr_list
    stat_method: str, optional {'se', 'iqr', 'ci'}
        method for line plot calculation: `se` - mean +/- standat error of mean,
        `iqr` - median +/- IQR, `ci` - mean +/- 95% confidence interval

    """
    time_line = np.linspace(0, arr_list[0].shape[1]*t_scale, \
                            num=arr_list[0].shape[1])

    # https://aegis4048.github.io/comprehensive_confidence_intervals_for_python_developers

    # mean, se
    arr_se_stat = lambda x: (np.mean(x, axis=0), \
                             np.std(x, axis=0)/np.sqrt(x.shape[1]))  # mean, se
    
    # meadian, IQR
    arr_iqr_stat = lambda x: (np.median(x, axis=0), \
                              stats.iqr(x, axis=0))

    # mean, CI
    arr_ci_stat = lambda x, alpha=0.05: (np.mean(x, axis=0), \
                                         stats.t.ppf(1-alpha/2, df=x.shape[1]-1) \
                                                    *np.std(x, axis=0, ddof=1)/np.sqrt(x.shape[1]))

    stat_dict = {'se':arr_se_stat,
                 'iqr':arr_iqr_stat,
                 'ci':arr_ci_stat}

    plt.figure(figsize=figsize)
    for num in range(len(arr_list)):
        arr = arr_list[num]
        lab = lab_list[num]
        arr_val, arr_var = stat_dict[stat_method](arr)

        plt.errorbar(time_line, arr_val,
                     yerr = arr_var,
                     fmt ='-o', capsize=2, label=lab)

    if show_stim:
        plt.axvline(x=stim_t, color='k', linestyle=':')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()