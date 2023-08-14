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


class CMaps():
    def __init__(self):
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


def toRGB(r_img, g_img, b_img):
    """ Return three 2d arrays as RGB stack (input arrays must have same size)

    """
    r_norm_img = np.asarray(r_img, dtype='float')
    r_norm_img = (r_norm_img - np.min(r_norm_img)) / (np.max(r_norm_img) - np.min(r_norm_img))
    g_norm_img = np.asarray(g_img, dtype='float')
    g_norm_img = (g_norm_img - np.min(g_norm_img)) / (np.max(g_norm_img) - np.min(g_norm_img))
    b_norm_img = np.asarray(b_img, dtype='float')
    b_norm_img = (b_norm_img - np.min(b_norm_img)) / (np.max(b_norm_img) - np.min(b_norm_img))

    return np.stack([r_norm_img, g_norm_img, b_norm_img], axis=-1) 


def arr_cascade_plot(input_arr, y_shift=0.1):
    """ cascade plot of 2D arr (ROI per line)

    """
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