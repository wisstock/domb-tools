"""
Class for red-green maskig of co-recording WT and mutant protein

Optimized for individual neurons imaging

Requires WS_2x_2m type as input

"""

import numpy as np
from numpy import ma
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.express as px

from skimage.util import montage
from skimage.filters import rank
from skimage import morphology
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage import transform
from skimage import registration

from scipy import ndimage
from scipy import signal
from scipy import stats
from scipy import ndimage as ndi

from ..util import masking
from ..util import plot


class WTvsMut():
    def __init__(self, wt_img, mut_img, proc_mask, **kwargs):
        self.wt_img = wt_img
        self.mut_img = mut_img
        self.proc_mask = proc_mask

        # WT masking
        self.wt_up_mask, self.wt_up_label, self.wt_diff_img = self.up_mask_calc(self.wt_img,
                                                                                self.proc_mask,
                                                                                **kwargs)

        # mutant masking
        self.mut_up_mask, self.mut_up_label, self.mut_diff_img = self.up_mask_calc(self.mut_img,
                                                                                   self.proc_mask,
                                                                                   **kwargs)

        # masks modification
        self.connected_up_mask, self.connected_up_label = self.up_mask_connection(self.wt_up_mask,
                                                                                 self.mut_up_mask)
        self.halo_up_mask = np.copy(self.connected_up_mask)
        self.halo_up_mask[self.mut_up_mask] = 0
        self.halo_up_label = np.copy(self.connected_up_label)
        self.halo_up_label[self.mut_up_mask] = 0

        self.init_up_mask = np.copy(self.connected_up_mask)
        self.init_up_mask[self.halo_up_mask] = 0
        self.init_up_label = np.copy(self.connected_up_label)
        self.init_up_label[self.halo_up_mask] = 0


    @staticmethod
    def up_mask_calc(input_img_series, input_img_mask, sd_tolerance=2, base_frames=5, app_start=7, app_win=5):
        ref_img_series = filters.gaussian(input_img_series, sigma=1.25, channel_axis=0)

        img_base = np.mean(ref_img_series[:base_frames], axis=0)
        img_max = np.mean(ref_img_series[app_start:app_start+app_win], axis=0)

        img_diff = img_max - img_base
        img_diff = img_diff/np.max(np.abs(img_diff))

        diff_sd = np.std(ma.masked_where(~input_img_mask, img_diff))
        up_mask = img_diff > diff_sd * sd_tolerance

        up_mask_filt = morphology.opening(up_mask, footprint=morphology.disk(2))
        up_mask_filt = morphology.dilation(up_mask_filt, footprint=morphology.disk(1))
        up_label = measure.label(up_mask_filt)

        return up_mask_filt, up_label, img_diff


    @staticmethod
    def up_mask_connection(input_wt_mask, input_mutant_mask):
        wt_label, wt_num = ndi.label(input_wt_mask)

        sums = ndi.sum(input_mutant_mask, wt_label, np.arange(wt_num+1))
        connected = sums > 0
        debris_mask = connected[wt_label]

        fin_mask = np.copy(input_wt_mask)
        fin_mask[~debris_mask] = 0

        fin_label, fin_num = ndi.label(fin_mask)

        return fin_mask, fin_label


    def up_mask_pic(self):

        plt.figure(figsize=(20,20))
        ax0 = plt.subplot(121)
        ax0.set_title('Differential img, WT')
        ax0.imshow(self.wt_diff_img, cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('Differential img, mutant')
        ax1.imshow(self.mut_diff_img, cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
        ax1.axis('off')

        # ax2 = plt.subplot(223)
        # ax2.set_title(f'Up mask labels ({up_label.max()} regions)')
        # ax2.imshow(ctrl_img)
        # ax2.imshow(ma.masked_where(~up_mask_filt, up_label), alpha=0.5, cmap='bwr')
        # ax2.arrow(550,645,-30,0,width=10, alpha=0.25, color='white')
        # ax2.axis('off')

        # ax2 = plt.subplot(224)
        # ax2.set_title(f'Up mask labels ({up_label.max()} regions)')
        # ax2.imshow(ctrl_img)
        # ax2.imshow(ma.masked_where(~up_mask_filt, up_label), alpha=0.5, cmap='bwr')
        # ax2.arrow(550,645,-30,0,width=10, alpha=0.25, color='white')
        # ax2.axis('off')

        plt.tight_layout()
        plt.show()

