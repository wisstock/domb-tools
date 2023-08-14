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

        # WT profiles calc
        self.wt_conn_prof_dict, self.wt_conn_df_arr = masking.label_prof_arr_dict(input_labels=self.connected_up_label,
                                                                                  input_img_series=self.wt_img)
        self.wt_halo_prof_dict, self.wt_halo_df_arr = masking.label_prof_arr_dict(input_labels=self.halo_up_label,
                                                                                  input_img_series=self.wt_img)
        self.wt_init_prof_dict, self.wt_init_df_arr = masking.label_prof_arr_dict(input_labels=self.init_up_label,
                                                                                  input_img_series=self.wt_img)
        
        # mut profiles calc
        self.mut_conn_prof_dict, self.mut_conn_df_arr = masking.label_prof_arr_dict(input_labels=self.connected_up_label,
                                                                                    input_img_series=self.mut_img)
        self.mut_halo_prof_dict, self.mut_halo_df_arr = masking.label_prof_arr_dict(input_labels=self.halo_up_label,
                                                                                    input_img_series=self.mut_img)
        self.mut_init_prof_dict, self.mut_init_df_arr = masking.label_prof_arr_dict(input_labels=self.init_up_label,
                                                                                    input_img_series=self.mut_img)


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


    def df_mean_prof_pic(self, fsize=(15,10), stim_t=12):
        time_line = np.linspace(0, self.wt_conn_df_arr.shape[1]*2, num=self.wt_conn_df_arr.shape[1])

        arr_stat = lambda x: (np.mean(x, axis=0),  np.std(x, axis=0)/np.sqrt(x.shape[1]))

        wt_conn_df_mean, wt_conn_df_sem = arr_stat(self.wt_conn_df_arr)
        wt_halo_df_mean, wt_halo_df_sem = arr_stat(self.wt_halo_df_arr)
        wt_init_df_mean, wt_init_df_sem = arr_stat(self.wt_init_df_arr)

        mut_conn_df_mean, mut_conn_df_sem = arr_stat(self.mut_conn_df_arr)
        mut_halo_df_mean, mut_halo_df_sem = arr_stat(self.mut_halo_df_arr)
        mut_init_df_mean, mut_init_df_sem = arr_stat(self.mut_init_df_arr)

        plt.figure(figsize=fsize)
        ax0 = plt.subplot(311)
        ax0.set_title('Connected up mask')
        ax0.errorbar(time_line, wt_conn_df_mean,
                     yerr = wt_conn_df_sem,
                     fmt ='-o', color='k', capsize=2, label='WT')
        ax0.errorbar(time_line, mut_conn_df_mean,
                     yerr = mut_conn_df_sem,
                    fmt ='-o', color='r', capsize=2, label='Mut.')
        ax0.vlines(x=stim_t,
                   ymin=wt_conn_df_mean.min(),
                   ymax=wt_conn_df_mean.max(),
                   linestyles=':', color='k', label='Glu application')
        ax0.hlines(y=0, xmin=0, xmax=time_line.max(), linestyles='--', color='k')
        ax0.set_xlabel('Time, s')
        ax0.set_ylabel('ΔF/F')
        ax0.legend()

        ax1 = plt.subplot(312)
        ax1.set_title('Halo up mask')
        ax1.errorbar(time_line, wt_halo_df_mean,
                     yerr = wt_halo_df_sem,
                     fmt ='-o', color='k', capsize=2, label='WT')
        ax1.errorbar(time_line, mut_halo_df_mean,
                     yerr = mut_halo_df_sem,
                    fmt ='-o', color='r', capsize=2, label='Mut.')
        ax1.vlines(x=stim_t,
                   ymin=wt_halo_df_mean.min(),
                   ymax=wt_halo_df_mean.max(),
                   linestyles=':', color='k', label='Glu application')
        ax1.hlines(y=0, xmin=0, xmax=time_line.max(), linestyles='--', color='k')
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('ΔF/F')
        ax1.legend()

        ax2 = plt.subplot(313)
        ax2.set_title('Init up mask')
        ax2.errorbar(time_line, wt_init_df_mean,
                     yerr = wt_init_df_sem,
                     fmt ='-o', color='k', capsize=2, label='WT')
        ax2.errorbar(time_line, mut_init_df_mean,
                     yerr = mut_init_df_sem,
                    fmt ='-o', color='r', capsize=2, label='Mut.')
        ax2.vlines(x=stim_t,
                   ymin=wt_init_df_mean.min(),
                   ymax=wt_init_df_mean.max(),
                   linestyles=':', color='k', label='Glu application')
        ax2.hlines(y=0, xmin=0, xmax=time_line.max(), linestyles='--', color='k')
        ax2.set_xlabel('Time, s')
        ax2.set_ylabel('ΔF/F')
        ax2.legend()

        plt.tight_layout()
        plt.show()
 

    def diff_img_pic(self):
        plt.figure(figsize=(15,15))
        ax0 = plt.subplot(121)
        ax0.set_title('Differential img, WT')
        ax0.imshow(self.wt_diff_img, cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('Differential img, mutant')
        ax1.imshow(self.mut_diff_img, cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
        ax1.axis('off')

        plt.tight_layout()
        plt.show()


    def mask_diff_pic(self):
        up_mask_diff = plot.toRGB(r_img=self.wt_up_mask,
                                  g_img=self.mut_up_mask,
                                  b_img=np.zeros_like(self.wt_up_mask))
        con_v_init_diff = plot.toRGB(r_img=self.connected_up_mask,
                                     g_img=self.init_up_mask,
                                     b_img=np.zeros_like(self.connected_up_mask))

        plt.figure(figsize=(15,15))
        ax0 = plt.subplot(121)
        ax0.set_title('WT up mask (red) vs. mut. up mask (green)')
        ax0.imshow(up_mask_diff)
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title('Connected mask (red) vs. init mask (green)')
        ax1.imshow(con_v_init_diff)
        ax1.axis('off')

        plt.tight_layout()
        plt.show()


    def mask_set_pic(self):
        plt.figure(figsize=(15,15))
        ax0 = plt.subplot(131)
        ax0.set_title('Connected up mask elements')
        ax0.imshow(self.connected_up_label, cmap='jet')
        ax0.axis('off')

        ax1 = plt.subplot(132)
        ax1.set_title('Halo up mask elements')
        ax1.imshow(self.halo_up_label, cmap='jet')
        ax1.axis('off')

        ax2 = plt.subplot(133)
        ax2.set_title('Init up mask elements')
        ax2.imshow(self.init_up_label, cmap='jet')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()