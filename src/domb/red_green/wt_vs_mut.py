import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import morphology
from skimage import exposure
from skimage import measure
from skimage import filters

from scipy import ndimage as ndi
from scipy import stats

from ..utils import masking
from ..utils import plot


class wt_vs_mut():
    def __init__(self, wt_img:np.ndarray, mut_img:np.ndarray,
                 proc_mask:np.ndarray, narrow_proc_mask:np.ndarray,
                 **kwargs):
        """ Class is designed to create differential images and insertion masks
        for two separate image time series (for instance, wild-type NCS and mutant NCS,
        or two distinct NCSs). The up_mask_calc method is responsible for detecting
        insertion regions.

        Suitable for image series with __single__ application/stimulation only.

        The class attributes contain various types of insertion masks
        and corresponding profiles for these masks (raw intensity and ΔF/F profiles).

        __Requires wf_x2_m2 instance type as input!__

        __NB: Optimized for long stimuli/applications with strong translocation
        and analysis of colocalization of the insertion sites.
        Methods of this class may not work with image series with weak translocation range.__

        Parameters
        ----------
        wt_img: ndarray [t,x,y]
            wild type images time series
        mut_img: ndarray [t,x,u]
            mutant images time series
        proc_mask: ndarray [x,y]
            cell processes boolean mask, extended
        narrow_proc_mask: ndarray [x,y]
            cell processes boolean mask, unextended

        Attributes
        ----------
        wt_img: ndarray [t,x,y]
            wild type images time series
        mut_img: ndarray [t,x,y]
            mutant images time series
        proc_mask: ndarray [x,y]
            cell processes boolean mask, extended
        narrow_proc_mask: ndarray [x,y]
            cell processes boolean mask, unextended
        wt_up_mask: ndarray [x,y]
            boolean mask of intensity increase regions for wild-type channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        wt_up_label: ndarray [x,y]
            label image of intensity increase regions for wild-type channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        wt_diff_img: ndarray [x,y]
            differential image of intensity changes after stimulation for wild-type channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        mut_up_mask: ndarray [x,y]
            boolean mask of intensity increase regions for mutant/2nd NCS channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        mut_up_label: ndarray [x,y]
            label image of intensity increase regions for mutant/2nd NCS channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        mut_diff_img: ndarray [x,y]
            differential image of intensity changes after stimulation for mutant/2nd NCS channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        connected_up_mask: ndarray [x,y]
            regions of `wt_up_mask` which overlay with `mut_up_mask`,
            created with `utils.masking.mask_connection()`
        halo_up_mask: ndarray [x,y]
            `connecter_up_mask` without mutant intensity increase regions
        init_up_mask: ndarray [x,y]
            `connected_up_mask` without `halo_up_mask`
            (only regions of `mut_up_mask` which overlay with `wt_up_mask`)
        wt_conn_arr: ndarray [int, t]
            intensity profiles of wild-type channel for each label element of `connected_up_mask`,
            created with `utils.masking.label_prof_arr()`
        wt_conn_df_arr: ndarray [ΔF/F, t]
            ΔF/F profiles of wild-type channel for each label element of `connected_up_mask`,
            created with `utils.masking.label_prof_arr()`
        wt_halo_arr: ndarray [int, t]
            intensity profiles of wild-type channel for each label element of `halo_up_mask`,
            created with `utils.masking.label_prof_arr()`
        wt_halo_df_arr: ndarray [ΔF/F, t]
            ΔF/F profiles of wild-type channel for each label element of `halo_up_mask`,
            created with `utils.masking.label_prof_arr()`
        wt_init_arr: ndarray [int, t]
            intensity profiles of wild-type channel for each label element of `init_up_mask`,
            created with `utils.masking.label_prof_arr()`
        wt_init_df_arr: ndarray [ΔF/F, t]
            ΔF/F profiles of wild-type channel for each label element of `init_up_mask`,
            created with `utils.masking.label_prof_arr()`
        mut_conn_arr: ndarray [int, t]
            intensity profiles of mutant/2nd NCS channel for each label element of `connected_up_mask`,
            created with `utils.masking.label_prof_arr()`
        mut_conn_df_arr: ndarray [ΔF/F, t]
            ΔF/F profiles of mutant/2nd NCS channel for each label element of `connected_up_mask`,
            created with `utils.masking.label_prof_arr()`
        mut_halo_arr: ndarray [int, t]
            intensity profiles of mutant/2nd NCS channel for each label element of `halo_up_mask`,
            created with `utils.masking.label_prof_arr()`
        mut_halo_df_arr: ndarray [ΔF/F, t]
            ΔF/F profiles of mutant/2nd NCS channel for each label element of `halo_up_mask`,
            created with `utils.masking.label_prof_arr()`
        mut_init_arr: ndarray [int, t]
            intensity profiles of mutant/2nd NCS channel for each label element of `init_up_mask`,
            created with `utils.masking.label_prof_arr()`
        mut_init_df_arr: ndarray [ΔF/F, t]
            ΔF/F profiles of mutant/2nd NCS channel for each label element of `init_up_mask`,
            created with `utils.masking.label_prof_arr()`

        """
        self.wt_img = wt_img
        self.mut_img = mut_img
        self.proc_mask = proc_mask
        self.narrow_proc_mask = narrow_proc_mask

        # WT masking
        self.wt_up_mask, self.wt_up_label, self.wt_diff_img = self.up_mask_calc(self.wt_img,
                                                                                self.proc_mask,
                                                                                **kwargs)

        # mutant masking
        self.mut_up_mask, self.mut_up_label, self.mut_diff_img = self.up_mask_calc(self.mut_img,
                                                                                   self.proc_mask,
                                                                                   **kwargs)

        # masks modification
        self.connected_up_mask, self.connected_up_label = masking.mask_connection(input_master_mask=self.wt_up_mask,
                                                                                  input_minor_mask=self.mut_up_mask)
        self.halo_up_mask = np.copy(self.connected_up_mask)
        self.halo_up_mask[self.mut_up_mask] = 0
        self.halo_up_label = np.copy(self.connected_up_label)
        self.halo_up_label[self.mut_up_mask] = 0

        self.init_up_mask = np.copy(self.connected_up_mask)
        self.init_up_mask[self.halo_up_mask] = 0
        self.init_up_label = np.copy(self.connected_up_label)
        self.init_up_label[self.halo_up_mask] = 0

        # WT ins profiles calc
        self.wt_conn_df_arr, self.wt_conn_arr = masking.label_prof_arr(input_label=self.connected_up_label,
                                                                       input_img_series=self.wt_img)
        self.wt_halo_df_arr, self.wt_halo_arr = masking.label_prof_arr(input_label=self.halo_up_label,
                                                                       input_img_series=self.wt_img)
        self.wt_init_df_arr, self.wt_init_arr = masking.label_prof_arr(input_label=self.init_up_label,
                                                                       input_img_series=self.wt_img)
        
        # # WT trans profiles calc
        # self.wt_conn_rois_trans_arr, self.wt_conn_tot_trans_arr = masking.trans_prof_arr(input_total_mask=self.narrow_proc_mask,
        #                                                                                  input_labels=self.connected_up_label,
        #                                                                                  input_img_series=self.wt_img)
        # self.wt_halo_rois_trans_arr, self.wt_halo_tot_trans_arr = masking.trans_prof_arr(input_total_mask=self.narrow_proc_mask,
        #                                                                                  input_labels=self.halo_up_label,
        #                                                                                  input_img_series=self.wt_img)
        # self.wt_init_rois_trans_arr, self.wt_init_tot_trans_arr = masking.trans_prof_arr(input_total_mask=self.narrow_proc_mask,
        #                                                                                  input_labels=self.init_up_label,
        #                                                                                  input_img_series=self.wt_img)

        # mut profiles calc
        self.mut_conn_df_arr, self.mut_conn_arr = masking.label_prof_arr(input_label=self.connected_up_label,
                                                                         input_img_series=self.mut_img)
        self.mut_halo_df_arr, self.mut_halo_arr = masking.label_prof_arr(input_label=self.halo_up_label,
                                                                         input_img_series=self.mut_img)
        self.mut_init_df_arr, self.mut_init_arr = masking.label_prof_arr(input_label=self.init_up_label,
                                                                         input_img_series=self.mut_img)
        
        # # mut trans profiles calc
        # self.mut_conn_rois_trans_arr, self.mut_conn_tot_trans_arr = masking.trans_prof_arr(input_total_mask=self.narrow_proc_mask,
        #                                                                                  input_labels=self.connected_up_label,
        #                                                                                  input_img_series=self.mut_img)
        # self.mut_halo_rois_trans_arr, self.mut_halo_tot_trans_arr = masking.trans_prof_arr(input_total_mask=self.narrow_proc_mask,
        #                                                                                  input_labels=self.halo_up_label,
        #                                                                                  input_img_series=self.mut_img)
        # self.mut_init_rois_trans_arr, self.mut_init_tot_trans_arr = masking.trans_prof_arr(input_total_mask=self.narrow_proc_mask,
        #                                                                                  input_labels=self.init_up_label,
        #                                                                                  input_img_series=self.mut_img)


    @staticmethod
    def up_mask_calc(input_img_series:np.ndarray, input_img_mask:np.ndarray,
                     sd_tolerance:int=2,
                     base_frames:int=5, stim_start:int=7, stim_win:int=5):
        """ Function for generating an insertion regions mask
        using the differential image method.

        Could be used in stand-alone mode as a static method of the WTvsMut class.

        Parameters
        ----------
        input_img_series: ndarray [t,x,y]
            image time series
        input_img_mask: ndarray [x,y]
            cell region boolean mask
        sd_tolerance: int
            insertion ("up") region detection threshold: the number
            of standard deviations of extracellular noise
            (measured in the area outside of `input_img_mask`)
        base_frames: int
            number of frames from the beginning
            of the image series used to create
            an image of basal fluorescence
        stim_start: int
            index of the frame where stimulation begins
        stim_win: int
            stimulation window, the number of frames following `stim_start` index
            that are used to create an image displaying maximal insertions

        Returns
        -------
        up_mask_filt: ndarray [x,y]
            boolean mask of intensity increase regions
        up_label: ndarray [x,y]
            label image of intensity increase regions  
        img_diff: ndarray [x,y]
            differential image of intensity changes after stimulation

        """
        ref_img_series = filters.gaussian(input_img_series, sigma=1.25, channel_axis=0)

        img_base = np.mean(ref_img_series[:base_frames], axis=0)
        img_max = np.mean(ref_img_series[stim_start:stim_start+stim_win], axis=0)

        img_diff = img_max - img_base
        img_diff = img_diff/np.max(np.abs(img_diff))

        diff_sd = np.std(ma.masked_where(~input_img_mask, img_diff))
        up_mask = img_diff > diff_sd * sd_tolerance

        up_mask_filt = morphology.opening(up_mask, footprint=morphology.disk(2))
        up_mask_filt = morphology.dilation(up_mask_filt, footprint=morphology.disk(1))
        up_label = measure.label(up_mask_filt)

        return up_mask_filt, up_label, img_diff


    def df_mean_prof_pic(self, fsize=(10,10), p_size=15, stim_t=8):
        time_line = np.linspace(0, self.wt_conn_df_arr.shape[1]*2, num=self.wt_conn_df_arr.shape[1])

        arr_stat = lambda x: (np.mean(x, axis=0),  np.std(x, axis=0)/np.sqrt(x.shape[1]))

        wt_conn_df_mean, wt_conn_df_sem = arr_stat(self.wt_conn_df_arr)
        wt_halo_df_mean, wt_halo_df_sem = arr_stat(self.wt_halo_df_arr)
        wt_init_df_mean, wt_init_df_sem = arr_stat(self.wt_init_df_arr)

        mut_conn_df_mean, mut_conn_df_sem = arr_stat(self.mut_conn_df_arr)
        mut_halo_df_mean, mut_halo_df_sem = arr_stat(self.mut_halo_df_arr)
        mut_init_df_mean, mut_init_df_sem = arr_stat(self.mut_init_df_arr)

        y_max = np.vstack([wt_conn_df_mean, wt_halo_df_mean, wt_init_df_mean, \
                           mut_conn_df_mean, mut_halo_df_mean, mut_init_df_mean]).max()
        y_max = y_max + y_max*0.1
        y_min = np.vstack([wt_conn_df_mean, wt_halo_df_mean, wt_init_df_mean, \
                           mut_conn_df_mean, mut_halo_df_mean, mut_init_df_mean]).min()
        y_min = y_min + y_min*0.1

        def two_arr_u_test(arr_1, arr_2):
            p_list = []
            for i in range(arr_1.shape[1]):
                t_1 = arr_1[:,i]
                t_2 = arr_2[:,i]
                amp_u_test = stats.mannwhitneyu(t_1, t_2)
                p_val = amp_u_test[1]

                if p_val > 0.05:
                    p_list.append(' ')
                elif p_val > 0.01:
                    p_list.append('*')
                elif p_val <= 0.01:
                    p_list.append('**')
            return p_list

        plt.figure(figsize=fsize)
        ax0 = plt.subplot(311)
        ax0.set_title('Connected up mask')
        ax0.errorbar(time_line, wt_conn_df_mean,
                     yerr = wt_conn_df_sem,
                     fmt ='-o', color='k', capsize=2, label='WT')
        ax0.errorbar(time_line, mut_conn_df_mean,
                     yerr = mut_conn_df_sem,
                    fmt ='-o', color='r', capsize=2, label='Mut.')
        conn_u_test_p = two_arr_u_test(self.wt_conn_df_arr, self.mut_conn_df_arr)
        for idx in range(len(time_line)):
            ax0.text(x=time_line[idx], y=wt_conn_df_mean[idx]*1.1,
                     s=conn_u_test_p[idx], fontsize=p_size)  
        ax0.axvline(x=stim_t, linestyle=':', color='k', label='Glu application')      
        ax0.hlines(y=0, xmin=0, xmax=time_line.max(), linestyles='--', color='k')
        ax0.set_xlabel('Time, s')
        ax0.set_ylabel('ΔF/F')
        ax0.set_ylim([y_min, y_max])
        ax0.legend()

        ax1 = plt.subplot(312)
        ax1.set_title('Halo up mask')
        ax1.errorbar(time_line, wt_halo_df_mean,
                     yerr = wt_halo_df_sem,
                     fmt ='-o', color='k', capsize=2, label='WT')
        ax1.errorbar(time_line, mut_halo_df_mean,
                     yerr = mut_halo_df_sem,
                    fmt ='-o', color='r', capsize=2, label='Mut.')
        halo_u_test_p = two_arr_u_test(self.wt_halo_df_arr, self.mut_halo_df_arr)
        for idx in range(len(time_line)):
            ax1.text(x=time_line[idx], y=wt_halo_df_mean[idx]*1.1,
                     s=halo_u_test_p[idx], fontsize=p_size)  
        ax1.axvline(x=stim_t, linestyle=':', color='k', label='Glu application')      
        ax1.hlines(y=0, xmin=0, xmax=time_line.max(), linestyles='--', color='k')
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('ΔF/F')
        ax1.set_ylim([y_min, y_max])
        ax1.legend()

        ax2 = plt.subplot(313)
        ax2.set_title('Init up mask')
        ax2.errorbar(time_line, wt_init_df_mean,
                     yerr = wt_init_df_sem,
                     fmt ='-o', color='k', capsize=2, label='WT')
        ax2.errorbar(time_line, mut_init_df_mean,
                     yerr = mut_init_df_sem,
                    fmt ='-o', color='r', capsize=2, label='Mut.')
        init_u_test_p = two_arr_u_test(self.wt_init_df_arr, self.mut_init_df_arr)
        for idx in range(len(time_line)):
            ax2.text(x=time_line[idx], y=wt_init_df_mean[idx]*1.1,
                     s=init_u_test_p[idx], fontsize=p_size)  
        ax2.axvline(x=stim_t, linestyle=':', color='k', label='Glu application')      
        ax2.hlines(y=0, xmin=0, xmax=time_line.max(), linestyles='--', color='k')
        ax2.set_xlabel('Time, s')
        ax2.set_ylabel('ΔF/F')
        ax2.set_ylim([y_min, y_max])
        ax2.legend()

        plt.tight_layout()
        plt.show()
 

    def diff_img_pic(self):
        cell_contour = measure.find_contours(self.proc_mask, level=0.5)

        plt.figure(figsize=(15,15))
        ax0 = plt.subplot(121)
        ax0.set_title(f'Differential img, WT ({np.max(self.wt_up_label)} ROIs)')
        ax0.imshow(self.wt_diff_img, cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
        for ce_c in cell_contour:
            ax0.plot(ce_c[:, 1], ce_c[:, 0], linewidth=1, color='w')
        ax0.axis('off')

        ax1 = plt.subplot(122)
        ax1.set_title(f'Differential img, mutant ({np.max(self.mut_up_label)} ROIs)')
        ax1.imshow(self.mut_diff_img, cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
        for ce_c in cell_contour:
            ax1.plot(ce_c[:, 1], ce_c[:, 0], linewidth=1, color='w')
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


    def connection_contour_pic(self):
        plt.figure(figsize=(10,10))
        plt.imshow(plot.toRGB(self.connected_up_mask, self.init_up_mask, np.zeros_like(self.wt_up_mask)))
        plt.axis('off')
        plt.title('Up mask overlay (red-WT, green-mutant)')
        plt.tight_layout()
        plt.show()


        cell_contour = measure.find_contours(self.proc_mask, level=0.5)
        conn_contour = measure.find_contours(self.connected_up_mask, level=0.5)
        init_contour = measure.find_contours(self.init_up_mask, level=0.5)

        plt.figure(figsize=(20,20))
        plt.imshow(np.mean(self.wt_img, axis=0) *-1, cmap='Greys')
        for ce_c in cell_contour:
            plt.plot(ce_c[:, 1], ce_c[:, 0], linewidth=1, color='w')
        for co_c in conn_contour:
            plt.plot(co_c[:, 1], co_c[:, 0], linewidth=1, color='r', alpha=.75)
        for in_c in init_contour:
            plt.plot(in_c[:, 1], in_c[:, 0], linewidth=1, color='g', alpha=.75)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()