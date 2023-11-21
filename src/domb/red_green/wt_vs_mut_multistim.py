import numpy as np
from numpy import ma

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import morphology
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage import segmentation

from scipy import ndimage as ndi
from scipy import stats

from ..utils import masking
from ..utils import plot


class wt_vs_mut_multistim():
    def __init__(self, wt_img:np.ndarray, mut_img:np.ndarray,
                 proc_mask:np.ndarray, narrow_proc_mask:np.ndarray,
                 **kwargs):
        """ Class is designed to create differential images and insertion masks
        for two separate image time series (for instance, wild-type NCS and mutant NCS,
        or two distinct NCSs). The up_mask_calc method is responsible for detecting
        insertion regions.

        Suitable for image series with __multiple__ application/stimulation only.

        The class attributes contain various types of insertion masks
        and corresponding profiles for these masks (raw intensity and ΔF/F profiles).

        __Requires wf_x2_m2 instance type as input!__

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
        wt_up_mask_list: ndarray [stimuli, x,y]
            boolean mask of intensity increase regions for wild-type channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        wt_up_label_list: ndarray [stimuli, x,y]
            label image of intensity increase regions for wild-type channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        wt_diff_img_list: ndarray [stimuli, x,y]
            differential image of intensity changes after stimulation for wild-type channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        mut_up_mask_list: ndarray [stimuli, x,y]
            boolean mask of intensity increase regions for mutant/2nd NCS channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        mut_up_label_list: ndarray [stimuli, x,y]
            label image of intensity increase regions for mutant/2nd NCS channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        mut_diff_img_list: ndarray [stimuli, x,y]
            differential image of intensity changes after stimulation for mutant/2nd NCS channel,
            created with `red_green.wt_ws_mut.up_mask_calc()`
        connected_up_mask_list: ndarray [stimuli, x,y]
            regions of `wt_up_mask` which overlay with `mut_up_mask` (pairwise for each stimulation),
            created with `utils.masking.mask_connection()`

        """
        self.wt_img = wt_img
        self.mut_img = mut_img
        self.proc_mask = proc_mask
        self.narrow_proc_mask = narrow_proc_mask

        cell_contour_mask = np.copy(self.proc_mask)
        cell_contour_mask[self.narrow_proc_mask] = 0
        # WT masking
        self.wt_up_mask_list, self.wt_up_label_list, self.wt_diff_img_list = self.up_mask_calc(self.wt_img,
                                                                                               cell_contour_mask,
                                                                                               **kwargs)
        
        # mutant masking
        self.mut_up_mask_list, self.mut_up_label_list, self.mut_diff_img_list = self.up_mask_calc(self.mut_img,
                                                                                                  cell_contour_mask,
                                                                                                  **kwargs)

        # self.wt_diff_img_series = self.diff_img_series_calc(self.wt_img)
        # self.mut_diff_img_series = self.diff_img_series_calc(self.mut_img)

        # masks modification
        self.connected_up_mask_list, self.connected_up_label_list = np.empty_like(self.proc_mask), np.empty_like(self.proc_mask)
        for stim_num in range(self.wt_up_mask_list.shape[0]):
            connected_up_mask, connected_up_label = masking.mask_connection(input_master_mask=self.wt_up_mask_list[stim_num],
                                                                            input_minor_mask=self.mut_up_mask_list[stim_num])
            self.connected_up_mask_list = np.dstack((self.connected_up_mask_list,
                                                     connected_up_mask))
            self.connected_up_label_list = np.dstack((self.connected_up_label_list,
                                                     connected_up_label))
        self.connected_up_mask_list = np.moveaxis(self.connected_up_mask_list, -1, 0)
        self.connected_up_label_list = np.moveaxis(self.connected_up_label_list, -1, 0)


    @staticmethod
    def up_mask_calc(input_img_series:np.ndarray, input_img_mask:np.ndarray,
                     sd_tolerance:int=2, stim_list:list[int]=[10],
                     base_frames:int=2, stim_win:int=1, stim_shift:int=0):
        """ Function for generating a set of insertion regions mask
        for image series with multiple stimuli/applications using the differential image method.

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
        stim_list: list[int]
            indexes of the frames where stimulations begins
        stim_win: int
            stimulation window, the number of frames following each `stim_list` indexes
            that are used to create set of images displaying maximal insertions after each stimuli

        Returns
        -------
        up_mask_filt_list: ndarray [stimuli, x,y]
            boolean masks of intensity increase regions
        up_label_list: ndarray [stimuli, x,y]
            label images of intensity increase regions  
        img_diff_list: ndarray [stimuli, x,y]
            differential images of intensity changes after stimulation

        """
        ref_img_series = filters.gaussian(input_img_series, sigma=1.5, channel_axis=0)

        up_mask_filt_list = []
        up_label_list = []
        img_diff_list = []
        stim_i_pairs_list = []
        for stim_i in stim_list:
            base_i = stim_i-base_frames

            img_base = np.mean(ref_img_series[base_i:stim_i], axis=0) 

            start_i = stim_i+stim_shift
            fin_i = stim_i+stim_win + stim_shift
            img_max = np.mean(ref_img_series[start_i:fin_i], axis=0)

            img_diff = img_max - img_base
            img_diff_norm = img_diff / np.max(np.abs(img_diff))

            diff_sd = np.std(img_diff, where=input_img_mask)
            up_mask = img_diff > diff_sd * sd_tolerance
            # up_mask =  img_diff_norm > sd_tolerance

            # NEW MASK PROCESSING
            up_mask_filt = ndi.binary_fill_holes(up_mask)
            up_mask_filt = morphology.remove_small_objects(up_mask_filt, min_size=20)
            up_mask_filt = segmentation.expand_labels(up_mask_filt, distance=1)

            # OLD MASK PROCESSING
            # up_mask_filt = morphology.opening(up_mask, footprint=morphology.disk(3))
            # up_mask_filt = morphology.dilation(up_mask_filt, footprint=morphology.disk(2))

            up_mask_filt = segmentation.clear_border(up_mask_filt)
            up_label = measure.label(up_mask_filt)

            up_mask_filt_list.append(up_mask_filt)
            up_label_list.append(up_label)
            img_diff_list.append(img_diff_norm)

            stim_i_pairs_list.append([[base_i, stim_i],[start_i, fin_i]])

        print(stim_i_pairs_list)

        return np.asarray(up_mask_filt_list), np.asarray(up_label_list), np.asarray(img_diff_list)


    @staticmethod
    def up_mask_calc_old(input_img_series:np.ndarray, input_img_mask:np.ndarray,
                     sd_tolerance:int=2, base_frames:int=5,
                     stim_list:list[int]=[10], stim_win:int=5):
        """ Function for generating a set of insertion regions mask
        for image series with multiple stimuli/applications using the differential image method.

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
        stim_list: list[int]
            indexes of the frames where stimulations begins
        stim_win: int
            stimulation window, the number of frames following each `stim_list` indexes
            that are used to create set of images displaying maximal insertions after each stimuli

        Returns
        -------
        up_mask_filt_list: ndarray [stimuli, x,y]
            boolean masks of intensity increase regions
        up_label_list: ndarray [stimuli, x,y]
            label images of intensity increase regions  
        img_diff_list: ndarray [stimuli, x,y]
            differential images of intensity changes after stimulation

        """
        ref_img_series = filters.gaussian(input_img_series, sigma=1.25, channel_axis=0)

        img_base = np.mean(ref_img_series[:base_frames+1], axis=0)

        up_mask_filt_list = []
        up_label_list = []
        img_diff_list = []
        stim_i_pairs_list = []
        for stim_start in stim_list:
            start_i = stim_start
            stop_i = stim_start+stim_win+1

            stim_i_pairs_list.append([start_i, stop_i-1])

            img_max = np.mean(ref_img_series[start_i:stop_i], axis=0)

            img_diff = img_max - img_base
            img_diff_norm = img_diff/np.max(np.abs(img_diff))

            diff_sd = np.std(img_diff, where=input_img_mask)
            up_mask = img_diff > diff_sd * sd_tolerance
            # up_mask =  img_diff_norm > sd_tolerance

            up_mask_filt = morphology.opening(up_mask, footprint=morphology.disk(2))
            up_mask_filt = morphology.dilation(up_mask_filt, footprint=morphology.disk(1))
            up_mask_filt = segmentation.clear_border(up_mask_filt)
            up_label = measure.label(up_mask_filt)

            up_mask_filt_list.append(up_mask_filt)
            up_label_list.append(up_label)
            img_diff_list.append(img_diff_norm)

        print(stim_i_pairs_list)

        return np.asarray(up_mask_filt_list), np.asarray(up_label_list), np.asarray(img_diff_list)


    @staticmethod
    def diff_img_series_calc(input_img_series:np.ndarray, diff_step=1):
        diff_img_series = []
        for i in range(input_img_series.shape[0]-diff_step):
            img_1 = input_img_series[i]
            img_2 = input_img_series[i+diff_step]
            diff_img = img_2 - img_1
            diff_img_series.append(diff_img)
        return np.asarray(diff_img_series)


    def diff_img_pic(self):
        """ Differential images for each simulation.
        
        """
        cell_contour = measure.find_contours(self.proc_mask, level=0.5)

        stim_num = self.wt_up_mask_list.shape[0]
        fig = plt.figure(figsize= (30, 30))
        for i in range(stim_num):
            ax0 = fig.add_subplot(2, stim_num, i+1)
            ax0.text(40, 40, f'WT ({int(np.max(self.wt_up_label_list[i]))} ROIs)', fontsize=20, color='white')
            ax0.imshow(self.wt_diff_img_list[i], cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
            for ce_c in cell_contour:
                ax0.plot(ce_c[:, 1], ce_c[:, 0], linewidth=1, color='w')
            ax0.axis('off')
            ax0.axis('off')

            ax1 = fig.add_subplot(2, stim_num, i+1+stim_num)
            ax1.text(40, 40, f'Mut. ({int(np.max(self.mut_up_label_list[i]))} ROIs)', fontsize=20, color='white')
            ax1.imshow(self.mut_diff_img_list[i], cmap=plot.CMaps().cmap_red_green, vmax=1, vmin=-1)
            for ce_c in cell_contour:
                ax1.plot(ce_c[:, 1], ce_c[:, 0], linewidth=2, color='w')
            ax1.axis('off')
        plt.tight_layout()
        plt.show()


    def mask_diff_pic(self):
        """ WT vs. mutant up masks comparison with overlay (red - WT, green - mutant).
        
        """
        cell_contour = measure.find_contours(self.proc_mask, level=0.5)

        stim_num = self.wt_up_mask_list.shape[0]
        fig = plt.figure(figsize= (20, 20))
        for i in range(stim_num):
            wt_mask = np.asarray(self.wt_up_mask_list[i], dtype=int) 
            mut_mask = np.asarray(self.mut_up_mask_list[i], dtype=int)
            sum_mask = wt_mask + mut_mask
            tot_mask = sum_mask != 0

            # up_mask_diff = plot.toRGB(r_img=wt_mask,
            #                           g_img=mut_mask,
            #                           b_img=np.zeros_like(wt_mask))
            ax0 = plt.subplot(1, stim_num, i+1)
            ax0.imshow(np.mean(self.mut_img, axis=0)*-1, cmap='Greys')
            ax0.imshow(ma.masked_where(~tot_mask, sum_mask)*-1, cmap=plot.CMaps().cmap_red_green)
            ax0.text(100, 100, f'WT {int(np.max(self.wt_up_label_list[i]))} ROIs\nMut. {int(np.max(self.mut_up_label_list[i]))} ROIs',
                     fontsize=20, color='white')
            for ce_c in cell_contour:
                ax0.plot(ce_c[:, 1], ce_c[:, 0], linewidth=2, color='w')
            ax0.axis('off')

        plt.tight_layout()
        plt.show()