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

        __Requires WF_2x_2m instance type as input!__

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

        """
        self.wt_img = wt_img
        self.mut_img = mut_img
        self.proc_mask = proc_mask
        self.narrow_proc_mask = narrow_proc_mask