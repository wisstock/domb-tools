domb-napari
===========
## DoMB Tools for napari

[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/domb-napari)](https://napari-hub.org/plugins/domb-napari)
![PyPI - Version](https://img.shields.io/pypi/v/domb-napari)
![PyPI - License](https://img.shields.io/pypi/l/domb-napari)
![Website](https://img.shields.io/website?up_message=domb.bio&up_color=%23038C93&url=https%3A%2F%2Fdomb.bio%2F)

A napari plugin offers widgets to analyze fluorescence-labeled proteins redistribution in widefield epifluorescence time-lapse acquisitions. Useful for studying calcium-dependent translocation of neuronal calcium sensors, synaptic receptors traffic during long-term plasticity induction, membrane protein tracking, etc.

![](pic/translocation.gif)
__Hippocalcin (neuronal calcium sensor) redistributes in dendritic branches upon NMDA application__

## Widgets
### Image Preprocessing
Provides functions for preprocessing multi-channel fluorescence acquisitions:
- If the input image has 4 dimensions (time, channel, x-axis, y-axis), channels will be split into individual 3 dimensions images (time, x-axis, y-axis) with the `_ch%index%` suffix.
- If the `gaussian blur` option is selected, the image will be blurred with a Gaussian filter using sigma=`gaussian sigma`.
- If the `photobleaching correction` option is selected, the image will undergo correction with exponential (method `exp`) or bi-exponential (method `bi_exp`) fitting.

![](pic/pic_0.png)

### Red-Green Series
Primary method for detecting fluorescent-labeled targets redistribution in time. Returns a series of differential images representing the intensity difference between the current frame and the previous one as new image with the `_red-green` suffix.

Parameters:

- `left frames` - number of previous frames for pixel-wise averaging.
- `space frames` - number of frames between the last left and first right frames.
- `right frames` - number of subsequent frames for pixel-wise averaging.
- `save mask series` - if selected, a series of labels will be created for each frame of the differential image with the threshold `insertion threshold`.

![](pic/pic_1.png)

### Up Mask
Generates labels for insertion sites (regions with increasing intensity) based on `-red-green` images. Returns labels layer with `_up-labels` suffix.

Parameters:

- `detection img index` - index of the frame from `-red-green` image used for insertion sites detection.
- `insertion threshold` - threshold value for insertion site detection, intensity on selected `_red-green` frame normalized in -1 - 0 range.
- `save mask` - if selected, a total up mask (containing all ROIs) will be created with the `_up-mask` suffix.

![](pic/pic_2.png)

### Individual Labels Profiles
Builds a plot with mean intensity profiles for each ROI in `labels` using absolute intensity (if `raw intensity` is selected) or relative intensities (ΔF/F0).

The `time scale` sets the number of seconds between frames for x-axis scaling.

The baseline intensity for ΔF/F0 profiles is estimated as the mean intensity of the initial profile points (`ΔF win`).

Filters ROIs by minimum (`min amplitude`) and maximum (`max amplitude`) intensity amplitudes.

_Note: Intensity filtering is most relevant for ΔF/F0 profiles._

Additionally, you can save ROI intensity profiles as .csv using the `save data frame` option and specifying the `saving path`. The output data frames `%img_name%_lab_prof.csv` will contain the following columns:

- __id__ - unique image ID, the name of the input `napari.Image` object.
- __roi__ - ROI number, consecutively numbered starting from 1.
- __int__ - ROI mean intensity, raw or ΔF/F0 according to the `raw intensity` option.
- __time__ - frame time point according to the `time scale`.

_Note: The data frame will contain information for all ROIs; filtering options pertain to plotting only._

![](pic/pic_3.png)

### Labels Profile
Builds a plot with the averaged intensity of all ROIs in `labels`. Can take two images (`img 0` and `img 1`) as input if `two profiles` are selected.

The `time scale` and `ΔF win` are the same as in the __Individual Labels Profiles__.

The `stat method` provides methods for calculating intensity errors:

- `se` - standard error of mean.
- `iqr` - interquartile range.
- `ci` - 95% confidence interval for t-distribution.

![](pic/pic_4.png)