DoMB Tools
==========
## Python Toolkit of Department of Molecular Biophysics

![PyPI - Version](https://img.shields.io/pypi/v/domb)
![PyPI - License](https://img.shields.io/pypi/l/domb)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/domb)
![Website](https://img.shields.io/website?up_message=domb.bio&up_color=%23038C93&url=https%3A%2F%2Fdomb.bio%2F)

# Description
__Registration types__ integrate acquired data (images, electrophysiological recordings, metadata) into unified data structures. Attributes within this structure support subsequent analysis through functions in the _modules_.

__Modules__ provide distinct data analysis approaches with predefined pipelines. These pipelines transform inputs into approach-specific illustrations and organized pre-processed images, optimized for further quantitative analysis. Modules require specific _registration types_ as inputs.

__Utilities__ offer reusable functions for multidimensional image processing, advanced visualization, and uploading data in specific formats (OIF/OIB, HEKA, etc.).


# Structure Overview
```
└── domb
    ├── reg_type                  # data types for different registration designs
    │   └── wf_x2_m2.py             #  widefield, 2 excitation wavelengths, 2 emission channels
    |
    ├── red_green                 # translocation detection using differential image comparison
    │   ├── wt_vs_mut.py            # co-imaging of two NCSs with single stimuli, requires wf_x2_m2 as input
    │   └── wt_vs_mut_multistim.py  # co-imaging of two NCSs with multiple stimuli, requires wf_x2_m2 as input
    |
    ├── fret                      # Förster resonance energy transfer (FRET) estimation
    │   ├── e_fret                  # 3-cube approach for FRET efficiency estimation
    │   │   ├── coef_calc.py          # estimation of calibration coefficients
    │   │   └── e_app.py              # FRET efficiency calculation, requires wf_x2_m2 as input
    │   └── b_fret                # Bayesian inference implementation for 3-cube E-FRET approach
    |
    └── utils                     # utilities
        ├── masking.py              # functions for masking multi-dimensional images
        ├── plot.py                 # functions for various pretty plotting
        └── oiffile.py              # Olympus OIF/OIB files uploading

```

# Installation
Set up a new conda environment with Python 3.9:
```
conda create -y -n domb -c conda-forge python=3.9
conda acticate domb
```

### From pip
```
python -m pip install domb
```

### From GitHub
Clone the repo:
```
git clone -b master git@github.com:wisstock/domb-tools.git
```

To install the package, simply navigate to the repository folder and install with pip: 
```
cd DoMB_tools
python -m pip install .
```

But if you're planning to make changes and work on the source code actively, you might want to consider using the editable mode:
```
python -m pip install -e .
```

# Borrowed modules
### OIF File
_Copyright © 2012-2022 [Christoph Gohlke](https://www.cgohlke.com/)_

Oiffile is a Python library to read image and metadata from Olympus Image
Format files. OIF is the native file format of the Olympus FluoView(tm)
software for confocal microscopy.

There are two variants of the format:

- OIF (Olympus Image File) is a multi-file format that includes a main setting
  file (.oif) and an associated directory with data and setting files (.tif,
  .bmp, .txt, .pyt, .roi, and .lut).

- OIB (Olympus Image Binary) is a compound document file, storing OIF and
  associated files within a single file.

### B-FRET
_Copyright © 2022 [Emonet Lab](https://github.com/emonetlab), [Kamino et al.,2023](https://www.pnas.org/doi/10.1073/pnas.2211807120)_

__*This module is currently not implemented!*__

This package uses Bayesian inference to generate posterior distributions of FRET signals from noisy measured FRET data. B-FRET, generally applicable to standard 3-cube FRET-imaging data. Based on Bayesian filtering theory, B-FRET implements a statistically optimal way to infer molecular interactions and thus drastically improves the SNR. 