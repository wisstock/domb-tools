DoMB tools
==========

__Python Toolkit of Department of Molecular Biophysics__



# Instalation
Go ahead and clone the package repository. As a starting point, set up a fresh conda environment using this command:
```
conda env create -n domb
```

Once the environment is ready to go, activate it. To install the package, simply navigate to the repository folder and run: 
```
pip install .
```


But if you're planning to make changes and work on the source code actively, you might want to consider using the editable mode:
```
pip install -e .
```


# Description
__Registration types__ combine acquired data (images, electrophysiological recordings, metadata) into unified data structures. Attributes within this structure support subsequent analysis through _modules_ functions.

__Modules__ offer distinct data analysis approaches with predefined pipelines. These pipelines transform inputs into approach-specific illustrations and organized data frames, optimized for statistical analysis. Modules require specific _registration types_ as inputs.

__Utilities__ provide reusable functions for multidimensional image processing, advanced visualization, and specific data format (OIF/OIB, HEKA) uploading.



# Structure Overview
```
└── domb
    ├── reg_type           # data types for different registration designs
    │   └── wf_x2_m2.py       #  widefield, 2 excitation wavelengths, 2 emission channels
    |
    |
    ├── red_green          # translocation detection using differential image comparison
    │   └── wt_vs_mut.py      # co-registration of two NCSs, requires wf_x2_m2 as input
    |
    ├── fret               # FRET detection
    │   ├── coef_calc.py      # estimation of calibration coefficients (3-cube approach)
    │   └── e_app.py          # FRET calculation (3-cube approach), requires wf_x2_m2 as input
    |
    └── util               # utilities
        ├── masking.py        # functions for masking multi-dimensional images
        └── plot.py           # functions for various pretty plotting

```


# Registration types
- ### Widefield 2 exicitation 2 emission (wf_x2_m2)


# Modules
## FRET module
Based on _[Zal and Gascoigne, 2004](https://pubmed.ncbi.nlm.nih.gov/15189889/)_
- __coef_calc__

  Calculation of 4 channels crosstalk coefficients and G parameter

- __e_app__

  FRET efficiency ($E_{app}$) estimation with 3-cube approach

## RG module. Translocation registration with differential (red-green) images method
Based on _[Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/)_
- __wt_vs_mut__ (wild type vs. mutant)

  Detection of translocation of two modifications of the same NCS (e.g. HPCA WT and N75K) or two different NCSs (e.g. HPCA and NCALD)  