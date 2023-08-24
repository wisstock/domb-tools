DoMB tools
==========

__Python Toolkit of Department of Molecular Biophysics__




# Instalation
Clone the sources of this repo, create a new environment:
```
conda env create -n domb
```

Activate the environment, and then do:
```
pip install -e .
```

This installation mode is suitable for active development on the sources.
Or use just:
```
pip install .
```



# Structure
## Overview

```
└── domb
    ├── reg_type           # data types for different registration designs
    │   └── wf_x2_m2.py       #  widefield, 2 excitation wavelengths, 2 emission channels
    |
    |
    ├── red_green          # translocation detection with differential images comparison
    │   └── wt_vs_mut.py      # co-registration of two NCSs, requires wf_x2_m2 as input
    |
    ├── fret               # FRET detection
    │   ├── coef_calc.py      # calibration coefficients estimation (3-cube approach)
    │   └── e_app.py          # FRET calculation (3-cube approach), requires wf_x2_m2 as input
    |
    └── util               # utilities
        ├── masking.py        # functions for multi-dimensional images masking
        └── plot.py           # functions for various pretty plotting

```


## Registration types
- ### Widefield 2 exicitation 2 emission (wf_x2_m2)



## Modules
### FRET module
Based on _[Zal and Gascoigne, 2004](https://pubmed.ncbi.nlm.nih.gov/15189889/)_
- __coef_calc__

  Calculation of 4 chennels crosstalc coeficients and G parameter

- __e_app__

  FRET efficiency ($E_{app}$) estimation with 3-cube approach

### RG module. Translocation registration with differential (red-green) images method
Based on _[Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/)_
- __wt_vs_mut__ (wild type vs. mutant)

  Detection of translocation of two modifications of the same NCS (e.g. HPCA WT and N75K) or two different NCSs (e.g. HPCA and NCALD)  