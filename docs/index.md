DoMB tools
==========

__Python Toolkit of Department of Molecular Biophysics__

# Structure Overview

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

# Registration types
::: domb.reg_type.wf_x2_m2


# Modules
## FRET
::: domb.fret.coef_calc
::: domb.fret.e_app

## RG
::: domb.red_green.wt_vs_mut.WTvsMut

## Utils
::: domb.utils.masking
::: domb.utils.plot