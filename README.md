DoMB tools
==========

__Python Toolkit of Department of Molecular Biophysics__




## Instalation
Clone the sources of this repo, create an environment with all the prereqs with:

`conda env create -n domb -f environment.yml`

Activate the environment, and then do:

`pip install -e .`

This installing mode is suitable for active development on the sources.



## Structure
### FRET module
__FRET efficiency (E app) estimation with 3-cube method__

Based on _[Zal and Gascoigne, 2004](https://pubmed.ncbi.nlm.nih.gov/15189889/)_

$$F_c = I_{DA} - a (I_{AA} - c I_{DD}) - d (I_{DD} - b I_{AA})$$
$$E_{app} = \frac{R}{R+G}, R = \frac{F_c}{I_{DD}}$$

### RG module
__Translocation registration with differential (red-green) images method__

Based on _[Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/)_