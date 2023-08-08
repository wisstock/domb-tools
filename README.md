DoMB tools
==========

__Python Toolkit of Department of Molecular Biophysics__




# Instalation
Clone the sources of this repo, create an environment with all the prereqs with:

`conda env create -n domb -f environment.yml`

Activate the environment, and then do:

`pip install -e .`

This installation mode is suitable for active development on the sources.
Or use just:

`pip install .`



# Structure
## FRET module. FRET efficiency ($E_{app}$) estimation with 3-cube method
Based on _[Zal and Gascoigne, 2004](https://pubmed.ncbi.nlm.nih.gov/15189889/)_


## RG module. Translocation registration with differential (red-green) images method
Based on _[Dovgan et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20704590/)_