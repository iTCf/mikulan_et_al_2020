# Simultaneous human intracerebral stimulation and HD-EEG, ground-truth for source localization methods - Scripts and usage demonstration

This repository contains usage demonstration scripts and the code used for the preparation and validation of the Localize-MI dataset, published in :

[Mikulan, E. et al. Simultaneous human intracerebral stimulation and HD-EEG, ground-truth for source localization methods. Scientific Data 7, 1–8 (2020).](https://www.nature.com/articles/s41597-020-0467-x)

Please make sure to download the latest version of the dataset which can be found [here](https://gin.g-node.org/ezemikulan/Localize-MI)


_Running the demonstration_

In order to run the demo you will need the appropriate libraries. In order to install these modules, please refer to the [MNE-Python](https://martinos.org/mne/stable/install_mne_python.html) website and follow the instructions. In addition you will need a module that will allow to import our custom function to load the BIDS data. Please run _pip install ipynb_ on your terminal or command prompt once you have activated the environment. Once the environment is ready, open the file [_00_getting_started.ipynb](https://github.com/iTCf/mikulan_et_al_2020/blob/master/_00_getting_started.ipynb) (you could use [_jupyter lab_](https://jupyterlab.readthedocs.io/en/stable/) to do so), update it with the path of the folder where you downloaded the data and run the cells in order.



If you have any questions please open an issue.



Note: The scripts here provided are a subset of all the scripts used given that some contained sensitive private information that can not be made public. However the provided scripts contain all the relevant code used for the preparation and validation of the dataset.
