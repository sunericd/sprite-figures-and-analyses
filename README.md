# SPRITE Figures and Analyses

Intermediate results for downstream analysis are generated using the ```impute.py``` script and take the form of AnnData objects saved in h5ad format.

This repository contains all Jupyter notebooks and Python scripts for generating data and figures associated with the SPRITE manuscript. If you use this code or find it useful, we would appreciate it if you cite the relevant publication:

"SPRITE: improving spatial gene expression imputation with gene and cell networks" (preprint to be updated soon)


To set up a conda environment for these analyses, we recommend installing all dependencies in a new conda environment and then setting that new environment as a jupyter kernel for use in the notebooks. The conda environment is outlined in ```environment.yml```.

Jupyter notebooks containing code for making figures and running experiments can be found in ```notebooks```. They are generally listed in order of chronology within the SPRITE manuscript. In detail, these notebooks are:
- ```1_sprite_performance.ipynb``` - evaluation of SPRITE performance/improvement over baseline and the effect of ablating Reinforce/Smooth corresponding to Figures 2 and 3
- ```2_downstream_clustering.ipynb``` - experiments with clustering quality corresponding to Figure 4B
- ```3_downstream_visualization.ipynb``` - experiments with visualization quality corresponding to Figure 4C
- ```4_downstream_visualization_dynamicviz.ipynb``` - experiments with visualization quality corresponding to Figure 4D (relies on output from the previous notebook)
- ```5_predictive_models.ipynb``` - experiments with cell type classifiers corresponding to Figure 5

To run the notebooks, you should copy over the `main.py` and `downstream.py` Python scripts from the ```scripts/``` folder over to the ```notebooks/``` folder as some of the notebooks import functions from these scripts.

Python scripts for generating intermediate outputs from SPRITE for use in the notebook analyses can be found in ```scripts```. In detail, these scripts are:
- ```main.py``` - includes all methods associated with the main SPRITE meta-algorithm and additional utilities (in the future, this will split into multiple scripts for modularity)
- ```impute.py``` - script that generates intermediate data files by building baseline predictions and applying SPRITE
- ```downstream.py``` - includes various functions for downstream tasks that are used in evaluating SPRITE performance


All batch jobs scripts for generating intermediate date files can found in ```scripts/slurm_jobs``` and provide an (over-)estimate of the computational resources required for each job (e.g. memory, time, number of CPU/GPUs).