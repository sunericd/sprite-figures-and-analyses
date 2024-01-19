#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J impute17 # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 3600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --mem=64G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o impute17_%A.out # Standard out goes to this file 
#SBATCH -e impute17_%A.err # Standard err goes to this filehostname 

source activate geneimputation
python impute.py Dataset17 10 10 --save_intermediate
source deactivate