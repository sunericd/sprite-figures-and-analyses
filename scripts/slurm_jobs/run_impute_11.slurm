#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J impute11 # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 3600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --mem=64G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o impute11_%A.out # Standard out goes to this file 
#SBATCH -e impute11_%A.err # Standard err goes to this filehostname 

source activate geneimputation
python impute.py Dataset11 10 10 --save_intermediate
source deactivate