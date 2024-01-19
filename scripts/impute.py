import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA, NMF
import anndata as ad
import warnings
import time

from main import *

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of dataset folder in DataUpload/")
parser.add_argument("n_cv_folds", help="number of CV folds to perform", type=int)
parser.add_argument("n_inner_folds", help="either 'all' or an integer for n_folds in predict_gene_expression()", type=str)
parser.add_argument('--save_intermediate', action='store_true')
parser.add_argument('--dont_save_intermediate', dest='save_intermediate', action='store_false')
parser.set_defaults(save_intermediate=True)
args = parser.parse_args()

dataset_name = args.dataset
ncvfolds = args.n_cv_folds
n_folds = args.n_inner_folds
if n_folds == "all":
    n_folds = None
else:
    n_folds = int(n_folds)
save_intermediate = args.save_intermediate
    
methods = ["knn", "spage", "tangram"]
alphas_list = np.linspace(0.01, 0.9, 10)
savedir = "SPRITE_001_09_10"

# read data
if os.path.isfile("DataUpload/"+dataset_name+"/Metadata.txt"):
    adata, RNAseq_adata = load_paired_datasets("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                                "DataUpload/"+dataset_name+"/Locations.txt",
                                                "DataUpload/"+dataset_name+"/scRNA_count.txt",
                                                spatial_metadata = "DataUpload/"+dataset_name+"/Metadata.txt")
else:
    adata, RNAseq_adata = load_paired_datasets("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                                "DataUpload/"+dataset_name+"/Locations.txt",
                                                "DataUpload/"+dataset_name+"/scRNA_count.txt")
adata.var_names = [x.lower() for x in adata.var_names]
RNAseq_adata.var_names = [x.lower() for x in RNAseq_adata.var_names]

# preprocess RNAseq data
preprocess_data(RNAseq_adata, standardize=False, normalize=True)

# subset spatial data into shared genes
gene_names = np.intersect1d(adata.var_names, RNAseq_adata.var_names)
adata = adata[:, gene_names]

# folds for CV
np.random.seed(444)
np.random.shuffle(gene_names)
folds = np.array_split(gene_names, ncvfolds)

# run-time results
method_col = []
graph_time_col = []
presmooth_time_col = []
predict_time_col = []
npredict_col = []
reinforce_time_col = []
smooth_time_col = []


adata_copy = adata.copy()

# set fold id variable
fold_ids = np.zeros(len(adata.var_names))
for i, fold in enumerate(folds):
    fold_ids[adata.var_names.isin(fold)] = i
adata.var["fold"] = fold_ids.copy()

# try different methods and make predictions
for method in methods:
    
    for presmooth in ["_raw"]: # "_presmooth"
    
        for i, fold in enumerate(folds):
        
            method_col.append(method)
            
            # subset folds
            sub_adata = adata_copy[:, ~adata_copy.var_names.isin(fold)].copy()
            target_genes = list(fold)
            
            # preprocess spatial data
            preprocess_data(sub_adata, standardize=False, normalize=False)
            
            # Spatial Graph
            start_time = time.time()
            build_spatial_graph(sub_adata, method="fixed_radius", n_neighbors=50)
            calc_adjacency_weights(sub_adata, method="cosine")#, confidence=method+"_combined_loo_expression")
            graph_time_col.append(time.time() - start_time)
            
            # Pre-smoothing
            if presmooth == "_presmooth":
                start_time = time.time()
                smooth(sub_adata, predicted=None)
                presmooth_time_col.append(time.time() - start_time)
                sub_adata.X = sub_adata.obsm["presmoothed_X"].values
            else:
                presmooth_time_col.append(np.nan)
            
            # Predict expression
            start_time = time.time()
            if method == "spage":
                if len(sub_adata.var_names) < 40:
                    n_pv = 20
                else:
                    #n_pv = round(len(sub_adata.var_names)/2)
                    n_pv = round(np.min([len(sub_adata.var_names), len(sub_adata.obs_names)])/2)
                
                predict_gene_expression (sub_adata, RNAseq_adata, target_genes,
                                         method=method, n_folds=n_folds, n_pv=n_pv)
            elif method == "knn":
                predict_gene_expression (sub_adata, RNAseq_adata, target_genes,
                                         method=method, n_folds=n_folds, n_neighbors=10)
            else:
                predict_gene_expression (sub_adata, RNAseq_adata, target_genes,
                                         method=method, n_folds=n_folds)
            predict_time_col.append(time.time() - start_time)
            npredict_col.append(1+len(sub_adata.var_names))
            
            ### save for testing purposes
            #sub_adata.write(dataset_name+"_testing_fold_"+".h5ad")
            
            # Reinforce
            start_time = time.time()
            reinforce_gene(sub_adata, predicted=method+"_predicted_expression", update_method='joint',
                           alpha=alphas_list, tol=1e-8, optimization_metric="MAE", savedir=os.path.join("R_alpha_"+dataset_name,"fold"+str(i)), cv=5)
            reinforce_time_col.append(time.time() - start_time)
            
            # Smooth
            start_time = time.time()
            smooth(sub_adata, predicted="reinforced_gene_joint_"+method+"_predicted_expression", alpha=alphas_list, tol=1e-8, update_method="joint", optimization_metric="Ensemble", savedir=os.path.join("S_alpha_"+dataset_name,"fold"+str(i)))
            smooth_time_col.append(time.time() - start_time)
            smooth(sub_adata, predicted=method+"_predicted_expression", alpha=alphas_list, tol=1e-8, update_method="joint", optimization_metric="Ensemble")
            
            
            # Add new predictions
            if i == 0:
                for obs_name in [method+"_predicted_expression",
                                 "reinforced_gene_joint_"+method+"_predicted_expression",
                                 "reinforced_joint_"+method+"_predicted_expression",
                                 "reinforced_joint_pca_E_"+method+"_predicted_expression",
                                 "reinforced_joint_pca_G_"+method+"_predicted_expression",
                                 "reinforced_joint_pca_rna_"+method+"_predicted_expression",
                                 "smoothed_reinforced_gene_joint_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_pca_E_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_pca_G_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_pca_rna_"+method+"_predicted_expression",
                                 "smoothed_"+method+"_predicted_expression"]:
                    if obs_name in sub_adata.obsm.keys():
                        adata.obsm[obs_name+presmooth] = sub_adata.obsm[obs_name][fold].copy()
            else:
                for obs_name in [method+"_predicted_expression",
                                 "reinforced_gene_joint_"+method+"_predicted_expression",
                                 "reinforced_joint_"+method+"_predicted_expression",
                                 "reinforced_joint_pca_E_"+method+"_predicted_expression",
                                 "reinforced_joint_pca_G_"+method+"_predicted_expression",
                                 "reinforced_joint_pca_rna_"+method+"_predicted_expression",
                                 "smoothed_reinforced_gene_joint_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_pca_E_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_pca_G_"+method+"_predicted_expression",
                                 "smoothed_reinforced_joint_pca_rna_"+method+"_predicted_expression",
                                 "smoothed_"+method+"_predicted_expression"]:
                    if obs_name in sub_adata.obsm.keys():
                        adata.obsm[obs_name+presmooth][fold] = sub_adata.obsm[obs_name][fold].copy().values
                        
                        
            # save adata within fold
            if save_intermediate is True:
                if not os.path.exists(savedir+"/"+dataset_name+"_intermediate/"):
                    os.makedirs(savedir+"/"+dataset_name+"_intermediate/")
                large_save(sub_adata, savedir+"/"+dataset_name+"_intermediate/"+method+"/"+"fold"+str(i))
                if i == 0: # save folds for downstream work
                    np.save(savedir+"/"+dataset_name+"_intermediate/"+method+"/folds.npy", folds)

# save results in anndata
preprocess_data(adata, standardize=False, normalize=False) # to keep consistent with predictions

# if error loading (i.e. metadata too large), then large_save instead
try:
    adata.write(savedir+"/"+dataset_name+"_"+"_".join(methods)+".h5ad")
    adata2 = sc.read_h5ad(savedir+"/"+dataset_name+"_"+"_".join(methods)+".h5ad")
except:
    large_save(adata, savedir+"/"+dataset_name+"_"+"_".join(methods))
    os.remove(savedir+"/"+dataset_name+"_"+"_".join(methods)+".h5ad")


# save runtimes as dataframe
rt_df = pd.DataFrame([])
rt_df["method"] = method_col
rt_df["predict_time"] = predict_time_col
rt_df["number_of_predicts"] = npredict_col
rt_df["graph_time"] = graph_time_col
rt_df["reinforce_time"] = reinforce_time_col
rt_df["smooth_time_col"] = smooth_time_col
rt_df.to_csv(savedir+"/"+"runtimes_"+dataset_name+"_"+"_".join(methods)+".csv", index=False)