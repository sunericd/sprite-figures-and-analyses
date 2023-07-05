import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import anndata as ad
import seaborn as sns

from main import *




def get_correlation_between_error_corr_and_expression_corr (adata, keys=None):
    '''
    Takes in adata (AnnData) and returns correlation between pairwise error correlation and gene expression correlation
    
    keys = list of keys in adata.obsm to report correlations for; if None will use all of them with "predicted_expression"
    '''
    expression_mat = adata.X
    cor_exp = np.corrcoef(expression_mat.T)

    r_list = []
    
    if keys == None:
        keys = [x for x in list(adata.obsm.keys()) if "predicted_expression" in x]

    for model in keys:
                
        error_mat = adata.obsm[model]
        cor_err = np.corrcoef(error_mat.T)
        
        r,p=pearsonr(cor_exp.flatten(), cor_err.flatten())
        r_list.append(r)
    
    return(r_list)
    
    
    
def get_metrics (adata, keys=None, presmooth=False):
    '''
    Computes various performance metrics such as PCC, MAE, etc and separates them by the postprocessing method
    
    adata is output of the SPRITE method
    keys = list of keys in adata.obsm to use; if None will use all of them with "predicted_expression"
    presmooth = True or False (which values to take; defaults to False)
    '''
    if presmooth is False:
        presmooth = "_raw"
    else:
        presmooth = "_presmooth"
        
    if keys == None:
        keys = [x for x in list(adata.obsm.keys()) if "predicted_expression" in x]
    
    rho_list = []
    r_list = []
    mae_list = []
    pp_method = []
    model_base = []
    genes = []

    for model in keys:
        
        if presmooth != "_"+model.split("_")[-1]:
            continue
        
        for i in range(adata.X.shape[1]):
            gene = adata.var_names[i]
            if gene in adata.obsm[model].columns:
                genes.append(gene)
                truth = adata.X[:,i]
                predicted = adata.obsm[model][gene].values
                nas = np.logical_or(np.isnan(truth), np.isnan(predicted))
                try:
                    r, p = pearsonr(truth[~nas], predicted[~nas])
                    rho, p = spearmanr(truth[~nas], predicted[~nas])
                    mae = np.nanmean(np.abs(truth-predicted))
                except:
                    r = np.nan
                    rho = np.nan
                    mae = np.nan
                r_list.append(r)
                rho_list.append(rho)
                mae_list.append(mae)
                
                # determine model
                if "spage" in model:
                    model_base.append("SpaGE")
                elif "tangram" in model:
                    model_base.append("Tangram")
                elif "knn" in model:
                    model_base.append("Harmony-kNN")
                
                # determine SPRITE config
                if "filtered" in model:
                    if "reinforced" not in model:
                        pp_method.append("S only (filtered)")
                    else:
                        if ("reinforced_joint_pca_E" in model):
                            pp_method.append("R.E + S (filtered)")
                        elif ("reinforced_joint_pca_G" in model):
                            pp_method.append("R.G + S (filtered)")
                        elif ("reinforced_joint_pca_rna" in model):
                            pp_method.append("R.rna + S (filtered)")
                        elif ("reinforced_joint" in model):
                            pp_method.append("R + S (filtered)")
                        elif ("reinforced_gene_joint" in model):
                            pp_method.append("R.gene + S (filtered)")
                        else:
                            raise Exception(model+" not recognized")
                elif "smoothed" in model:
                    if "reinforced" not in model:
                        pp_method.append("S only")
                    else:
                        if ("reinforced_joint_pca_E" in model):
                            pp_method.append("R.E + S")
                        elif ("reinforced_joint_pca_G" in model):
                            pp_method.append("R.G + S")
                        elif ("reinforced_joint_pca_rna" in model):
                            pp_method.append("R.rna + S")
                        elif ("reinforced_joint" in model):
                            pp_method.append("R + S")
                        elif ("reinforced_gene_joint" in model):
                            pp_method.append("R.gene + S")
                        else:
                            raise Exception(model+" not recognized")
                elif "reinforced" in model:
                    if ("reinforced_joint_pca_E" in model):
                        pp_method.append("R.E only")
                    elif ("reinforced_joint_pca_G" in model):
                        pp_method.append("R.G only")
                    elif ("reinforced_joint_pca_rna" in model):
                        pp_method.append("R.rna only")
                    elif ("reinforced_joint" in model):
                            pp_method.append("R only")
                    elif ("reinforced_gene_joint" in model):
                            pp_method.append("R.gene only")
                    else:
                        raise Exception(model+" not recognized")
                else:
                    pp_method.append("Baseline")
    
    # Make dataframe
    df = pd.DataFrame([rho_list, r_list, mae_list, pp_method, model_base, genes]).transpose()
    df.columns = ["SCC", "PCC", "MAE", "Postprocessing", "Model", "Gene"]
    df = df.astype({"SCC":"float", "PCC":"float", "MAE":"float", "Postprocessing":"str", "Model":"str", "Gene":"str"})
    
    # Sanity check
    #print(np.unique(df["Postprocessing"], return_counts=True))
    
    return(df)
    
    
def get_performance_over_baseline (df, models_list=None, pp_methods_list=None, use_log_mae=False):
    '''
    Computes performance over baseline for the output dataframe of get_metrics()
        Subsetted for models in models_list and postprocessing methods in pp_methods_list
        For either of these arguments, if it is None (default), it will use all possible values from df
    '''
    if use_log_mae is True:
        mae_key = "log MAE"
    else:
        mae_key = "MAE"
    
    models = []
    pp_methods = []
    method_Rs = []
    baseline_Rs = []
    method_Ss = []
    baseline_Ss = []
    method_MAEs = []
    baseline_MAEs = []
    genes = []
    
    for model in models_list:
        for pp_method in pp_methods_list:
            for gene in np.unique(df["Gene"]):
            
                models.append(model)
                pp_methods.append(pp_method)
                
                if df[(df["Model"]==model)&(df["Postprocessing"]==pp_method)&(df["Gene"]==gene)]["PCC"].shape[0] > 1:
                    raise Exception ("Filtering not unique")
                
                method_R = df[(df["Model"]==model)&(df["Postprocessing"]==pp_method)&(df["Gene"]==gene)]["PCC"].values[0]
                baseline_R = df[(df["Model"]==model)&(df["Postprocessing"]=="Baseline")&(df["Gene"]==gene)]["PCC"].values[0]
                method_Rs.append(method_R)
                baseline_Rs.append(baseline_R)
                
                method_S = df[(df["Model"]==model)&(df["Postprocessing"]==pp_method)&(df["Gene"]==gene)]["SCC"].values[0]
                baseline_S = df[(df["Model"]==model)&(df["Postprocessing"]=="Baseline")&(df["Gene"]==gene)]["SCC"].values[0]
                method_Ss.append(method_S)
                baseline_Ss.append(baseline_S)
                
                method_MAE = df[(df["Model"]==model)&(df["Postprocessing"]==pp_method)&(df["Gene"]==gene)][mae_key].values[0]
                baseline_MAE = df[(df["Model"]==model)&(df["Postprocessing"]=="Baseline")&(df["Gene"]==gene)][mae_key].values[0]
                method_MAEs.append(method_MAE)
                baseline_MAEs.append(baseline_MAE)
                genes.append(gene)
                
    df_paired = pd.DataFrame([genes, models, pp_methods, method_Ss, baseline_Ss, method_Rs, baseline_Rs, method_MAEs, baseline_MAEs]).transpose()
    df_paired.columns = ["Gene", "Model", "Postprocessing", "Method SCC", "Baseline SCC", "Method PCC", "Baseline PCC", "Method MAE", "Baseline MAE"]
    df_paired = df_paired.astype({"Gene":"str", "Model":"str", "Postprocessing":"str", "Baseline SCC":"float", "Method SCC":"float", "Baseline PCC":"float", "Method PCC":"float", "Baseline MAE":"float", "Method MAE":"float"})
    df_paired["Improvement in SCC"] = df_paired["Method SCC"].values - df_paired["Baseline SCC"].values
    df_paired["Improvement in PCC"] = df_paired["Method PCC"].values - df_paired["Baseline PCC"].values
    df_paired["Improvement in MAE"] = df_paired["Baseline MAE"].values - df_paired["Method MAE"].values
    
    return (df_paired)
    
    
def compute_gene_properties (df_paired, dataset_name):
    '''
    Takes output of get_performance_over_baseline() and computes various properties of genes
    
    Uses the scRNAseq data associated with dataset_name
    '''
    # read scRNAseq and process
    spatial_adata, RNAseq_adata = load_paired_datasets("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                                "DataUpload/"+dataset_name+"/Locations.txt",
                                                "DataUpload/"+dataset_name+"/scRNA_count.txt")
    del spatial_adata
    
    RNAseq_adata.var_names = [x.lower() for x in RNAseq_adata.var_names]
    preprocess_data(RNAseq_adata, standardize=False, normalize=True)
    
    # get properties
    mean_exp = []
    var_exp = []
    frac_zero = []

    for gene in df_paired['Gene']:
        try:
            gex = RNAseq_adata[:, RNAseq_adata.var_names == gene].X.toarray()
        except:
            gex = RNAseq_adata[:, RNAseq_adata.var_names == gene].X
        mean_exp.append(np.nanmean(gex))
        var_exp.append(np.nanvar(gex))
        frac_zero.append(np.sum(gex == 0)/len(gex))
        
    df_paired["Mean Expression"] = mean_exp
    df_paired["Variance Expression"] = var_exp
    df_paired["Sparsity"] = frac_zero
    
    return(df_paired)
    
    
def leiden_clustering(adata, pca=True, inplace=False, **kwargs):
    '''
    Performs Leiden clustering using settings in the PBMC3K tutorial from Scanpy:
    
    https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    
    Adds under key "leiden" in adata.obs
    '''
    if inplace is False:
        adata = adata.copy()
    if pca is True:
        adata.X[np.isnan(adata.X)] = 0
        adata.X[adata.X < 0] = 0
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack')
    else:
        adata.obsm['X_pca'] = adata.X
    sc.pp.neighbors(adata)#, n_neighbors=10, n_pcs=15)
    sc.tl.leiden(adata, **kwargs)
    
    return (adata.obs['leiden'].copy(), adata.obsm['X_pca'].copy())


def get_pseudotime_trajectory (adata, method="dpt"):
    '''
    
    '''


    return (pseudotimes, terminal_states)


def run_umap(adata):
    '''
    Performs UMAP using settings from Scanpy:
    
    https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    
    Adds under key "umap" in adata.obs
    '''
    adata = adata.copy()
    adata.X[np.isnan(adata.X)] = 0
    adata.X[adata.X < 0] = 0
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    return (adata.obsm['X_umap'].copy(), adata.obsm['X_pca'].copy())
    
    
    
def pca_correlation(pcs1, pcs2):
    '''
    Computes Pearson correlation between concatenated/flattened pcs1 and pcs2
    
    Effectively equal to variance-weighted average of the column-wise correlations
    '''
    corr, p = pearsonr(pcs1.flatten(), pcs2.flatten())
    
    return(corr)