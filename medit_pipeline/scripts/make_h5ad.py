import pandas as pd
from scipy.io import mmread
import anndata as ad

DATA_DIR = "data/raw/weinreb_stateFate_inVitro"

# 1) Expression matrix (cells x genes)
X = mmread(f"{DATA_DIR}/GSM4185642_stateFate_inVitro_normed_counts.mtx.gz")
X = X.tocsr()  # convert to CSR for scanpy

# 2) Gene names (columns)
gene_names = pd.read_csv(
    f"{DATA_DIR}/GSM4185642_stateFate_inVitro_gene_names.txt.gz",
    header=None,
    sep="\t"
)[0].astype(str).values

# 3) Cell barcodes (rows)
cell_barcodes = pd.read_csv(
    f"{DATA_DIR}/GSM4185642_stateFate_inVitro_cell_barcodes.txt.gz",
    header=None,
    sep="\t"
)[0].astype(str).values

# 4) Metadata (one row per cell, same order as matrix rows)
metadata = pd.read_csv(
    f"{DATA_DIR}/GSM4185642_stateFate_inVitro_metadata.txt.gz",
    sep="\t"
)

# Sanity checks
assert X.shape[0] == len(cell_barcodes) == metadata.shape[0], "cell dimension mismatch"
assert X.shape[1] == len(gene_names), "gene dimension mismatch"

# 5) Build AnnData
adata = ad.AnnData(X=X)
adata.obs_names = cell_barcodes
adata.var_names = gene_names

# attach metadata to obs, aligned by row order
metadata.index = adata.obs_names
adata.obs = metadata

# 6) Optional: clone membership matrix
clone_mtx = mmread(f"{DATA_DIR}/GSM4185642_stateFate_inVitro_clone_matrix.mtx.gz").tocsr()
adata.obsm["X_clone_membership"] = clone_mtx

# 7) Save as .h5ad
adata.write_h5ad(f"{DATA_DIR}/stateFate_inVitro_normed_counts.h5ad", compression="gzip")
