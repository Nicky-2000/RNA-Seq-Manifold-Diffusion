# Inspect contents of the Weinreb .h5ad to update QC ingestion
python - << 'EOF'
import numpy as np
import scanpy as sc

path = "data/raw/stateFate_inVitro_normed_counts.h5ad"
print(f"Loading {path}")
ad = sc.read_h5ad(path, backed="r")

print("\n=== AnnData overview ===")
print(ad)
print("shape (n_cells, n_genes):", ad.shape)
print("X class:", type(ad.X))

# ---------- OBS (cell metadata) ----------
print("\n=== OBS (cell metadata) ===")
print("obs columns:", list(ad.obs.columns))
print("\nobs.head():")
print(ad.obs.head())

print("\n[obs summary by column]")
for col in ad.obs.columns:
    s = ad.obs[col]
    nunique = s.nunique()
    print(f"\n---- {col} ----")
    print("dtype:", s.dtype)
    print("n_unique:", nunique)
    if nunique <= 20:
        # categorical-ish: show value counts
        print("value_counts():")
        print(s.value_counts().head(20))
    else:
        # continuous / high-cardinality: show basic stats or examples
        if np.issubdtype(s.dtype, np.number):
            print(
                "min/mean/max:",
                float(s.min()),
                float(s.mean()),
                float(s.max()),
            )
        else:
            print("example values:", s.iloc[:10].tolist())

# Highlight likely-important columns for QC / modeling if present
interesting_obs = [
    "timepoint", "day", "clone_id", "clone", "lineage",
    "condition", "sample", "batch", "donor"
]
print("\n=== Selected interesting obs columns (if present) ===")
for col in interesting_obs:
    if col in ad.obs:
        print(f"\n---- {col} ----")
        s = ad.obs[col]
        print("n_unique:", s.nunique())
        print(s.value_counts().head(20))

# ---------- VAR (gene metadata) ----------
print("\n=== VAR (gene metadata) ===")
print("var columns:", list(ad.var.columns))
print("\nvar.head():")
print(ad.var.head())
print("\nvar_names (first 10):")
print(ad.var_names[:10].tolist())

EOF
