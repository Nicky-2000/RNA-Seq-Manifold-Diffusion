# First, inspect obs to find how controls are labeled
python - << 'EOF'
import scanpy as sc
ad = sc.read_h5ad("data/raw/K562_gwps/K562_gwps_raw_singlecell_01.h5ad", backed="r")
print("obs columns:", list(ad.obs.columns))

for col in ["gene", "gene_id", "transcript", "sgID_AB"]:
    if col in ad.obs:
        print("\n====", col, "====")
        s = ad.obs[col]
        print("n_unique:", s.nunique())
        print(s.value_counts().head(20))