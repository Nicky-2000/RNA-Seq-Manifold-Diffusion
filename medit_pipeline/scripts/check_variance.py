import numpy as np
import scanpy as sc
import yaml
from pathlib import Path
from EGGFM.eggfm import run_eggfm_dimred
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def compute_ari(X, labels, k):
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10)
    km.fit(X[:, :k])
    return adjusted_rand_score(labels, km.labels_)


def main():
    params = yaml.safe_load(Path("configs/params.yml").read_text())
    spec = params["spec"]
    k = spec.get("ari_n_dims", spec.get("n_pcs", 10))

    ad = sc.read_h5ad("./data/prep/qc.h5ad")
    # your hvg+norm pipeline
    from EGGFM.prep import prep_for_manifolds

    # base = prep_for_manifolds(ad, hvg_n_top_genes=2000)
    base = ad
    labels = base.obs[spec["ari_label_key"]].to_numpy()

    scores_eggfm = []
    scores_pca = []

    for run in range(10):
        print(f"=== Run {run+1}/10 ===")
        qc = base.copy()
        qc, _ = run_eggfm_dimred(qc, params)

        # PCA→DM
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_pca")
        sc.tl.diffmap(qc, n_comps=k)
        X_pca_dm = qc.obsm["X_diffmap"][:, :k]

        # EGGFM DM
        # X_eggfm = qc.obsm["X_eggfm"][:, :k]

        # scores_pca.append(compute_ari(X_pca_dm, labels, k))
        # scores_eggfm.append(compute_ari(X_eggfm, labels, k))

        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_eggfm")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_eggfm = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_eggfm"] = X_diff_eggfm

        scores_pca.append(compute_ari(X_pca_dm, labels, k))
        scores_eggfm.append(compute_ari(X_diff_eggfm, labels, k))

    print("\n=== Variance results ===")
    print(f"PCA→DM:   mean={np.mean(scores_pca):.4f}, std={np.std(scores_pca):.4f}")
    print(f"EGGFM DM: mean={np.mean(scores_eggfm):.4f}, std={np.std(scores_eggfm):.4f}")


if __name__ == "__main__":
    main()
