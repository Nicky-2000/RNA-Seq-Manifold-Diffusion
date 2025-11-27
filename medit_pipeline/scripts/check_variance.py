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

    base = sc.read_h5ad(spec.get("ad_file"))
    labels = base.obs[spec["ari_label_key"]].to_numpy()

    scores_eggfm = []
    scores_eggfm_2 = []
    scores_eggfm_3 = []
    scores_eggfm_4 = []
    scores_eggfm_5 = []
    # scores_eggfm_6 = []
    scores_pca = []
    scores_pca_2 = []
    total = 20
    for run in range(total):
        print(f"=== Run {run+1}/{total} ===")
        qc = base.copy()
        qc, _ = run_eggfm_dimred(qc, params)

        # PCA → Diffmap
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_pca")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_pca = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_pca"] = X_diff_pca

        # PCA → Diffmap → Diffmap
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_diff_pca")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_pca_double = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_pca_x2"] = X_diff_pca_double

        # EGGFM
        X_eggfm = qc.obsm["X_eggfm"][:, :k]

        # EGGFM DM
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_eggfm")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_eggfm = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_eggfm"] = X_diff_eggfm

        # EGGFM DM DM
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_diff_eggfm")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_eggfm_x2 = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_eggfm_x2"] = X_diff_eggfm_x2

        # EGGFM DM DM DM
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_diff_eggfm_x2")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_eggm_x3 = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_eggm_x3"] = X_diff_eggm_x3

        # EGGFM DM DM DM DM
        sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_diff_eggm_x3")
        sc.tl.diffmap(qc, n_comps=k)
        X_diff_eggm_x4 = qc.obsm["X_diffmap"][:, :k]
        qc.obsm["X_diff_eggm_x4"] = X_diff_eggm_x4

        # # EGGFM DM DM DM DM DM
        # sc.pp.neighbors(qc, n_neighbors=30, use_rep="X_diff_eggm_x4")
        # sc.tl.diffmap(qc, n_comps=k)
        # X_diff_eggm_x5 = qc.obsm["X_diffmap"][:, :k]
        # qc.obsm["X_diff_eggm_x5"] = X_diff_eggm_x5

        scores_pca.append(compute_ari(X_diff_pca, labels, k))
        scores_pca_2.append(compute_ari(X_diff_pca_double, labels, k))
        scores_eggfm.append(compute_ari(X_eggfm, labels, k))
        scores_eggfm_2.append(compute_ari(X_diff_eggfm, labels, k))
        scores_eggfm_3.append(compute_ari(X_diff_eggfm_x2, labels, k))
        scores_eggfm_4.append(compute_ari(X_diff_eggm_x3, labels, k))
        scores_eggfm_5.append(compute_ari(X_diff_eggm_x4, labels, k))
        # scores_eggfm_6.append(compute_ari(X_diff_eggm_x5, labels, k))

    print("\n=== Variance results ===")
    print(f"PCA→DM:    mean={np.mean(scores_pca):.4f}, std={np.std(scores_pca):.4f}")
    print(
        f"PCA→DM2:   mean={np.mean(scores_pca_2):.4f}, std={np.std(scores_pca_2):.4f}"
    )
    print(
        f"EGGFM:     mean={np.mean(scores_eggfm):.4f}, std={np.std(scores_eggfm):.4f}"
    )
    print(
        f"EGGFM DM:  mean={np.mean(scores_eggfm_2):.4f}, std={np.std(scores_eggfm_2):.4f}"
    )
    print(
        f"EGGFM DM2: mean={np.mean(scores_eggfm_3):.4f}, std={np.std(scores_eggfm_3):.4f}"
    )
    print(
        f"EGGFM DM3: mean={np.mean(scores_eggfm_4):.4f}, std={np.std(scores_eggfm_4):.4f}"
    )
    print(
        f"EGGFM DM4: mean={np.mean(scores_eggfm_5):.4f}, std={np.std(scores_eggfm_5):.4f}"
    )
    # print(
    #     f"EGGFM DM5: mean={np.mean(scores_eggfm_6):.4f}, std={np.std(scores_eggfm_6):.4f}"
    # )


if __name__ == "__main__":
    main()
