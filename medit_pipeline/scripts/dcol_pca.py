import numpy as np
from scipy.sparse.linalg import eigsh

# Optional plotting / clustering helpers (for plot_result)
# Comment these out if you don't need them.
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path

def _standardize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Center and scale X along the given axis (like R's scale()).
    axis=0 => column-wise standardization.
    """
    X = np.asarray(X, float)
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


###########################################################
##### Utilities used in both DCOL-PCA & DCOL-CCA ##########
###########################################################

def scol_matrix_order(a: np.ndarray, x: np.ndarray) -> np.ndarray | float:
    """
    Python version of scol.matrix.order(a, x).

    x: 1D array used to order samples.
    a: either a vector of length n_samples or a matrix with shape (n_rows, n_samples).
       Returns:
         - scalar if a is a vector / single-row
         - 1D array of length n_rows if a is 2D (row-wise DCOL distances).
    """
    a = np.asarray(a)
    x = np.asarray(x)
    order = np.argsort(x)

    # a is effectively a vector (R's "is.null(nrow(a)) | nrow(a) == 1")
    if a.ndim == 1 or a.shape[0] == 1:
        a_vec = a.ravel()[order]
        d = np.diff(a_vec)
        dd = np.sum(d ** 2)
        return float(dd)

    # otherwise: matrix case, rows = features, cols = samples
    a_sorted = a[:, order]
    d = np.diff(a_sorted, axis=1)
    dd = np.sum(d ** 2, axis=1)  # rowSums
    return dd


def find_dcol(a: np.ndarray, b: np.ndarray, n_nodes: int = 1) -> np.ndarray:
    """
    Python version of findDCOL(a, b, nNodes).

    a, b: 2D arrays with shape (n_rows, n_samples).
          Rows are features, columns are samples.
    n_nodes: kept for API parity; current implementation is sequential.

    Returns
    -------
    dcol : np.ndarray, shape (nrow(a), nrow(b))
        Symmetric DCOL distance matrix when a and b refer to the same set.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # vector vs vector case
    if a.ndim == 1 or a.shape[0] == 1:
        # a, b are treated as vectors
        return np.array(scol_matrix_order(a, b), ndmin=1)

    n_a = a.shape[0]
    n_b = b.shape[0]

    # NOTE: for simplicity, this is sequential. You can parallelize these loops
    # with multiprocessing / joblib if needed.
    dcolab = np.zeros((n_a, n_b), dtype=float)
    dcolba = np.zeros((n_a, n_b), dtype=float)

    # dcolab[i_column] = scol_matrix_order(a, b[i, ])
    for i in range(n_b):
        dcolab[:, i] = scol_matrix_order(a, b[i, :])

    # dcolba[j_row] = scol_matrix_order(b, a[j, ])
    for j in range(n_a):
        dcolba[j, :] = scol_matrix_order(b, a[j, :])

    # retain the smaller entry to enforce symmetry
    dcol = np.minimum(dcolab, dcolba)
    return dcol


def get_cov(dcol_matrix: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray | float:
    """
    Python version of getCov(DCOLMatrix, X, Y).

    X, Y: data matrices with shape (n_samples, n_features), rows = samples.
    dcol_matrix:
      - scalar / length-1 => vector case (single pair).
      - 2D matrix (p x p) => DCOL distances between features (columns of Y).

    Returns
    -------
    - scalar in the vector case
    - CovMatrix (DCOL-correlation matrix), same shape as dcol_matrix in matrix case.
    """
    dcol = np.asarray(dcol_matrix, float)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    # Vector / scalar case
    if dcol.ndim == 0 or (dcol.ndim == 1 and dcol.shape[0] == 1):
        X = X.ravel()
        Y = Y.ravel()
        n = X.shape[0]
        var_Y = np.var(Y, ddof=1)
        if var_Y <= 0:
            # Degenerate case: no variance in Y, return 0 correlation
            return 0.0
        value = np.sqrt(max(0.0, 1.0 - dcol.item() / (2.0 * (n - 2.0) * var_Y)))
        return float(value)

    # Matrix case
    n = X.shape[0]
    var_list = np.var(Y, axis=0, ddof=1)  # sample variance over samples

    eps = 1e-12
    zero_var = var_list <= eps
    if np.any(zero_var):
        print(
            f"[get_cov] {zero_var.sum()} zero-variance features; stabilizing",
            flush=True,
        )

    scale = np.zeros_like(var_list)
    ok = ~zero_var
    scale[ok] = 1.0 / (2.0 * (n - 2.0) * var_list[ok])

    # eigenMapMatMult(DCOLMatrix, diag(scale)) == column-wise scaling by 'scale'
    cov_matrix = 1.0 - dcol * scale  # broadcast scale across rows
    cov_matrix[cov_matrix < 0] = 0.0  # clamp negs to 0 for num stability

    # For zero-var features, zero out row/cl and set diag to 1
    if np.any(zero_var):
        cov_matrix[:, zero_var] = 0.0
        cov_matrix[zero_var, :] = 0.0
        idx = np.where(zero_var)[0]
        cov_matrix[idx, idx] = 1.0

    cov_matrix = np.sqrt(cov_matrix)
    return cov_matrix


###########################################################
##### DCOL-PCA (feature-based and cell-based versions) ####
###########################################################

def dcol_pca0(
    X: np.ndarray,
    image: int = 0,
    k: int = 4,
    labels=None,
    Scale: bool = True,
    nNodes: int = 1,
    nPC_max: int = 100,
) -> dict[str, Any]:
    """
    Python version of Dcol_PCA0(X, ...).
    PCA with n = cells and k = principal k features
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Columns are features (genes), rows are samples (cells).
    image : int
        If 1, you can add plotting code here (not implemented by default).
    k : int
        Number of dimensions to keep for 'data.r' (visualization).
    labels : array-like, optional
        Group labels for plotting (unused unless you add plotting).
    Scale : bool
        Whether to standardize features before computing DCOL.
    nNodes : int
        Kept for API compatibility; current implementation is sequential.
    nPC_max : int
        Maximum number of principal components to compute.

    Returns
    -------
    dict with keys:
      - 'cov_D'      : DCOL-based correlation matrix (p x p)
      - 'vecs'   : eigenvectors of cov_D (p x nPC)
      - 'vals'    : eigenvalues (nPC,)
      - 'data_r'     : embedding (n_samples x min(k, nPC))
      - 'X_proj'     : full projection (n_samples x nPC)
    """
    X = np.asarray(X, float)
    X_o = X.copy()  # used for final projection

    if Scale:
        X = _standardize(X, axis=0)  # column-wise (features)

    # DCOL matrix over features: findDCOL(t(X), t(X))
    DcolMatrix = find_dcol(X.T, X.T, n_nodes=nNodes)
    cov_D = get_cov(DcolMatrix, X, X)  # DCOL-correlation matrix

    # Suppose cov_D is the matrix passed to eigsh
    print("[dcol_pca] cov_D finite?", np.isfinite(cov_D).all(), flush=True)
    print("[dcol_pca] cov_D min/max:", np.nanmin(cov_D), np.nanmax(cov_D), flush=True)

    # Eigen-decomposition (like RSpectra::eigs_sym on symmetric cov_D)
    p = cov_D.shape[0]
    nPC = min(nPC_max, p)

    # Enforce symmetry and remove NaN/Inf just in case
    cov_D = 0.5 * (cov_D + cov_D.T)
    cov_D = np.nan_to_num(cov_D, nan=0.0, posinf=0.0, neginf=0.0)

    # Add tiny diagonal jitter for numerical stablility
    cov_D.flat[:: p + 1] += 1e-8

    if p <= nPC + 10:
        # small matrix: use dense eigh
        vals, vecs = np.linalg.eigh(cov_D)
        idx = np.argsort(vals)[::-1][:nPC]
        vals = vals[idx]
        vecs = vecs[:, idx]
    else:
        # large matrix: sparse eigensolver
        vals, vecs = eigsh(cov_D, k=nPC, which="LM")
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

    # Project original (unscaled) X onto eigenvectors
    X_proj = X_o @ vecs  # (n_samples x nPC)
    data_r = X_proj[:, : min(k, X_proj.shape[1])]

    # You can add plotting here if image == 1 (using matplotlib).

    return {
        "cov_D": cov_D,
        "vecs": vecs,
        "vals": vals,
        "data_r": data_r,
        "X_proj": X_proj,
    }


def dcol_pca(
    X: np.ndarray,
    image: int = 0,
    k: int = 4,
    labels=None,
    Scale: bool = True,
    nNodes: int = 1,
    nPC_max: int = 100,
):
    """
    Python version of the alternative Dcol_PCA(X, ...).

    This version operates more on cell-cell similarity. In the R code,
    it transposes X, scales across samples, and builds a DCOL matrix
    that ultimately yields a cell-level embedding.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Rows are samples/cells, columns are features.
    image, k, labels, Scale, nNodes, nPC_max : as above.

    Returns
    -------
    dict with keys:
      - 'cov_D'    : DCOL-based correlation / similarity between cells (n_samples x n_samples)
      - 'vecs' : eigenvectors (n_samples x nPC)
      - 'vals'  : eigenvalues (nPC,)
      - 'data_r'   : embedding (n_samples x min(k, nPC))
    """
    X = np.asarray(X, float)
    X_o = X.copy()
    X_t = X.T  # features x samples

    if Scale:
        # In the R version, scale() is applied to X after transposing,
        # so this standardizes each sample (column) across features.
        X_t = _standardize(X_t, axis=0)

    # DCOL over cells: findDCOL(t(X), t(X)) with the modified X_t
    DcolMatrix = find_dcol(X_t.T, X_t.T, n_nodes=nNodes)
    cov_D = get_cov(DcolMatrix, X_t, X_t)  # n_samples x n_samples

    n_cells = cov_D.shape[0]
    nPC = min(nPC_max, n_cells)

    if n_cells <= nPC + 10:
        vals, vecs = np.linalg.eigh(cov_D)
        idx = np.argsort(vals)[::-1][:nPC]
        vals = vals[idx]
        vecs = vecs[:, idx]
    else:
        vals, vecs = eigsh(cov_D, k=nPC, which="LM")
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]

    PCs = vecs
    data_r = PCs[:, : min(k, PCs.shape[1])]

    # Again, you can add plotting if image == 1.

    return {
        "cov_D": cov_D,
        "vecs": vecs,
        "vals": vals,
        "data_r": data_r,
    }


######### Visualize results (optional) ############

def plot_result(reduced_data: np.ndarray, group_info, k: int = 2):
    """
    Rough Python version of plot.result().

    reduced_data : array-like, shape (n_samples, d)
        Low-dimensional embedding (e.g. output of dcol_pca['data_r'] or X_proj).
    group_info : array-like
        True group labels.
    k : int
        Number of dimensions from reduced_data to use.

    Returns
    -------
    ARI (float). Also produces a scatter plot when k <= 2.
    """
    reduced_data = np.asarray(reduced_data, float)
    group_info = np.asarray(group_info)

    n_clusters = len(np.unique(group_info))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km.fit(reduced_data[:, :k])
    ari = round(adjusted_rand_score(group_info, km.labels_), 3)

    if k <= 2:
        plt.figure()
        plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=group_info,
            s=20,
            cmap="tab10",
        )
        plt.title(f"ARI = {ari}")
        plt.xlabel("Factor1")
        plt.ylabel("Factor2")
        plt.tight_layout()
    else:
        # For k > 2, you could expand this to pairplots if needed.
        plt.figure()
        plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=group_info,
            s=20,
            cmap="tab10",
        )
        plt.title(f"ARI (first 2 dims) = {ari}")
        plt.xlabel("Factor1")
        plt.ylabel("Factor2")
        plt.tight_layout()

    return ari


def plot_spectral(
    vals: np.ndarray, out_dir: Path, title_prefix: str = "DCOL-PCA"
) -> Path:
    """
    Make scree + cumulative variance plots for spectral decomp e.g., PCA, DCOL-PCA, etc eigenvalues.

    Parameters
    ----------
    vals : array-like
        1D array of spectra (eigenvalues) (typically res["Evalues"]),
        sorted in descending order.
    title_prefix : str
        Prefix for subplot titles (e.g. "DCOL-PCA", "PCA", etc.)
    """
    ev = np.asarray(vals, float)

    # Guard against weird inputs
    ev = ev[ev > 0]  # ignore non-positive eigenvalues if any
    if ev.size == 0:
        print("[plot_dcol_scree] No positive eigenvalues to plot.")
        return

    var_ratio = ev / ev.sum()
    cum_ratio = np.cumsum(var_ratio)
    k = np.arange(1, ev.size + 1)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)

    # Scree: raw eigenvalues
    axes[0].plot(k, ev, marker="o")
    axes[0].set_title(f"{title_prefix}: Eigenvalues (Scree)")
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Eigenvalue")

    # Cumulative variance
    axes[1].plot(k, cum_ratio, marker="o")
    axes[1].set_title(f"{title_prefix}: Cumulative 'variance' explained")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_ylim(0, 1.05)

    k_png = out_dir / f"{title_prefix}.png"
    plt.savefig(k_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return k_png
