import numpy as np
import numpy as np
import plotly.graph_objects as go

def spiral_curve(t, a=0.5, b=0.2):
    """
    Compute points on an Archimedean spiral in 2D.

    Parameters
    ----------
    t : float or np.ndarray
        Parameter values (can be scalar or array).
    a : float
        Base radius term.
    b : float
        Linear growth rate of the radius.

    Returns
    -------
    X : np.ndarray, shape (len(t), 2)
        2D coordinates of the spiral at each t.
    """
    t = np.asarray(t)
    r = a + b * t
    
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    return np.stack([x, y], axis=-1)



import numpy as np
import plotly.graph_objects as go

def plot_spiral(X, title="Spiral Curve"):
    """
    Plot a 2D curve (e.g., spiral) using Plotly.

    Parameters
    ----------
    X : np.ndarray, shape (N, 2)
        Array of 2D points making up the spiral.
    title : str
        Title for the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.
    """
    X = np.asarray(X)
    assert X.ndim == 2 and X.shape[1] == 2, "X must be of shape (N, 2)"

    x = X[:, 0]
    y = X[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='curve'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleratio=1),
        template="plotly_white"
    )

    return fig


import numpy as np

def compute_true_tangent_spiral(t, a=0.5, b=0.2):
    """
    Compute the true unit tangent vectors for an Archimedean spiral in 2D.

    Spiral definition:
        r(t)   = a + b t
        x(t)   = r(t) * cos(t)
        y(t)   = r(t) * sin(t)
        gamma'(t) = [b cos(t) - r(t) sin(t),
                     b sin(t) + r(t) cos(t)]

    Parameters
    ----------
    t : float or np.ndarray
        Parameter values where we want the tangent.
    a : float
        Base radius term in r(t) = a + b t.
    b : float
        Growth rate of the radius in r(t) = a + b t.

    Returns
    -------
    T : np.ndarray, shape (..., 2)
        Unit tangent vectors at each t. If t is 1D of shape (N,),
        T will be of shape (N, 2).
    """
    t = np.asarray(t)
    r = a + b * t

    dx = b * np.cos(t) - r * np.sin(t)
    dy = b * np.sin(t) + r * np.cos(t)

    deriv = np.stack([dx, dy], axis=-1)  # shape (..., 2)
    norms = np.linalg.norm(deriv, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)  # avoid division by zero

    T = deriv / norms
    return T


import numpy as np
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def plot_spiral_with_tangents(
    X,
    T,
    N=None,
    indices=None,
    every_n=None,
    scale_tangent=0.5,
    scale_normal=0.5,
    title="Spiral with Tangents/Normals",
):
    """
    Plot the spiral and overlay tangent (and optionally normal) vectors at multiple points.

    Parameters
    ----------
    X : np.ndarray, shape (N, 2)
        Spiral points.
    T : np.ndarray, shape (N, 2)
        Unit tangent vectors at each point.
    N : np.ndarray, shape (N, 2), optional
        Unit normal vectors at each point. If provided, normals are plotted too.
    indices : list[int] or np.ndarray, optional
        Explicit indices where to draw vectors.
    every_n : int, optional
        If indices is None, draw vectors at every_n points (e.g. every 20th point).
    scale_tangent : float
        Length scaling factor for tangent line segments.
    scale_normal : float
        Length scaling factor for normal line segments.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.
    """
    X = np.asarray(X)
    T = np.asarray(T)
    assert X.shape == T.shape and X.shape[1] == 2, "X and T must both be (N, 2)"

    if N is not None:
        N = np.asarray(N)
        assert N.shape == X.shape, "N must have same shape as X if provided"

    N_points = X.shape[0]

    # Decide which indices to use
    if indices is None:
        if every_n is None:
            every_n = max(N_points // 20, 1)  # default: ~20 locations
        indices = np.arange(0, N_points, every_n)
    else:
        indices = np.asarray(indices)

    # Base curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='lines',
        name='Spiral',
    ))

    first_tangent = True
    first_normal = True

    for idx in indices:
        p = X[idx]
        t_vec = T[idx]

        # Tangent segment
        p_t_start = p - 0.5 * scale_tangent * t_vec
        p_t_end   = p + 0.5 * scale_tangent * t_vec

        fig.add_trace(go.Scatter(
            x=[p_t_start[0], p_t_end[0]],
            y=[p_t_start[1], p_t_end[1]],
            mode='lines',
            name='Tangent' if first_tangent else None,
            showlegend=first_tangent,
            line=dict(width=2),
        ))
        first_tangent = False

        # Normal segment (if provided)
        if N is not None:
            n_vec = N[idx]
            p_n_start = p - 0.5 * scale_normal * n_vec
            p_n_end   = p + 0.5 * scale_normal * n_vec

            fig.add_trace(go.Scatter(
                x=[p_n_start[0], p_n_end[0]],
                y=[p_n_start[1], p_n_end[1]],
                mode='lines',
                name='Normal' if first_normal else None,
                showlegend=first_normal,
                line=dict(width=2, dash='dot'),
            ))
            first_normal = False

        # Mark the point itself
        fig.add_trace(go.Scatter(
            x=[p[0]],
            y=[p[1]],
            mode='markers',
            showlegend=False,
            marker=dict(size=4, symbol='circle-open'),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleratio=1),
        template="plotly_white",
    )

    return fig

import numpy as np

def compute_true_normal_spiral(t, a=0.5, b=0.2):
    """
    Compute the true unit normal vectors for an Archimedean spiral in 2D.

    The normal is obtained by rotating the unit tangent by +90 degrees:
        if T = (Tx, Ty), then N = (-Ty, Tx)

    Parameters
    ----------
    t : float or np.ndarray
        Parameter values where we want the normal.
    a : float
        Base radius term in r(t) = a + b t.
    b : float
        Growth rate of the radius in r(t) = a + b t.

    Returns
    -------
    N : np.ndarray, shape (..., 2)
        Unit normal vectors at each t. If t is 1D of shape (N,),
        N will be of shape (N, 2).
    """
    # Get unit tangents first
    T = compute_true_tangent_spiral(t, a=a, b=b)  # shape (..., 2)

    Tx = T[..., 0]
    Ty = T[..., 1]

    # Rotate by +90 degrees: (-Ty, Tx)
    Nx = -Ty
    Ny =  Tx

    N = np.stack([Nx, Ny], axis=-1)

    # Just to be safe, re-normalize (should already be unit length)
    norms = np.linalg.norm(N, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    N = N / norms

    return N


import numpy as np

def compute_knn_neighbors(X, k):
    """
    Compute k nearest neighbors for each point in X (function-free).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Data points.
    k : int
        Number of nearest neighbors (excluding the point itself).

    Returns
    -------
    neighbors_idx : np.ndarray, shape (N, k)
        neighbors_idx[i] contains the indices of the k nearest neighbors of X[i].
    """
    X = np.asarray(X)
    N, d = X.shape

    # Pairwise squared distances
    # dists[i, j] = ||X[i] - X[j]||^2
    diff = X[:, None, :] - X[None, :, :]  # shape (N, N, d)
    dists = np.sum(diff**2, axis=-1)      # shape (N, N)

    # Sort indices by distance for each point
    # The closest point is itself (distance 0), so we skip index 0.
    sorted_idx = np.argsort(dists, axis=1)   # shape (N, N)
    neighbors_idx = sorted_idx[:, 1:k+1]     # skip self

    return neighbors_idx


def estimate_tangent_normal_pca(X, neighbors_idx):
    """
    Estimate tangent and normal vectors at each point using local PCA.

    This is the *function-free* estimator: it only uses the point cloud X
    and its k-NN neighborhoods.

    Parameters
    ----------
    X : np.ndarray, shape (N, 2)
        Data points (here, 2D spiral).
    neighbors_idx : np.ndarray, shape (N, k)
        neighbors_idx[i] are indices of neighbors of X[i].

    Returns
    -------
    T_est : np.ndarray, shape (N, 2)
        Estimated unit tangent vectors at each point.
    N_est : np.ndarray, shape (N, 2)
        Estimated unit normal vectors at each point.
    """
    X = np.asarray(X)
    neighbors_idx = np.asarray(neighbors_idx)
    N, d = X.shape
    assert d == 2, "This version assumes 2D data (for the spiral)."

    T_est = np.zeros_like(X)
    N_est = np.zeros_like(X)

    for i in range(N):
        # Neighbor coordinates
        neigh = X[neighbors_idx[i]]  # shape (k, 2)

        # Center neighbors
        mean = neigh.mean(axis=0)
        Y = neigh - mean

        # Local covariance (2x2)
        C = (Y.T @ Y) / Y.shape[0]

        # Eigen-decomposition
        # eigh returns eigenvalues in ascending order
        evals, evecs = np.linalg.eigh(C)  # evecs[:, j] is eigenvector for evals[j]

        # Tangent: eigenvector with largest eigenvalue
        tangent_vec = evecs[:, 1]   # index 1 = largest eigenvalue in 2D
        normal_vec  = evecs[:, 0]   # smaller eigenvalue

        # Ensure unit length (should already be)
        tangent_vec = tangent_vec / (np.linalg.norm(tangent_vec) + 1e-12)
        normal_vec  = normal_vec  / (np.linalg.norm(normal_vec)  + 1e-12)

        T_est[i] = tangent_vec
        N_est[i] = normal_vec

    return T_est, N_est
