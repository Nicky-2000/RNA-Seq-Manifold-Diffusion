import numpy as np

def compute_angle_errors(v_true, v_est, eps=1e-8):
    """
    Compute per-point angle errors between true and estimated unit vectors.

    Angle is defined as:
        theta_i = arccos( | v_true_i · v_est_i | )

    Parameters
    ----------
    v_true : np.ndarray, shape (N, d)
        True unit vectors.
    v_est : np.ndarray, shape (N, d)
        Estimated unit vectors.
    eps : float
        Numerical epsilon for clipping.

    Returns
    -------
    angles_rad : np.ndarray, shape (N,)
        Angle error at each point, in radians.
    """
    v_true = np.asarray(v_true)
    v_est = np.asarray(v_est)
    assert v_true.shape == v_est.shape, "v_true and v_est must have same shape"

    # Ensure unit length (in case caller didn't)
    v_true_norm = v_true / (np.linalg.norm(v_true, axis=-1, keepdims=True) + eps)
    v_est_norm  = v_est  / (np.linalg.norm(v_est,  axis=-1, keepdims=True) + eps)

    dots = np.sum(v_true_norm * v_est_norm, axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)  # abs removes orientation sign

    angles_rad = np.arccos(dots)
    return angles_rad

def summarize_angle_errors(angles_rad):
    """
    Compute summary statistics for angle errors.

    Parameters
    ----------
    angles_rad : np.ndarray, shape (N,)
        Angle errors in radians.

    Returns
    -------
    stats : dict
        Dictionary with basic summary statistics, in both radians and degrees.
    """
    angles_rad = np.asarray(angles_rad)
    angles_deg = np.degrees(angles_rad)

    stats = {
        "mean_rad":   float(np.mean(angles_rad)),
        "median_rad": float(np.median(angles_rad)),
        "p90_rad":    float(np.percentile(angles_rad, 90)),
        "max_rad":    float(np.max(angles_rad)),

        "mean_deg":   float(np.mean(angles_deg)),
        "median_deg": float(np.median(angles_deg)),
        "p90_deg":    float(np.percentile(angles_deg, 90)),
        "max_deg":    float(np.max(angles_deg)),
    }
    return stats


def evaluate_tangent_normal_errors(T_true, T_est, N_true=None, N_est=None):
    """
    Evaluate angle errors for tangent (and optionally normal) vectors.

    Parameters
    ----------
    T_true : np.ndarray, shape (N, d)
        True tangent vectors.
    T_est : np.ndarray, shape (N, d)
        Estimated tangent vectors.
    N_true : np.ndarray, shape (N, d), optional
        True normal vectors.
    N_est : np.ndarray, shape (N, d), optional
        Estimated normal vectors.

    Returns
    -------
    results : dict
        {
          "tangent": {
              "angles_rad": np.ndarray,
              "stats": { ... }
          },
          "normal": { ... }  # only if N_true and N_est are provided
        }
    """
    results = {}

    # Tangents
    tan_angles = compute_angle_errors(T_true, T_est)
    tan_stats = summarize_angle_errors(tan_angles)
    results["tangent"] = {
        "angles_rad": tan_angles,
        "stats": tan_stats,
    }

    # Normals (optional)
    if N_true is not None and N_est is not None:
        norm_angles = compute_angle_errors(N_true, N_est)
        norm_stats = summarize_angle_errors(norm_angles)
        results["normal"] = {
            "angles_rad": norm_angles,
            "stats": norm_stats,
        }

    return results
