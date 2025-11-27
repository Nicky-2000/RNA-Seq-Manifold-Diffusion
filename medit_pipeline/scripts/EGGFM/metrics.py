# EGGFM/metrics.py

from typing import Dict, Any
import numpy as np
import torch
from scipy import sparse as sp_sparse
import scanpy as sc


# ---------- helpers ----------

def pairwise_norm(
    Xi: np.ndarray,
    Xj: np.ndarray,
    norm: str = "l2",
) -> np.ndarray:
    V = Xj - Xi
    if norm == "l2":
        return np.linalg.norm(V, axis=1)
    elif norm == "l1":
        return np.sum(np.abs(V), axis=1)
    elif norm == "l0":
        return np.sum(V != 0, axis=1)
    elif norm == "linf":
        return np.max(np.abs(V), axis=1)
    else:
        raise ValueError(f"Unknown norm: {norm}")

def compute_scalar_conformal_field(
    X_energy: np.ndarray,
    energy_model,
    diff_cfg: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """
    G(x) = gamma + lambda * exp(clip(E_norm(x))),
    with E_norm median-centered, MAD-scaled, and clipped to ±energy_clip_abs.

    X_energy: dense (n_cells, D_energy) array in the SAME space the energy
              model was trained on (HVG or PCA).
    """
    if sp_sparse.issparse(X_energy):
        X_energy = X_energy.toarray()
    X_energy = np.asarray(X_energy, dtype=np.float32)
    n_cells = X_energy.shape[0]

    metric_gamma = float(diff_cfg.get("metric_gamma", 0.2))
    metric_lambda = float(diff_cfg.get("metric_lambda", 4.0))
    energy_batch_size = int(diff_cfg.get("energy_batch_size", 2048))
    max_abs = float(diff_cfg.get("energy_clip_abs", 3.0))

    energy_model = energy_model.to(device)
    energy_model.eval()

    # per-gene mean / std (same as AnnDataExpressionDataset)
    mean = X_energy.mean(axis=0, keepdims=True)
    std = X_energy.std(axis=0, keepdims=True) + 1e-6
    X_energy_std = (X_energy - mean) / std
    print("[EGGFM SCM] computing energies E(x) for all cells...", flush=True)
    with torch.no_grad():
        X_energy_tensor = torch.from_numpy(X_energy_std).to(
            device=device, dtype=torch.float32
        )
        E_list = []
        for start in range(0, n_cells, energy_batch_size):
            end = min(start + energy_batch_size, n_cells)
            xb = X_energy_tensor[start:end]
            Eb = energy_model(xb)
            E_list.append(Eb.detach().cpu().numpy())
        E_vals = np.concatenate(E_list, axis=0).astype(np.float64)

    
    E_vals = np.concatenate(E_list, axis=0).astype(np.float64)  # (n_cells,)

    med = np.median(E_vals)
    mad = np.median(np.abs(E_vals - med)) + 1e-8
    E_norm = (E_vals - med) / mad

    E_clip = np.clip(E_norm, -max_abs, max_abs)

    G = metric_gamma + metric_lambda * np.exp(E_clip)
    if not np.isfinite(G).all():
        raise ValueError(
            "[EGGFM SCM] non-finite values in G after exp; check energy scaling."
        )

    print(
        "[EGGFM SCM] energy stats: "
        f"raw_min={E_vals.min():.4f}, raw_max={E_vals.max():.4f}, "
        f"norm_min={E_norm.min():.4f}, norm_max={E_norm.max():.4f}, "
        f"clip=±{max_abs:.1f}",
        flush=True,
    )
    print(
        "[EGGFM SCM] metric G stats: "
        f"min={G.min():.4f}, max={G.max():.4f}, mean={G.mean():.4f}",
        flush=True,
    )
    return G
# ---------- base class ----------

class BaseMetric:
    """
    Metric strategy API:

      state = prepare(ad_prep, energy_model, device)
      dist_vals = edge_distances(X_geom, rows, cols, state, norm_type)
    """

    def __init__(self, diff_cfg: Dict[str, Any]):
        self.diff_cfg = diff_cfg

    def prepare(
        self,
        ad_prep: sc.AnnData,
        energy_model,
        device: str,
    ) -> Dict[str, Any]:
        return {}

    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        raise NotImplementedError


# ---------- Euclidean metric ----------

class EuclideanMetric(BaseMetric):
    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        Xi = X_geom[rows]
        Xj = X_geom[cols]
        return pairwise_norm(Xi, Xj, norm=norm_type)


class SCMMetric(BaseMetric):
    """
    Scalar conformal metric:

      - prepare() computes G(x) from energies in energy space
      - edge_distances() builds d_ij from G and chosen norm
    """

    def prepare(
        self,
        ad_prep: sc.AnnData,
        energy_model,
        device: str,
    ) -> Dict[str, Any]:
        # Choose energy space to match how the model was trained.
        energy_source = self.diff_cfg.get("energy_source", "hvg").lower()        
        if energy_source == "hvg":
            X_energy = ad_prep.X
        elif energy_source == "pca":
            if "X_pca" not in ad_prep.obsm:
                raise ValueError(
                    "[SCMMetric] energy_source='pca' but 'X_pca' not found in ad_prep.obsm. "
                    "Run PCA / prep_for_manifolds before EGGFM."
                )
            X_energy = ad_prep.obsm["X_pca"]
        else:
            raise ValueError(f"[SCMMetric] Unknown energy_source: {energy_source}")

        G = compute_scalar_conformal_field(
            X_energy=X_energy,
            energy_model=energy_model,
            diff_cfg=self.diff_cfg,
            device=device,
        )
        return {"G": G}

    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        G = state["G"]
        Xi = X_geom[rows]
        Xj = X_geom[cols]

        base_dist = pairwise_norm(Xi, Xj, norm=norm_type)
        Gi = G[rows]
        Gj = G[cols]
        G_edge = 0.5 * (Gi + Gj)

        ell_sq = G_edge * (base_dist**2)
        ell_sq[ell_sq < 1e-12] = 1e-12
        return np.sqrt(ell_sq)

# ---------- Hessian-mixed metric ----------

def _hessian_quadratic_form_batched(
    energy_model,
    X_batch: np.ndarray,
    V_batch: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    energy_model.eval()
    X = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
    V = torch.from_numpy(V_batch).to(device=device, dtype=torch.float32)
    X.requires_grad_(True)

    E = energy_model(X)
    E_sum = E.sum()
    (grad_x,) = torch.autograd.grad(
        E_sum,
        X,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    Hv = torch.autograd.grad(
        grad_x,
        X,
        grad_outputs=V,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    q = (Hv * V).sum(dim=1)
    return q.detach().cpu().numpy()


class HessianMixedMetric(BaseMetric):
    def prepare(
        self,
        ad_prep: sc.AnnData,
        energy_model,
        device: str,
    ) -> Dict[str, Any]:
        energy_source = self.diff_cfg.get("energy_source", "hvg").lower()
        if energy_source == "hvg":
            X_energy = ad_prep.X
        elif energy_source == "pca":
            if "X_pca" not in ad_prep.obsm:
                raise ValueError(
                    "[HessianMixedMetric] energy_source='pca' but 'X_pca' missing."
                )
            X_energy = ad_prep.obsm["X_pca"]
        else:
            raise ValueError(f"Unknown energy_source: {energy_source}")

        if sp_sparse.issparse(X_energy):
            X_energy = X_energy.toarray()
        X_energy = np.asarray(X_energy, dtype=np.float32)

        return {
            "X_energy": X_energy,
            "energy_model": energy_model.to(device),
            "device": device,
        }

    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        diff_cfg = self.diff_cfg
        X_energy = state["X_energy"]
        energy_model = state["energy_model"]
        device = state["device"]

        edge_batch_size = diff_cfg.get("hvp_batch_size", 1024)

        hessian_mix_mode = diff_cfg.get("hessian_mix_mode", "additive")
        hessian_mix_alpha = float(diff_cfg.get("hessian_mix_alpha", 0.3))
        hessian_beta = float(diff_cfg.get("hessian_beta", 0.3))
        hessian_clip_std = float(diff_cfg.get("hessian_clip_std", 2.0))
        hessian_use_neg = bool(diff_cfg.get("hessian_use_neg", True))

        n_edges = rows.shape[0]
        ell_sq = np.empty(n_edges, dtype=np.float64)

        n_batches = (n_edges + edge_batch_size - 1) // edge_batch_size
        print("[EGGFM Metrics] computing Hessian-mixed edge lengths...", flush=True)

        for b in range(n_batches):
            start = b * edge_batch_size
            end = min((b + 1) * edge_batch_size, n_edges)
            if start >= end:
                break

            i_batch = rows[start:end]
            j_batch = cols[start:end]

            Xi_batch = X_energy[i_batch]
            Xj_batch = X_energy[j_batch]
            V_batch = Xj_batch - Xi_batch

            norms = np.linalg.norm(V_batch, axis=1, keepdims=True) + 1e-8
            # eucl2 = (norms.squeeze(-1)) ** 2
            eucl2 = pairwise_norm(Xi_batch, Xj_batch, norm=norm_type)

            if hessian_mix_mode == "none":
                ell_sq[start:end] = eucl2
            else:
                V_unit = V_batch / norms
                q_dir = _hessian_quadratic_form_batched(
                    energy_model, Xi_batch, V_unit, device
                )
                q_dir = np.asarray(q_dir, dtype=np.float64)

                if hessian_use_neg:
                    q_dir = -q_dir
                q_dir[np.isnan(q_dir)] = 0.0
                q_dir[q_dir < 1e-12] = 1e-12

                if hessian_mix_mode == "additive":
                    med_e = np.median(eucl2)
                    med_q = np.median(q_dir)
                    scale = med_e / (med_q + 1e-8)
                    q_rescaled = q_dir * scale
                    q_rescaled[q_rescaled < 1e-12] = 1e-12

                    alpha = max(0.0, min(1.0, hessian_mix_alpha))
                    ell_sq[start:end] = (1.0 - alpha) * eucl2 + alpha * q_rescaled

                elif hessian_mix_mode == "multiplicative":
                    med_q = np.median(q_dir)
                    mad_q = np.median(np.abs(q_dir - med_q)) + 1e-8
                    q_std = (q_dir - med_q) / mad_q

                    c = hessian_clip_std
                    q_std = np.clip(q_std, -c, c)

                    beta = hessian_beta
                    factor = np.exp(beta * q_std)
                    ell_sq[start:end] = eucl2 * factor

                else:
                    raise ValueError(f"Unknown hessian_mix_mode: {hessian_mix_mode!r}")

            if (b + 1) % 50 == 0 or b == n_batches - 1:
                print(
                    f"  [EGGFM Metrics] batch {b+1}/{n_batches} ({end}/{n_edges} edges)",
                    flush=True,
                )

        ell_sq[ell_sq < 1e-12] = 1e-12
        return np.sqrt(ell_sq)
