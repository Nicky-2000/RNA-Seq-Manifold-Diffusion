import numpy as np

import numpy as np
from scipy import sparse


# Weinreb stateFate_inVitro raw cell count mapping
class Weinreb:
    """
    Schema for the Weinreb et al. stateFate_inVitro AnnData object.

    This reflects:
      - ad.obs columns
      - ad.var index (gene symbols)
      - ad.obsm['X_clone_membership']
      Shape of AnnData object is a matrix (n_cells, n_genes): (130887, 25289)
      Each cell in the 130887 rows has meta data associated which is stored in the columns of ad.obs
    """

    def __init__(self) -> None:        
        # ---- ad.obs columns ----

        # Library ID / batch / sub-replicate label
        # e.g. 'LK_d2', 'LSK_d4_1_2', 'd6_2_2'
        self.library: str

        # InDrops cell barcode (unique-ish ID per captured cell)
        # e.g. 'AAACAAAC-AAACTGGT'
        self.cell_barcode: str

        # Time point in days after culture (2.0, 4.0, 6.0)
        self.time_point: np.float64

        # FACS starting population (initial condition)
        # 'Lin-Kit+Sca1+' (LSK, stem-like) or 'Lin-Kit+Sca1-' (LK, progenitor)
        self.starting_population: str

        # Manual / semi-supervised cell type annotation
        # e.g. 'Undifferentiated', 'Neutrophil', 'Monocyte', ...
        self.cell_type_annotation: str

        # Plate / well index (technical batch; 0, 1, 2)
        self.well: np.int64

        # SPRING 2D layout coordinates for visualization
        # (not needed for modeling; purely for plotting)
        self.spring_x: np.float64
        self.spring_y: np.float64

        # ---- ad.var "columns" (gene metadata) ----

        # Gene symbol from ad.var_names
        # e.g. '0610006L08Rik', 'Gata1', 'Mpo', ...
        self.gene_symbol: str

        # ---- ad.obsm entries ----

        # Clone membership matrix (cells x clones), typically sparse 0/1.
        # Each row: clone assignments for that cell.
        self.X_clone_membership: sparse.csr_matrix
