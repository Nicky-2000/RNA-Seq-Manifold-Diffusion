import numpy as np
import np.int64

# Replogie hda5 raw cell count mapping
class Replogie():
    def __init__(self):
        # ad.obs cols
        self.gem_group: np.int64
        self.gene: str
        self.gene_id: str
        self.transcript: str
        self.gene_transcript: str
        self.sgID_AB: str
        self.mitopercent: np.float32
        self.UMI_count: np.float32
        self.z_gemgroup_UMI: np.float32
        self.core_scale_factor: np.float32
        self.core_adjusted_UMI_count: np.float32

        # ad.var cols
        self.gene_name: str
        self.chr: str
        self.start: np.int64
        self.end: np.int64
        self.Class: str
        self.strand: str
        self.length: np.int64
        self.in_matrix: bool
        self.mean: np.float32
        self.std: np.float32
        self.cv: np.float32
        self.fano: np.float32
