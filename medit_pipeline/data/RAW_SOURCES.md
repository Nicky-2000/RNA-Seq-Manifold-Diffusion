# Raw data registry

List each raw file you add under `data/raw/`, along with its provenance and checksum.

Example entry:
- source: ukb_rbc_gwas
  - file: data/raw/ukb_rbc_gwas/MCH.sumstats.bgz
  - url: https://example.org/ukb/MCH.sumstats.bgz
  - sha256: <fill-after-download>
  - notes: UKB Neale Lab MCH summary stats (build GRCh37)

Sources to expect:
- ukb_rbc_gwas — UKB RBC trait summary stats (MCH, RDW, IRF)
- smr_curated — SMR/HEIDI results files (gene, effect s, standard error se)
- k562_gwps — K562 Perturb-seq GWPS (beta matrices, regulator metadata)
- hct116_gwps — HCT116/Orion dose strata and beta, if applicable
