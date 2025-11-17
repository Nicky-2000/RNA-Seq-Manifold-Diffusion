# =========================
# MEDIT — Repro Makefile
# =========================

PROJECT ?= medit-478122
ZONE    ?= us-west4-a
VM      ?= medit-g2
BUCKET  ?= medit-uml-prod-uscentral1-8e7a

SHELL := /bin/bash
GCLOUD ?= gcloud
REMOTE = $(VM) --project=$(PROJECT) --zone="$(ZONE)"

# ---- Runners ----
# Local runner
LOCAL_PY ?= python

# VM runner (conda env "venv" on the VM)
VM_CONDA_RUN = conda run -n venv
VM_PY        = $(VM_CONDA_RUN) python

.PHONY: help
help:
	@echo "Targets:"
	@echo "  bootstrap        : Upload and run tools/bootstrap_vm.sh on VM"
	@echo "  bootstrap.run    : Re-run bootstrap on VM (script already there)"
	@echo "  gcs.init         : Ensure GCS prefixes (data/raw, data/prep, out)"
	@echo "  gs.ls            : List bucket top-level and common prefixes"
	@echo "  vm.ssh           : Open interactive SSH to VM"
	@echo "  vm.cuda          : Check nvidia-smi & PyTorch CUDA on VM (uses conda env 'venv')"
	@echo "  vm.run           : Run pipeline target on VM (PIPELINE=<target>, default: all; enforces 'venv')"
	@echo "  vm.ensure_dirs   : Ensure ~/Medit/data/raw, data/prep, out on VM"
	@echo "  all              : Run qc locally (assumes raw data already in data/raw)"
	@echo "  qc               : Run QC + preprocessing to data/prep locally"
	@echo "  data.download    : Download raw data on VM → GCS (via tools/download_data.sh)"
	@echo "  cleanup.*        : Cleanup helpers on VM"

# -------------------------
# VM bootstrap
# -------------------------
.PHONY: bootstrap bootstrap.run
bootstrap:
	$(GCLOUD) compute scp tools/bootstrap_vm.sh "$(VM):~/bootstrap_vm.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "chmod +x ~/bootstrap_vm.sh && PROJECT=$(PROJECT) BUCKET=$(BUCKET) ~/bootstrap_vm.sh"'

bootstrap.run:
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "PROJECT=$(PROJECT) BUCKET=$(BUCKET) ~/bootstrap_vm.sh"'

# -------------------------
# GCS helpers
# -------------------------
.PHONY: gcs.init gs.ls
gcs.init:
	$(GCLOUD) compute scp tools/init_gcs.sh "$(VM):~/init_gcs.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "chmod +x ~/init_gcs.sh && BUCKET=$(BUCKET) ~/init_gcs.sh"'

gs.ls:
	$(GCLOUD) compute ssh $(REMOTE) -- \
	  'bash -lc "\
	    echo \"== bucket root ==\"; gsutil ls gs://$(BUCKET)/; \
	    echo \"== data/raw ==\";  gsutil ls gs://$(BUCKET)/data/raw/ || true; \
	    echo \"== data/prep ==\"; gsutil ls gs://$(BUCKET)/data/prep/ || true; \
	    echo \"== out ==\";       gsutil ls gs://$(BUCKET)/out/ || true"'

# -------------------------
# VM utilities
# -------------------------
.PHONY: vm.ssh vm.cuda vm.run vm.ensure_dirs
vm.ssh:
	$(GCLOUD) compute ssh $(REMOTE)

vm.cuda:
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
		PY=\"$(VM_PY)\"; \
		echo \"== nvidia-smi ==\"; (nvidia-smi || sudo nvidia-smi || true); \
		echo \"== PyTorch CUDA check ==\"; \
		$$PY - <<PY \
import torch, sys; \
print(\"torch:\", getattr(torch, \"__version__\", \"?\")); \
print(\"cuda available:\", torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\"); \
PY"'

# Which pipeline target to run on the VM (reuses this Makefile on the VM)
PIPELINE ?= all
vm.run:
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
		cd ~/MEDIT && \
		$(MAKE) $(PIPELINE) PY_CMD=\"$(VM_PY)\" "'

vm.ensure_dirs:
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
	  mkdir -p ~/MEDIT/data/raw ~/MEDIT/data/prep ~/MEDIT/out && \
	  echo \"[ok] ensured local workspace dirs\" "'

# -------------------------
# LOCAL pipeline (uses LOCAL_PY by default)
# You can override: `make all PY_CMD="conda run -n venv python"`
# -------------------------
PY_CMD ?= $(LOCAL_PY)

.PHONY: all qc
all: qc

# Run QC + preprocessing, writing final prepped data to data/prep
# (update --adata path if your raw file lives elsewhere)
qc:
	$(PY_CMD) scripts/qc_eda.py \
	  --params configs/params.yml \
	  --out data/prep \
	  --adata data/raw/stateFate_inVitro.h5ad

# -------------------------
# Data download to GCS from VM
# -------------------------
.PHONY: data.download
data.download:
	$(GCLOUD) compute scp tools/download_data.sh "$(VM):~/download_data.sh" \
	  --project="$(PROJECT)" --zone="$(ZONE)"
	$(GCLOUD) compute ssh $(REMOTE) -- 'bash -lc "\
		chmod +x ~/download_data.sh && \
		BUCKET='$(BUCKET)' \
		PREFIX=data/raw \
		MANIFEST=\$$HOME/MEDIT/medit_pipeline/configs/manifest/weinreb_manifest.csv \
		WORKDIR=\$$HOME/tmp_downloads \
		~/download_data.sh"'

# -------------------------
# Cleanup helpers on VM
# -------------------------
.PHONY: cleanup.staging cleanup.raw.local cleanup.all
cleanup.staging:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone="$(ZONE)" -- \
	  'bash -lc "cd ~/MEDIT && ./tools/cleanup.sh staging"'

cleanup.raw.local:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone="$(ZONE)" -- \
	  'bash -lc "cd ~/MEDIT && ./tools/cleanup.sh raw-local"'

cleanup.all:
	@$(GCLOUD) compute ssh $(VM) --project=$(PROJECT) --zone="$(ZONE)" -- \
	  'bash -lc "cd ~/MEDIT && ./tools/cleanup.sh all"'
