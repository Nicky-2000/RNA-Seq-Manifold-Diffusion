# RNA-Seq Manifold Diffusion

A modular research framework for experimenting with manifold-aware diffusion models on RNA-Seq and synthetic datasets.  
The goal is to explore how non-Gaussian forward noise processes—guided by a learned manifold—affect diffusion model training and sample quality.

Pipeline:

    Data → Manifold → Diffusion → Metrics → Experiments

---

## Quickstart

### 1. Clone the Repository

    git clone git@github.com:Nicky-2000/RNA-Seq-Manifold-Diffusion.git
    cd RNA-Seq-Manifold-Diffusion

### 2. Create a Virtual Environment

Using venv:

    python3 -m venv .venv
    source .venv/bin/activate

Or Conda:

    conda create -n rna-diffusion python=3.10
    conda activate rna-diffusion

### 3. Install Dependencies

    pip install -r requirements.txt

---

## Run Your First Experiment (Baseline Gaussian Diffusion)

All runnable scripts live under:  
`src/experiments`

Run the baseline:

    cd src
    python3 -m experiments.train_baseline

Expected output:

    [utils.logging] Seed set to 0
    Using device: cpu
    Loaded data with shape: torch.Size([5000, 3])
    Fitted IdentityManifold.
    Created GaussianNoiser with T=1000 timesteps.
    [baseline] epoch=1/3  loss=49.6607
    [baseline] epoch=2/3  loss=1.4176
    [baseline] epoch=3/3  loss=1.0149

Yay! Good job.

---

## 📁 Project Structure

    RNA-Seq-Manifold-Diffusion/
      README.md
      requirements.txt
      src/
        __init__.py
          data/
            __init__.py
            loaders.py         # Load RNA (.h5ad) or swiss roll
            preprocess.py      # Preprocessing hooks
          manifold/
            __init__.py
            base.py            # Manifold + Noiser interfaces
            identity.py        # Trivial manifold
            noisers.py         # Gaussian DDPM noiser
          diffusion/
            __init__.py
            base.py            # DiffusionModel (training logic)
            networks.py        # MLP denoiser
          metrics/
            __init__.py
            basic.py           # Evaluation stubs
          utils/
            __init__.py
            logging.py         # set_seed(), get_logger()
          experiments/
            __init__.py
            train_baseline.py  # Baseline Gaussian diffusion
            train_manifold.py  # Placeholder for manifold version
            eval_compare.py    # Compare two models
      tests/
        Will do this eventually ugh

---

## 🧱 Pipeline Overview

1. Data  
   Load RNA-seq (`.h5ad`) or synthetic datasets (swiss roll) as PyTorch tensors.

2. Manifold  
   Learn or define geometric structure.  
   Current:
   - IdentityManifold  
   - GaussianNoiser (DDPM)

3. Diffusion  
   Forward noise + denoising network (MLP).

4. Metrics  
   Sample quality, manifold distance, reconstruction, etc.

5. Experiments  
   Training, evaluation, and GCP job execution.

---

## 📌 Future Directions

- Manifold-aware noisers  
- Reverse sampling + visualization  
- Coverage & manifold-distance metrics  
- Sample-efficiency comparisons  
- Trajectory inference for RNA-seq  
- Automated GCP training pipelines  
