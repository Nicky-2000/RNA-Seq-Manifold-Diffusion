FROM mambaorg/micromamba:1.5.8-nonroot

# ---------------------------
# Base system deps
# ---------------------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ curl build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------
# CUDA + PyTorch base (CUDA 12.1)
# ---------------------------
# micromamba base image includes no CUDA — but pytorch-cuda=12.1 provides the runtime
COPY configs/env.yml /tmp/env.yml

# ---------------------------
# Create micromamba environment
# ---------------------------
RUN micromamba create -y -n medit -f /tmp/env.yml && \
    micromamba clean --all --yes

# Automatically activate env in all RUN steps from now on
ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["micromamba", "run", "-n", "medit", "/bin/bash", "-c"]

ENV PATH="/opt/conda/envs/medit/bin:$PATH"

# ---------------------------
# Project code
# ---------------------------
WORKDIR /workspace
COPY . /workspace

CMD ["bash"]
