# syntax=docker/dockerfile:1
#
# Image GPU : entraînement de Poppy sur MuJoCo Warp (mjlab + rsl_rl).
#
# Séparée de docker/Dockerfile : mjlab épingle mujoco ~= 3.11, l'image CPU
# est en 3.13. Les deux piles ne cohabitent pas dans un même environnement.
#
#   docker build -f gpu/Dockerfile -t poppy-gpu .
#   docker run --rm --gpus all -v "$PWD:/workspace" poppy-gpu python -m gpu.train
#
# Prérequis : pilote NVIDIA récent (CUDA >= 12.4 pour les graphes CUDA de
# mjlab), Docker Desktop en backend WSL 2 sous Windows.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace \
    MUJOCO_GL=egl

# git : rsl_rl journalise l'état des dépôts avec GitPython, qui exige le
# binaire. libegl1 / libgl1 / libosmesa6 : rendu MuJoCo hors écran (vidéos).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libegl1 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
COPY gpu/requirements.txt /tmp/requirements-gpu.txt
RUN python -m pip install --index-url "${TORCH_INDEX}" torch \
    && python -m pip install -r /tmp/requirements-gpu.txt

RUN useradd --create-home --uid 1000 poppy
WORKDIR /workspace
