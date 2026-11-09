#!/usr/bin/env bash
# Sur le pod RunPod (modèle PyTorch/CUDA officiel, root), dans ~/poppy :
#
#   bash gpu/runpod/setup.sh
#
# Pas de Docker : deux environnements Python, côte à côte, comme les deux
# images du dépôt (elles ne cohabitent pas : mjlab épingle mujoco 3.11, le banc
# CPU est en 3.13) :
#   .venv-gpu  torch CUDA 12.8 + gpu/requirements.txt (mjlab, MuJoCo Warp, rsl_rl) ;
#   .venv-cpu  requirements/train.lock (le banc CPU : scripts/evaluate.py, criteres.py).
# Puis la suite de tests GPU sert de porte d'entrée : si elle échoue, ne rien
# lancer d'autre, rapatrier les journaux et supprimer le pod.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
LOG="$ROOT/logs/runpod"
mkdir -p "$LOG"
exec > >(tee "$LOG/setup.log") 2>&1

echo "== Carte et pilote"
nvidia-smi
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
echo "CUDA du pilote : $DRIVER_CUDA"

PY="${PYTHON:-python3}"
"$PY" - <<'EOF'
import sys
assert (3, 10) <= sys.version_info[:2] < (3, 14), f"Python 3.10 à 3.13 requis (mjlab), trouvé {sys.version}"
print("Python", sys.version.split()[0])
EOF

echo "== .venv-gpu"
"$PY" -m venv .venv-gpu
.venv-gpu/bin/pip install -q -U pip
# Mêmes versions que l'image poppy-gpu validée (torch 2.11.0+cu128) ;
# les roues cu128 contiennent Ada (sm_89, 4090) et Blackwell (sm_120, 5090).
.venv-gpu/bin/pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
.venv-gpu/bin/pip install -q -r gpu/requirements.txt

# warp-lang 1.19.0 sur PyPI est construit avec le CUDA Toolkit 13 : il lui faut
# un pilote CUDA >= 13.0. Sinon, la roue +cu12 publiée par NVIDIA sur la même
# version (vérifiée par son SHA-256).
if [[ "${DRIVER_CUDA%%.*}" -lt 13 ]]; then
  echo "Pilote CUDA $DRIVER_CUDA < 13 : warp-lang 1.19.0+cu12"
  WHL="warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl"
  curl -fsSL -o "/tmp/$WHL" "https://github.com/NVIDIA/warp/releases/download/v1.19.0/$WHL"
  echo "f95d9581ed12663df272e45a50101b171e143fde1ef7b2433a8bfc9fba096598  /tmp/$WHL" | sha256sum -c -
  .venv-gpu/bin/pip install -q --no-deps --force-reinstall "/tmp/$WHL"
fi

echo "== Vérification torch / Warp sur la carte"
PYTHONPATH="$ROOT" .venv-gpu/bin/python - <<'EOF'
import torch, warp as wp
assert torch.cuda.is_available(), "torch ne voit pas le GPU"
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
print("torch", torch.__version__, "CUDA", torch.version.cuda, "carte", torch.cuda.get_device_name(0), arch)
assert arch in torch.cuda.get_arch_list(), f"torch n'a pas de noyaux pour {arch} : {torch.cuda.get_arch_list()}"
wp.init()
x = wp.array([1.0, 2.0], dtype=float, device="cuda:0")
print("warp", wp.config.version, "ok :", x.numpy())
EOF

echo "== .venv-cpu (banc CPU)"
"$PY" -m venv .venv-cpu
.venv-cpu/bin/pip install -q -U pip
.venv-cpu/bin/pip install -q -r requirements/train.lock

echo "== Porte d'entrée : tests GPU"
if PYTHONPATH="$ROOT" .venv-gpu/bin/python -m pytest gpu/tests -q -p no:cacheprovider \
    > "$LOG/setup_tests.log" 2>&1; then
  tail -1 "$LOG/setup_tests.log"
  echo "TESTS GPU : OK. Suite : bash gpu/runpod/bench.sh"
else
  echo "TESTS GPU : ÉCHEC (logs/runpod/setup_tests.log). Ne rien lancer d'autre :"
  echo "rapatrier logs/runpod/ puis SUPPRIMER LE POD."
  exit 1
fi
