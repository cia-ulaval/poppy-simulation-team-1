#!/usr/bin/env bash
# Mesure M1–M7 de nombreux modèles sur le banc CPU, en parallèle sur les cœurs du pod.
#
#   bash gpu/runpod/criteres_parallel.sh <config.yaml> <sortie> <modele.zip>... [-- terrain...]
#
# Exemples :
#   bash gpu/runpod/criteres_parallel.sh gpu/configs/h_placo_cpu.yaml logs/runpod/crit runs/*/model_*.zip
#   bash gpu/runpod/criteres_parallel.sh configs/palier2_hplaco.yaml logs/runpod/crit_p2 \
#        runs/p2_s7/model_*.zip -- 0.3 0.6 1.0
#
# Une tâche = un modèle × une difficulté, 10 épisodes, graine 1 : exactement
# gpu/criteres.py, dans .venv-cpu (MuJoCo C 3.13, le banc de référence). Les
# modèles GPU doivent être exportés avant (gpu/export_sb3.py).
#
# Un processus par cœur, chacun à un seul fil : sans OMP_NUM_THREADS=1, chaque
# PyTorch prend tous les cœurs et les mesures s'étouffent (vu sur Hetzner).
# JOBS (défaut : nombre de cœurs) règle le parallélisme.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
CONFIG="$1"; OUT="$2"; shift 2
models=(); terrains=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; terrains=("$@"); break; fi
  models+=("$1"); shift
done
[[ ${#terrains[@]} -eq 0 ]] && terrains=(0)
JOBS="${JOBS:-$(nproc)}"
mkdir -p "$OUT"
export ROOT CONFIG OUT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"

task() {
  local model="$1" terrain="$2"
  local name; name="$(basename "$(dirname "$model")")__$(basename "$model" .zip)__t$terrain"
  "$ROOT/.venv-cpu/bin/python" "$ROOT/gpu/criteres.py" --config "$CONFIG" --model "$model" \
    --terrain "$terrain" --episodes 10 --seed 1 --json "$OUT/$name.json" \
    > "$OUT/$name.md" 2> "$OUT/$name.err" && tail -1 "$OUT/$name.md"
}
export -f task

start=$(date +%s)
for m in "${models[@]}"; do for t in "${terrains[@]}"; do echo "$m $t"; done; done \
  | xargs -P "$JOBS" -L 1 bash -c 'task "$0" "$1"' | tee "$OUT/tableau.md"
echo "$(( ${#models[@]} * ${#terrains[@]} )) mesures en $(( $(date +%s) - start )) s sur $JOBS cœurs" \
  | tee -a "$OUT/tableau.md"
