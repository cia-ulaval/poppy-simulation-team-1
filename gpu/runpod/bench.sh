#!/usr/bin/env bash
# Sur le pod, après setup.sh :
#
#   bash gpu/runpod/bench.sh            # (a) à (c), puis (e) : ~25 à 35 min sur une 4090
#   TERRAIN=1 bash gpu/runpod/bench.sh  # + (d), débit sur terrain : ~10 min de plus
#
# Mesure, sans rien garder d'autre que des journaux :
#   (a) la carte : nvidia-smi, nom, mémoire vidéo ;
#   (b) un run H-Placo sur sol plat, 2 M de pas, 256 × 16 (répartition validée) ;
#   (c) 2 puis 4 runs H-Placo simultanés de 2 M : débit total, mémoire vidéo ;
#   (d) option TERRAIN=1 : palier 2 sur terrain, 1 M de pas, 256 × 16 puis 1024 × 4 ;
#   (e) tout dans logs/runpod/<carte>/ et une archive logs/runpod/<carte>.tar.gz.
# Le débit est celui que gpu/train.py écrit dans throughput.json (pas/s,
# compilation des noyaux comprise : c'est ce que coûte un vrai run).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PYG="$ROOT/.venv-gpu/bin/python"
export PYTHONPATH="$ROOT"
STEPS="${STEPS:-2000000}"

CARD="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' )"
OUT="$ROOT/logs/runpod/$CARD"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/bench.log") 2>&1
echo "== Banc RunPod, $(date -u +%FT%TZ), carte $CARD, commit $(cat "$ROOT/COMMIT" 2>/dev/null || echo '?')"

# (a) la carte
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv | tee "$OUT/carte.csv"

# Mémoire vidéo échantillonnée chaque seconde pendant tout le banc.
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 \
  > "$OUT/memoire.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

run() {  # run <nom> <config> <num_envs> <steps_per_env> <pas> [seed]
  local name="$1" cfg="$2" n="$3" t="$4" steps="$5" seed="${6:-42}"
  "$PYG" -m gpu.train --config "$cfg" --num-envs "$n" --steps-per-env "$t" \
    --timesteps "$steps" --seed "$seed" --log-dir "$OUT/$name" > "$OUT/$name.log" 2>&1
}

summary() {  # summary <dossier>... : somme des débits
  "$PYG" - "$@" <<'EOF'
import json, sys
from pathlib import Path
rates = []
for d in sys.argv[1:]:
    t = json.loads((Path(d) / "throughput.json").read_text())
    rates.append(t["steps_per_second"])
    print(f"  {Path(d).name:<24} {t['steps_per_second']:8.0f} pas/s  {t['seconds'] / 60:5.1f} min")
print(f"  TOTAL {sum(rates):.0f} pas/s sur {len(rates)} run(s)")
EOF
}

peak_mem() {  # pic de mémoire vidéo depuis la ligne <n> de memoire.csv
  tail -n +"$1" "$OUT/memoire.csv" | awk -F', ' '{gsub(/ MiB/,"",$2); if ($2>m) m=$2} END {print m " MiB"}'
}

# (b) un run seul, sol plat
echo "== (b) H-Placo, sol plat, 1 run de $STEPS pas"
L=$(wc -l < "$OUT/memoire.csv")
run plat_1x gpu/configs/h_placo.yaml 256 16 "$STEPS"
summary "$OUT/plat_1x" | tee "$OUT/b_plat_1x.txt"
echo "  mémoire vidéo max : $(peak_mem "$L")" | tee -a "$OUT/b_plat_1x.txt"

# (c) montée en parallèle
for k in 2 4; do
  echo "== (c) H-Placo, sol plat, $k runs simultanés de $STEPS pas"
  L=$(wc -l < "$OUT/memoire.csv")
  pids=()
  for i in $(seq 1 "$k"); do
    run "plat_${k}x_$i" gpu/configs/h_placo.yaml 256 16 "$STEPS" "$i" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  dirs=(); for i in $(seq 1 "$k"); do dirs+=("$OUT/plat_${k}x_$i"); done
  summary "${dirs[@]}" | tee "$OUT/c_plat_${k}x.txt"
  echo "  mémoire vidéo max : $(peak_mem "$L")" | tee -a "$OUT/c_plat_${k}x.txt"
done

# (d) terrain, en option
if [[ "${TERRAIN:-0}" == "1" ]]; then
  for layout in "256 16" "1024 4"; do
    set -- $layout
    echo "== (d) palier 2, terrain, $1 × $2, 1 M de pas"
    L=$(wc -l < "$OUT/memoire.csv")
    run "terrain_${1}x$2" gpu/configs/palier2_hplaco.yaml "$1" "$2" 1000000 7
    summary "$OUT/terrain_${1}x$2" | tee "$OUT/d_terrain_${1}x$2.txt"
    echo "  mémoire vidéo max : $(peak_mem "$L")" | tee -a "$OUT/d_terrain_${1}x$2.txt"
  done
fi

# (e) archive à rapatrier (sans les checkpoints, inutiles ici)
kill $SMI 2>/dev/null || true
tar -czf "$ROOT/logs/runpod/$CARD.tar.gz" -C "$ROOT/logs/runpod" \
  --exclude='*.pt' "$CARD" setup.log setup_tests.log 2>/dev/null || \
  tar -czf "$ROOT/logs/runpod/$CARD.tar.gz" -C "$ROOT/logs/runpod" --exclude='*.pt' "$CARD"
echo
echo "== Fini. Archive : logs/runpod/$CARD.tar.gz"
echo "   Depuis le portable : scp -P <port> root@<ip>:~/poppy/logs/runpod/$CARD.tar.gz logs/runpod/"
echo
echo "############################################################"
echo "#  SUPPRIMER LE POD dans l'interface RunPod (Terminate),   #"
echo "#  une fois l'archive rapatriée et vérifiée. Un pod arrêté #"
echo "#  mais non supprimé continue de facturer son disque.      #"
echo "############################################################"
