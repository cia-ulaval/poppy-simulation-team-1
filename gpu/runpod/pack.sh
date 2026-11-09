#!/usr/bin/env bash
# Sur le portable : prépare ce qu'on envoie au pod, sans rien pousser.
#
#   gpu/runpod/pack.sh            # code seul
#   gpu/runpod/pack.sh --models   # + best_model H-Placo s7 et s42 (pour --init-from)
#
# Produit dans logs/runpod/envoi/ (ignoré par git) :
#   poppy-<commit>.tar.gz   archive git du commit courant (git archive HEAD) ;
#   models.tar.gz           les best_model H-Placo CPU et leur vec_normalize.pkl.
# Les deux s'envoient par scp (voir gpu/runpod/README.md).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if ! git diff --quiet HEAD -- gpu src scripts configs assets requirements; then
  echo "Modifications non commitées dans le code : l'archive ne contiendrait que HEAD." >&2
  exit 1
fi
SHA="$(git rev-parse --short HEAD)"
OUT="logs/runpod/envoi"  # logs/ est ignoré par git
mkdir -p "$OUT"
git archive --format=tar.gz --prefix=poppy/ --add-virtual-file="poppy/COMMIT:$SHA" \
  -o "$OUT/poppy-$SHA.tar.gz" HEAD
echo "Code : $OUT/poppy-$SHA.tar.gz ($(du -h "$OUT/poppy-$SHA.tar.gz" | cut -f1))"

if [[ "${1:-}" == "--models" ]]; then
  # Les modèles vivent dans les logs du dépôt principal, hors git.
  MAIN="$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
  SRC="$MAIN/logs/hetzner/poppy-hplaco"
  files=()
  for seed in 7 42; do
    best="$(ls -d "$SRC"/exp_hp_s$seed/*/best_model | head -1)"
    files+=("${best#"$MAIN"/}")
  done
  tar -czf "$OUT/models.tar.gz" -C "$MAIN" "${files[@]}"
  echo "Modèles : $OUT/models.tar.gz (${files[*]})"
fi
