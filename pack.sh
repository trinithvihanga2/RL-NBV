#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-rl-nbv-trainer-bundle.zip}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

FILES=(
    Dockerfile
    docker-compose.yml
    config.yaml
    pyproject.toml
    uv.lock
    train.py
    custom_callback.py
    envs
    models/__init__.py
    models/pointnet2_cls_ssg.py
    models/pointnet2_utils.py
    optim
    distance/chamfer_distance.py
    distance/chamfer_distance.cpp
    distance/chamfer_distance.cu
    data
)

if [[ -d models/pretrained ]]; then
    FILES+=(models/pretrained)
else
    echo "Warning: models/pretrained not found; pretrained weights will be missing." >&2
fi

MISSING=()
for f in "${FILES[@]}"; do
    if [[ ! -e "$f" ]]; then
        MISSING+=("$f")
    fi
done

if (( ${#MISSING[@]} )); then
    echo "Warning: missing files/directories will be skipped:" >&2
    for f in "${MISSING[@]}"; do
        echo "  - $f" >&2
    done
fi

rm -f "$OUT"
zip -r "$OUT" "${FILES[@]}" \
    -x "**/__pycache__/**" "*.pyc" "*.pyo" "*.so"

echo "Created $OUT"
