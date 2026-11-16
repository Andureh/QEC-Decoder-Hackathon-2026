#!/usr/bin/env bash
# Download the trained network weights (2 x 135 MB) from the GitHub release into weights/.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/Andureh/QEC-Decoder-Hackathon-2026/releases/download/v1.0
declare -A SHA=(
  [X.npz]=bca97bbaf17a3c3d269738e7062df99877a39ddddcf568680282fbbb3e098833
  [Z.npz]=d8d071f655747e3fd22b2b2b9836285f7c17a5587fc15f21193e9925a7ae767b
)
mkdir -p weights
for f in X.npz Z.npz; do
  if [ -f "weights/$f" ] && echo "${SHA[$f]}  weights/$f" | sha256sum -c --quiet 2>/dev/null; then
    echo "weights/$f already present"; continue
  fi
  echo "downloading $f ..."
  curl -fL --retry 3 -o "weights/$f" "$BASE/$f"
  echo "${SHA[$f]}  weights/$f" | sha256sum -c
done
