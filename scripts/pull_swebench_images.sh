#!/bin/bash
# Pre-download swe-rex wheels + pull SWE-bench Verified sandbox images.
# Run ONCE before inference. Can be executed from anywhere.
#
# 1. Pre-download swe-rex + deps wheels (for offline sandbox pip install,
#    avoids 500-sandbox concurrent pip throttle on Tsinghua mirror).
# 2. Pull SWE-bench images from 火山引擎 CR (国内快, 国内镜像源),
#    docker tag to swebench/ Docker Hub format (what the parquet expects).
#
# Usage:
#   bash scripts/pull_swebench_images.sh [DATA_PATH]
#   bash scripts/pull_swebench_images.sh scripts/swe_bench_verified_modal.parquet

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH=${1:-${SCRIPT_DIR}/swe_bench_verified_modal.parquet}
WHEELS_DIR=${WHEELS_DIR:-/data1/lln/swe_wheels}
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: dataset not found: $DATA_PATH" >&2
    exit 1
fi

# ── 1. Pre-download swe-rex wheels ──
if [ ! -d "$WHEELS_DIR" ] || [ -z "$(ls -A "$WHEELS_DIR" 2>/dev/null)" ]; then
    echo "=== Pre-download swe-rex wheels to $WHEELS_DIR ==="
    mkdir -p "$WHEELS_DIR"
    pip download swe-rex -d "$WHEELS_DIR" -i "$PIP_INDEX"
    echo "wheels: $(ls "$WHEELS_DIR" | wc -l) files"
else
    echo "=== swe-rex wheels already in $WHEELS_DIR ($(ls "$WHEELS_DIR" | wc -l) files), skip ==="
fi

# ── 2. Pull SWE-bench images from 火山引擎 CR + tag ──
echo "=== Pull SWE-bench images from 火山引擎 CR ==="
echo "dataset: $DATA_PATH"

python3 - "$DATA_PATH" <<'PYEOF'
import sys, json, subprocess
import pyarrow.parquet as pq

t = pq.read_table(sys.argv[1])
ei = t.column('extra_info').to_pylist()
imgs = set()
for e in ei:
    s = json.loads(e) if isinstance(e, str) else e
    def find(o):
        if isinstance(o, dict):
            for v in o.values(): find(v)
        elif isinstance(o, str) and 'sweb.eval' in o:
            imgs.add(o)
    find(s)

CR = "enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified"
ok = fail = skip = 0
for img in sorted(imgs):
    inst = img.split('/')[-1]  # sweb.eval.x86_64.<instance>
    target = f"swebench/{inst}:latest"
    # skip if already tagged
    existing = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                              capture_output=True, text=True).stdout.split()
    if target in existing:
        skip += 1
        continue
    cr_img = f"{CR}/{inst}:v2"
    if subprocess.run(["docker", "pull", cr_img]).returncode == 0:
        subprocess.run(["docker", "tag", cr_img, target], capture_output=True)
        ok += 1
    else:
        print(f"  FAIL: {inst}")
        fail += 1

total = len(imgs)
print(f"=== images: total={total} pulled={ok} skipped={skip} failed={fail} ===")
PYEOF

echo "✅ Prep done: wheels in $WHEELS_DIR, images tagged swebench/*:latest"
