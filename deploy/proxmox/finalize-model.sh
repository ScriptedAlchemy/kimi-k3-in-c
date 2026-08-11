#!/usr/bin/env bash
# Verify the official checkpoint, pack the lossless BF16 trunk, and mark it ready.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/srv/kimi/model}
TRUNK_DIR=${TRUNK_DIR:-/srv/kimi/trunk}
MODEL_REVISION=${MODEL_REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}
REPO_DIR=${REPO_DIR:-/opt/kimi-k3-in-c}

if [[ ${EUID} -ne 0 ]]; then
    echo "run as root inside the LXC guest" >&2
    exit 1
fi

count=$(find "${MODEL_DIR}" -maxdepth 1 \
    -name 'model-*-of-000096.safetensors' | wc -l)
if [[ ${count} -ne 96 ]]; then
    echo "checkpoint is incomplete: ${count}/96 weight shards" >&2
    exit 1
fi

if [[ ${SKIP_HF_VERIFY:-0} != 1 ]]; then
    command -v hf >/dev/null || {
        echo "hf CLI is required for checksum verification" >&2
        exit 1
    }
    hf cache verify moonshotai/Kimi-K3 \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_DIR}" \
        --fail-on-missing-files
fi

"${REPO_DIR}/scripts/pack-trunk.sh" "${MODEL_DIR}" "${TRUNK_DIR}"
test -s "${TRUNK_DIR}/trunk.bin"
test -s "${TRUNK_DIR}/trunk.json"

{
    echo "repo=https://github.com/ScriptedAlchemy/kimi-k3-in-c"
    echo "source_commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"
    echo "model=moonshotai/Kimi-K3"
    echo "model_revision=${MODEL_REVISION}"
    echo "weight_format=official MXFP4 experts + BF16 trunk"
} > /srv/kimi/.ready

if systemctl cat k3serve.service >/dev/null 2>&1; then
    systemctl restart k3serve.service
fi

cat /srv/kimi/.ready
