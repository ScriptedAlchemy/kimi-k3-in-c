#!/usr/bin/env bash
# Idempotent installer for a Debian/Ubuntu Proxmox LXC guest.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/ScriptedAlchemy/kimi-k3-in-c.git}
REPO_REF=${REPO_REF:-main}
INSTALL_DIR=${INSTALL_DIR:-/opt/kimi-k3-in-c}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${EUID} -ne 0 ]]; then
    echo "run as root inside the LXC guest" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git libgomp1 python3

if ! getent group kimi >/dev/null; then
    groupadd --system kimi
fi
if ! id kimi >/dev/null 2>&1; then
    useradd --system --gid kimi --home-dir /srv/kimi --shell /usr/sbin/nologin kimi
fi

if [[ -e "${INSTALL_DIR}/.git" ]]; then
    if [[ -n $(git -C "${INSTALL_DIR}" status --porcelain) ]]; then
        echo "refusing to replace dirty checkout at ${INSTALL_DIR}" >&2
        exit 1
    fi
    git -C "${INSTALL_DIR}" remote set-url origin "${REPO_URL}"
    git -C "${INSTALL_DIR}" fetch --prune origin "${REPO_REF}"
    git -C "${INSTALL_DIR}" checkout -B "${REPO_REF}" FETCH_HEAD
else
    if [[ -e "${INSTALL_DIR}" ]]; then
        echo "refusing to replace non-git path at ${INSTALL_DIR}" >&2
        exit 1
    fi
    git clone --branch "${REPO_REF}" --single-branch "${REPO_URL}" "${INSTALL_DIR}"
fi

make -C "${INSTALL_DIR}" -j"$(nproc)" server-test
install -m 0644 "${INSTALL_DIR}/deploy/proxmox/k3serve.service" \
    /etc/systemd/system/k3serve.service
if [[ ! -e /etc/kimi-k3-inference.env ]]; then
    install -m 0600 "${INSTALL_DIR}/deploy/proxmox/k3serve.env" \
        /etc/kimi-k3-inference.env
fi

systemctl daemon-reload
systemctl enable k3serve.service
if [[ -e /srv/kimi/logs/.ready ]]; then
    systemctl restart k3serve.service
else
    echo "model is not finalized; service is enabled but remains stopped"
    echo "run: ${INSTALL_DIR}/deploy/proxmox/finalize-model.sh"
fi

echo "installed source commit: $(git -C "${INSTALL_DIR}" rev-parse HEAD)"
echo "configuration: /etc/kimi-k3-inference.env"
