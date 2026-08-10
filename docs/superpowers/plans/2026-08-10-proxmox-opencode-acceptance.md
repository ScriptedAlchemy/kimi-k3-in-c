# Proxmox LXC and OpenCode Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the exact-weight OpenAI service reproducibly in CT106, publish the vetted fork on `main`, and prove streamed reasoning plus a function-tool round trip from OpenCode while ubuntu-main stays configured at 128 GiB.

**Architecture:** Version reusable host/guest scripts in the fork and mirror them into Proxmox snippets. The installer follows fork `main` by default, records the exact resolved SHA, refuses dirty upgrades, keeps the 1.8 TiB Btrfs/zstd model volume in-place with `backup=0`, and installs systemd units for model preparation and the authenticated OpenAI server. Acceptance verifies the immutable official Hugging Face revision, 96 shard checksums, packed BF16 trunk, exact one-token inference, HTTP/SSE semantics, OpenCode tool execution, and persistent VM100 memory configuration.

**Tech Stack:** Proxmox VE `pct`/`qm`, unprivileged Debian LXC, Btrfs zstd, systemd, Bash/ShellCheck, Git/GitHub, curl, OpenCode, OpenAI-compatible SSE.

## Global Constraints

- CT106 stays `kimi-inference` with 160 GiB RAM and zero swap.
- VM100 `ubuntu-main` persistent `memory` and `balloon` remain exactly 131072 MiB.
- Do not reboot VM100 merely to prove its persistent limit.
- Do not create Proxmox guest backups, vzdump jobs, or backup archives.
- Model mount remains Btrfs with zstd compression, `backup=0`, and `replicate=0`.
- Never delete, move, reformat, redownload, or convert the verified official checkpoint during code deployment.
- Production model preparation must not set `K3_SKIP_CHECKSUM=1`.
- Fork `main` is published only after source, runtime, and server matrices are green.

---

### Task 1: Version and test the reusable LXC installer

**Files:**
- Create: `deploy/proxmox/provision-kimi-inference-lxc.sh`
- Create: `deploy/proxmox/setup-kimi-inference-guest.sh`
- Create: `deploy/proxmox/kimi-k3`
- Create: `deploy/proxmox/kimi-k3-download`
- Create: `deploy/proxmox/kimi-k3-doctor`
- Create: `deploy/proxmox/kimi-k3-profile.sh`
- Create: `deploy/proxmox/kimi-k3-prepare.service`
- Create: `deploy/proxmox/kimi-k3-openai.service`
- Create: `tests/deploy/test_proxmox_scripts.sh`

**Interfaces:**
- Consumes: standard Debian LXC archive and fork Git repository
- Produces: fail-closed CT provisioner and idempotent guest updater

- [ ] **Step 1: Copy the already-proven deployment artifacts into the repo**

Use the current output scripts as the source, preserving CT defaults:

```text
CTID=106
CT_HOSTNAME=kimi-inference
MEMORY_MIB=163840
SWAP=0
CORES=120
MODEL_GIB=1800
MODEL_MOUNT=/srv/kimi
```

- [ ] **Step 2: Write static installer tests first**

`tests/deploy/test_proxmox_scripts.sh` asserts:

```bash
grep -q 'K3_REPO_URL.*ScriptedAlchemy/kimi-k3-in-c' deploy/proxmox/setup-kimi-inference-guest.sh
grep -q 'K3_REPO_REF.*main' deploy/proxmox/setup-kimi-inference-guest.sh
grep -q 'backup=0,replicate=0' deploy/proxmox/provision-kimi-inference-lxc.sh
grep -q 'compression zstd' deploy/proxmox/provision-kimi-inference-lxc.sh
grep -q '/etc/kimi-k3-build' deploy/proxmox/setup-kimi-inference-guest.sh
! grep -Eq 'git clean|checkout .*--force|vzdump|pbs' deploy/proxmox/*
```

Also run `bash -n` on every shell script and `systemd-analyze verify` on both units when available.

- [ ] **Step 3: Run and observe failures against current defaults**

Run:

```bash
bash tests/deploy/test_proxmox_scripts.sh
```

Expected: it fails because the guest script still pins the upstream commit and uses force/clean.

- [ ] **Step 4: Switch to fork-main defaults with exact-SHA recording**

Guest defaults become:

```bash
K3_REPO_URL="${K3_REPO_URL:-https://github.com/ScriptedAlchemy/kimi-k3-in-c.git}"
K3_REPO_REF="${K3_REPO_REF:-main}"
REPO_DIR="${K3_REPO_DIR:-/opt/kimi-k3-in-c}"
```

For an existing checkout, refuse changes before fetching:

```bash
test -z "$(git -C "$REPO_DIR" status --porcelain)" || {
  echo "refusing to update dirty checkout at $REPO_DIR" >&2
  exit 1
}
git -C "$REPO_DIR" remote set-url origin "$K3_REPO_URL"
git -C "$REPO_DIR" fetch --depth 1 origin "$K3_REPO_REF"
RESOLVED_SHA="$(git -C "$REPO_DIR" rev-parse FETCH_HEAD)"
git -C "$REPO_DIR" checkout --detach "$RESOLVED_SHA"
printf '%s\n' "$RESOLVED_SHA" > /etc/kimi-k3-build
```

Do not use `git clean` or `--force`.

- [ ] **Step 5: Preserve model data across code upgrades**

The guest updater may create missing `/srv/kimi/{model,trunk,cache,logs}` directories but must not run `rm`, `mv`, `mkfs`, `hf download`, or `pack-trunk` unless the operator explicitly starts `kimi-k3-prepare.service`. The provisioner may format only the newly allocated, not-yet-mounted CT model volume after proving CTID did not exist.

- [ ] **Step 6: Install the OpenAI service securely**

`kimi-k3-openai.service` contains:

```ini
[Unit]
Description=Exact-weight Kimi K3 OpenAI-compatible service
After=network-online.target kimi-k3-prepare.service
ConditionPathExists=/srv/kimi/logs/.ready

[Service]
Type=simple
User=kimi
Group=kimi
EnvironmentFile=/etc/kimi-k3/server.env
WorkingDirectory=/opt/kimi-k3-in-c
ExecStart=/usr/bin/python3 -m k3serve /srv/kimi/model --trunk /srv/kimi/trunk --tok /srv/kimi/model --host 0.0.0.0 --port 8000 --model-id kimi-k3-local --trunk-gb 110 --cache-gb 13 --ctx 4096 --max-tokens 512 --api-key-env K3_API_KEY --no-request-log
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

Guest setup creates `/etc/kimi-k3/server.env` as root mode 0640, group `kimi`, generating a 32-byte random key only when the file does not already exist. It never prints the key.

- [ ] **Step 7: Run script and unit verification**

Run:

```bash
bash tests/deploy/test_proxmox_scripts.sh
shellcheck deploy/proxmox/*.sh
```

Expected: syntax, static invariants, service verification, and ShellCheck pass.

- [ ] **Step 8: Commit installer versioning**

Run:

```bash
git add deploy/proxmox tests/deploy
git commit -m "feat(deploy): provision the exact-weight OpenAI LXC"
```

### Task 2: Make official-model preparation produce durable verification receipts

**Files:**
- Modify: `scripts/download-model.sh`
- Modify: `scripts/pack-trunk.sh`
- Modify: `deploy/proxmox/kimi-k3-download`
- Create: `tests/deploy/test_model_receipts.sh`

**Interfaces:**
- Consumes: official `moonshotai/Kimi-K3` immutable revision
- Produces: atomic model revision/checksum/trunk readiness receipts

- [ ] **Step 1: Write receipt tests before changing scripts**

The test uses stub `hf`, `find`, and small fixture files to assert:

```text
no receipt exists before checksum verification
.k3-revision contains exactly one 40-hex commit
.k3-checksum-verified contains the same commit
K3_SKIP_CHECKSUM=1 prevents production .ready creation
trunk receipt records trunk.bin bytes and 93 manifest layers
```

- [ ] **Step 2: Run and observe missing receipts**

Run:

```bash
bash tests/deploy/test_model_receipts.sh
```

Expected: failure because the current downloader prints the revision but does not persist it.

- [ ] **Step 3: Write model receipts atomically after verification**

After `hf cache verify` succeeds:

```bash
printf '%s\n' "$REVISION" > "$DEST/.k3-revision.tmp"
mv "$DEST/.k3-revision.tmp" "$DEST/.k3-revision"
printf '%s\n' "$REVISION" > "$DEST/.k3-checksum-verified.tmp"
mv "$DEST/.k3-checksum-verified.tmp" "$DEST/.k3-checksum-verified"
```

When checksum verification is skipped, remove neither model data nor old receipts; print that this invocation cannot produce a new verified-ready receipt and exit non-zero in production wrapper `kimi-k3-download`.

- [ ] **Step 4: Write a packed-trunk receipt**

After `trunk.bin` and `trunk.json` exist, parse the manifest and atomically write `/srv/kimi/trunk/.k3-trunk-ready` as one JSON object:

```json
{"layers":93,"bytes":108810000000,"source_revision":"40_hex_sha"}
```

Use the actual `stat` byte count; the number above shows the schema, not a constant.

- [ ] **Step 5: Gate `.ready` on all receipts**

The wrapper checks:

```bash
test "$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)" -eq 96
test "$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -printf '%s\n' | awk '{s+=$1} END{print s+0}')" -eq 1560936091448
test -s "$MODEL_DIR/.k3-revision"
cmp -s "$MODEL_DIR/.k3-revision" "$MODEL_DIR/.k3-checksum-verified"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["layers"] == 93 and d["bytes"] > 100_000_000_000' "$TRUNK_DIR/.k3-trunk-ready"
```

Only then create `/srv/kimi/logs/.ready`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
bash tests/deploy/test_model_receipts.sh
shellcheck scripts/download-model.sh scripts/pack-trunk.sh deploy/proxmox/kimi-k3-download
git add scripts/download-model.sh scripts/pack-trunk.sh deploy/proxmox/kimi-k3-download tests/deploy/test_model_receipts.sh
git commit -m "feat(model): persist exact checkpoint verification receipts"
```

### Task 3: Complete the local release matrix and publish fork main

**Files:**
- Modify: no production files unless a release gate exposes a defect

**Interfaces:**
- Consumes: source integration, runtime library, OpenAI service, deployment scripts
- Produces: green fork `main` and immutable release tag

- [ ] **Step 1: Run the full local gate**

Run:

```bash
make clean
make -j2 portable libk3
make test
python3 -m unittest discover -s tests/k3serve -t . -v
bash tests/deploy/test_proxmox_scripts.sh
bash tests/deploy/test_model_receipts.sh
shellcheck scripts/*.sh deploy/proxmox/*.sh
git diff --check
```

Expected: every C, Python, installer, receipt, and static gate passes.

- [ ] **Step 2: Prove exact output after all integration**

Run:

```bash
./bin/k3_model tests/fixtures | tee build/release-oracle.log
rg "32/32|20/20|ENGINE MATCHES THE REFERENCE EXACTLY" build/release-oracle.log
```

Expected: all exact-token receipts are present.

- [ ] **Step 3: Push the integration branch and inspect GitHub checks**

Run:

```bash
git push fork integration/exact-chat-tools
gh run list --repo ScriptedAlchemy/kimi-k3-in-c --branch integration/exact-chat-tools --limit 10
```

Wait for the matching head SHA; do not use a green run from another commit.

- [ ] **Step 4: Fast-forward fork main only from the vetted SHA**

Run:

```bash
RELEASE_SHA="$(git rev-parse integration/exact-chat-tools)"
git push fork "$RELEASE_SHA:refs/heads/main"
git tag -a v1.0.0-openai-mvp.1 "$RELEASE_SHA" -m "Exact-weight Kimi K3 OpenAI MVP"
git push fork v1.0.0-openai-mvp.1
gh api repos/ScriptedAlchemy/kimi-k3-in-c/commits/main --jq .sha
```

Expected: remote `main` equals `RELEASE_SHA` exactly.

### Task 4: Audit CT106 and VM100 before deploying code

**Files:**
- Modify: no files

**Interfaces:**
- Consumes: Proxmox host state
- Produces: pre-deploy safety receipt

- [ ] **Step 1: Inspect guest configs and storage without mutation**

Run from the local machine:

```bash
ssh gthost-tor-pve-public 'pct config 106; qm config 100; pvesm status; zpool list 2>/dev/null || true; pct status 106; qm status 100'
```

Expected CT106: 163840 MiB memory, zero swap, model mount at `/srv/kimi`, `backup=0`, `replicate=0`. Expected VM100: both `memory` and `balloon` are 131072.

- [ ] **Step 2: Inspect compression and mount type**

Run:

```bash
ssh gthost-tor-pve-public 'pct exec 106 -- findmnt -no FSTYPE,OPTIONS /srv/kimi; pid=$(lxc-info -n 106 -pH); nsenter -t "$pid" -m -r -- btrfs property get /srv/kimi compression'
```

Expected: Btrfs mount and `compression=zstd`.

- [ ] **Step 3: Prove no guest-backup jobs are enabled**

Run:

```bash
ssh gthost-tor-pve-public 'pvesh get /cluster/backup --output-format json; systemctl list-timers --all | grep -E "vzdump|pbs" || true'
```

Expected: no scheduled Proxmox guest backup job targets CT106 or VM100. Do not alter host metadata/config backup services, which are outside guest backup scope.

- [ ] **Step 4: Inspect live preparation state**

Run:

```bash
ssh gthost-tor-pve-public 'pct exec 106 -- systemctl is-active kimi-k3-prepare.service; pct exec 106 -- journalctl -u kimi-k3-prepare.service -n 80 --no-pager; pct exec 106 -- sh -c "find /srv/kimi/model -maxdepth 1 -name \"*.safetensors\" | wc -l; du -sb /srv/kimi/model /srv/kimi/trunk"'
```

If preparation is active, leave it untouched and deploy only after its verification/packing finishes.

### Task 5: Deploy the vetted fork without touching model bytes

**Files:**
- Modify: Proxmox snippets and CT106 code installation

**Interfaces:**
- Consumes: fork `main` release SHA and existing verified model/trunk
- Produces: CT106 running that exact SHA and installed service

- [ ] **Step 1: Sync installer artifacts to Proxmox snippets with hashes**

Run:

```bash
scp deploy/proxmox/* gthost-tor-pve-public:/var/lib/vz/snippets/
sha256sum deploy/proxmox/* > build/proxmox-local.sha256
ssh gthost-tor-pve-public 'cd /var/lib/vz/snippets && sha256sum provision-kimi-inference-lxc.sh setup-kimi-inference-guest.sh kimi-k3 kimi-k3-download kimi-k3-doctor kimi-k3-profile.sh kimi-k3-prepare.service kimi-k3-openai.service'
```

Compare every hash before executing setup.

- [ ] **Step 2: Run guest updater against fork main**

Run:

```bash
ssh gthost-tor-pve-public 'pct push 106 /var/lib/vz/snippets/setup-kimi-inference-guest.sh /root/.kimi-bootstrap/setup-kimi-inference-guest.sh --perms 0700; pct exec 106 -- env K3_REPO_URL=https://github.com/ScriptedAlchemy/kimi-k3-in-c.git K3_REPO_REF=main /root/.kimi-bootstrap/setup-kimi-inference-guest.sh'
```

Expected: dirty-checkout check passes, code builds/tests, model data is not modified, and `/etc/kimi-k3-build` equals fork main.

- [ ] **Step 3: Verify installed commit and model inode stability**

Before and after setup, record:

```bash
find /srv/kimi/model -maxdepth 1 -name '*.safetensors' -printf '%f %s %i\n' | sort | sha256sum
```

Expected: identical hash, proving setup did not replace source weights.

### Task 6: Finish official preparation and run exact-weight smoke tests

**Files:**
- Modify: readiness receipts only

**Interfaces:**
- Consumes: active or completed preparation job
- Produces: verified 96-shard model, packed trunk, doctor, exact one-token receipt

- [ ] **Step 1: Wait for preparation with bounded status checks**

Every status check records service state, shard count, byte total, and trunk bytes. Completion requires `systemctl is-active` to return inactive and `systemctl show -p Result` to report success.

- [ ] **Step 2: Verify model and trunk receipts independently**

Run:

```bash
ssh gthost-tor-pve-public 'pct exec 106 -- sh -ceu '\''
test "$(find /srv/kimi/model -maxdepth 1 -name "*.safetensors" | wc -l)" -eq 96
test "$(find /srv/kimi/model -maxdepth 1 -name "*.safetensors" -printf "%s\n" | awk "{s+=\$1} END{print s+0}")" -eq 1560936091448
test "$(cat /srv/kimi/model/.k3-revision)" = "$(cat /srv/kimi/model/.k3-checksum-verified)"
python3 -c "import json; d=json.load(open(\"/srv/kimi/trunk/.k3-trunk-ready\")); assert d[\"layers\"] == 93 and d[\"bytes\"] == __import__(\"os\").stat(\"/srv/kimi/trunk/trunk.bin\").st_size"
test -f /srv/kimi/logs/.ready
'\'''
```

- [ ] **Step 3: Run doctor and one-token exact model smoke**

Run:

```bash
ssh gthost-tor-pve-public 'pct exec 106 -- kimi-k3-doctor; pct exec 106 -- kimi-k3 --ids 163584 --gen 1 --incremental --out /srv/kimi/logs/exact-smoke.json'
```

Expected: process exits 0, one token is present in `generated_ids`, and no expert drop is reported.

### Task 7: Start the authenticated service and prove OpenCode tools

**Files:**
- Create: `/srv/kimi/logs/opencode-acceptance.jsonl` in CT106
- Create: local acceptance transcript under `outputs/`

**Interfaces:**
- Consumes: ready model, installed service, OpenCode client
- Produces: streamed reasoning and two-turn function-tool proof

- [ ] **Step 1: Start and inspect the service without exposing its key**

Run:

```bash
ssh gthost-tor-pve-public 'pct exec 106 -- systemctl enable --now kimi-k3-openai.service; pct exec 106 -- systemctl is-active kimi-k3-openai.service; pct exec 106 -- ss -ltnp | grep :8000; pct exec 106 -- curl -fsS http://127.0.0.1:8000/health'
```

Expected: active service, port 8000, healthy model ID. The unauthenticated health route may remain public; chat requires bearer auth.

- [ ] **Step 2: Prove blocking and streamed response shapes**

From inside CT106, read the key into a shell variable without echoing it and call `/v1/chat/completions`. Verify blocking JSON has `reasoning_content` and `content`; verify SSE ends with `data: [DONE]` and reconstructed deltas match the blocking text for a deterministic seed.

- [ ] **Step 3: Configure OpenCode as an OpenAI-compatible provider**

Copy `examples/opencode.jsonc` to an isolated acceptance config directory and inject `K3_API_KEY` only through the process environment. Do not write the key into the config or transcript.

- [ ] **Step 4: Execute a deterministic local tool round trip**

Expose a tool named `read_acceptance_number` that returns `{"number": 1729}`. Ask Kimi to call the tool and state the returned number. Capture JSONL events and assert:

```text
at least one streamed reasoning delta is non-empty
assistant finish_reason is tool_calls
function name is read_acceptance_number
function.arguments parses as a JSON object
tool result is associated with the emitted tool_call_id
final assistant content contains 1729
no response field contains an XTML control marker
```

- [ ] **Step 5: Save a redacted acceptance transcript**

The transcript records timestamps, fork SHA, model revision, request IDs, reasoning/content presence, tool name/arguments/result, finish reasons, and pass/fail assertions. It excludes prompt bodies that contain secrets and excludes the bearer key entirely.

### Task 8: Final completion audit

**Files:**
- Create: `outputs/research_kimi_k3_local_serving_20260810.md`
- Create: `outputs/kimi-k3-deployment-receipt.md`

**Interfaces:**
- Consumes: every prior task receipt
- Produces: requirement-by-requirement proof and final user deliverables

- [ ] **Step 1: Re-check fork and installed SHA**

Verify GitHub fork `main`, `/etc/kimi-k3-build`, and local release tag all name the same SHA.

- [ ] **Step 2: Re-check persistent memory and storage invariants**

Verify:

```text
VM100 memory=131072 and balloon=131072
CT106 memory=163840 and swap=0
/srv/kimi is Btrfs with zstd
model mount backup=0 and replicate=0
no Proxmox guest backup job targets CT106
```

- [ ] **Step 3: Re-run release gates at the installed SHA**

Run CT106 weightless tests, tokenizer parity, exact oracle, server tests, model receipts, one-token smoke, health, SSE, and OpenCode tool assertions. Record exact exit codes.

- [ ] **Step 4: Publish research and deployment receipts**

The research report compares the Fareed engine/fork path with llama.cpp PR #26185, WARP, Deltafin, Colibri, and GPU frameworks, and explains why exact official safetensors plus the adapted API is the best fit for this host. The deployment receipt lists SHAs, model revision, shard/byte checks, trunk bytes, service status, OpenCode assertions, storage/compression, backup exclusion, and VM100 memory proof.

- [ ] **Step 5: Mark the active goal complete only after every receipt is proven**

If any explicit requirement lacks authoritative evidence, keep the goal active and continue that requirement. If all are proven, update the goal to complete and report the final receipts.
