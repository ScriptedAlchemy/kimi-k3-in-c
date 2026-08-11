# Proxmox LXC deployment

Inside the prepared guest, clone the fork and run one idempotent installer:

```bash
git clone https://github.com/ScriptedAlchemy/kimi-k3-in-c.git /opt/kimi-k3-in-c
sudo /opt/kimi-k3-in-c/deploy/proxmox/install-in-guest.sh
```

The installer builds and tests `libk3.so`, installs `k3serve.service`, and refuses to
overwrite a dirty checkout. It does not create a VM backup, LXC backup, archive, or
compressed template. The model remains a normal directory on the guest's Proxmox-backed
`/srv/kimi` mount, so the backing Btrfs/ZFS dataset retains filesystem compression.
The service starts only after `/srv/kimi/logs/.ready` records a completed preparation.

After the resumable Hugging Face download has all 96 official shards, run:

```bash
sudo /opt/kimi-k3-in-c/deploy/proxmox/finalize-model.sh
```

Edit `/etc/kimi-k3-inference.env` for memory, threads, context, or networking. Loopback
is the safe default. To serve another machine, set `K3_HOST=0.0.0.0` and a non-empty
`K3_API_KEY`; non-loopback startup without a key is refused.

The shipped 200 GiB profile uses a 16K context, the full 110 GiB packed-trunk
budget, a 13 GiB expert cache, and 80 OpenMP threads. The 80-thread cap leaves
Proxmox control-plane headroom on a 120-thread host.

Kimi K3 advertises a 1M-token architecture, but that is not a workable setting for
this deployment: the expanded MLA KV cache alone is about 2.5 TB at 1,048,576
positions, and the current C runtime caps prompts at 32K with unchunked prefill.

OpenAI clients use:

```text
base URL: http://127.0.0.1:8000/v1
model:    kimi-k3
API key:  value of K3_API_KEY, if configured
```

The installer also writes the official OpenCode 1 custom-provider shape to
`/home/kimi/.config/opencode/opencode.json` when that file does not already exist. It
uses `@ai-sdk/openai-compatible`, model `kimi-local/kimi-k3`, a 16K context, and the
local `/v1/chat/completions` endpoint. Existing OpenCode configuration is preserved.
The optional `kimi-proof` primary agent exposes only OpenCode's `bash` tool, keeping
its tool declaration small enough for a practical end-to-end CPU inference check.
