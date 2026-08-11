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

The shipped 160 GiB profile keeps the 8K context while reserving 70 GiB for the
packed trunk prefix, 8 GiB for the expert cache, and 80 OpenMP threads. Pinning the
entire 108.8 GB trunk together with an 8K KV cache can exceed a 160 GiB container;
increase these budgets only after measuring the container's peak cgroup memory.

OpenAI clients use:

```text
base URL: http://127.0.0.1:8000/v1
model:    kimi-k3
API key:  value of K3_API_KEY, if configured
```

The installer also writes the official OpenCode 1 custom-provider shape to
`/home/kimi/.config/opencode/opencode.json` when that file does not already exist. It
uses `@ai-sdk/openai-compatible`, model `kimi-local/kimi-k3`, an 8K context, and the
local `/v1/chat/completions` endpoint. Existing OpenCode configuration is preserved.
