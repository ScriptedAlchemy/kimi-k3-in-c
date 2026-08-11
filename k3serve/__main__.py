"""Run ``python3 -m k3serve``."""

from __future__ import annotations

import argparse
import os

from .engine import Engine
from .server import serve


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="OpenAI-compatible server for exact-weight Kimi K3")
    value.add_argument("--model", default="/srv/kimi/model")
    value.add_argument("--trunk", default="/srv/kimi/trunk")
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8000)
    value.add_argument("--model-id", default="kimi-k3")
    value.add_argument("--api-key", default=os.environ.get("K3_API_KEY"))
    value.add_argument("--context", type=int, default=8192)
    value.add_argument("--cache-gib", type=float, default=13.0)
    value.add_argument("--trunk-budget-gib", type=float, default=110.0)
    value.add_argument("--max-tokens", type=int, default=512)
    value.add_argument("--no-thinking", action="store_true")
    value.add_argument("--quiet", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    gib = 1024 ** 3
    with Engine(
            args.model,
            trunk_dir=args.trunk or None,
            tokenizer_dir=args.model,
            trunk_budget_bytes=int(args.trunk_budget_gib * gib),
            expert_cache_bytes=int(args.cache_gib * gib),
            context_tokens=args.context) as engine:
        server = serve(
            engine, host=args.host, port=args.port, model_id=args.model_id,
            api_key=args.api_key, default_max_tokens=args.max_tokens,
            default_thinking=not args.no_thinking,
            log_requests=not args.quiet)
        print(f"k3serve: {args.model_id} on http://{args.host}:{args.port}",
              flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
