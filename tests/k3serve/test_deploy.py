"""Deployment-template checks that run as part of ``make server-test``."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROXMOX = ROOT / "deploy" / "proxmox"


class OpenCodeDeploymentTests(unittest.TestCase):
    def test_proof_agent_exposes_only_the_minimal_custom_tool(self) -> None:
        config = json.loads((PROXMOX / "opencode.json").read_text())
        agent = config["agent"]["kimi-proof"]

        self.assertEqual(agent["model"], "kimi-local/kimi-k3")
        self.assertEqual(agent["permission"], {"*": "deny", "proof": "allow"})
        self.assertIn("proof tool", agent["prompt"])

    def test_installer_ships_the_global_proof_tool(self) -> None:
        tool = PROXMOX / "opencode-tools" / "proof.ts"
        source = tool.read_text()
        installer = (PROXMOX / "install-in-guest.sh").read_text()

        self.assertIn("args: {}", source)
        self.assertIn('return "KIMI_TOOL_OK"', source)
        self.assertIn("/home/kimi/.config/opencode/tools", installer)
        self.assertIn("opencode-tools/proof.ts", installer)


if __name__ == "__main__":
    unittest.main()
