import { tool } from "@opencode-ai/plugin"

export default tool({
  description: "Return the Kimi tool-call proof marker.",
  args: {},
  async execute() {
    return "KIMI_TOOL_OK"
  },
})
