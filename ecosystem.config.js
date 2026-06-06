// PM2 process definition for the LiveKit call agent.
// Run from the repo root on the VPS:
//   pm2 start ecosystem.config.js
//
// Notes:
// - LiveKit agents are *outbound* — they open a WebSocket to LiveKit Cloud
//   and wait for job dispatches. No inbound ports, no nginx, no public IP.
// - `script` points at the venv Python so PM2 doesn't try to invoke node.
// - `args: "agent.py start"` — the trailing `start` is required by the
//   livekit-agents CLI to launch the worker (vs. `dev`, `console`, etc.).
// - To run gemini_agent.py or azure_agent.py instead, change `args`.

module.exports = {
  apps: [
    {
      name: "call-agent",
      cwd: __dirname,
      script: "./.venv/bin/python",
      args: "vps_agent.py start",
      interpreter: "none",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        PYTHONUNBUFFERED: "1",
        // Override default 8081 — taken by the call-agent-saas worker on this VPS.
        // agent.py reads this and passes it to WorkerOptions(port=...).
        AGENT_HTTP_PORT: "8082",
        // livekit.plugins.openai validates this key in inference subprocesses before
        // user code (dotenv) runs on Linux. We use Groq, so any non-empty value works.
        OPENAI_API_KEY: "not-used-groq-only",
      },
      error_file: "./logs/error.log",
      out_file: "./logs/out.log",
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};
