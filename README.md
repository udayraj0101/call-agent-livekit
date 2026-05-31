# call-agent-livekit

Inbound LiveKit voice agent for Aspirantive (Hindi/Hinglish sales calls).

## Files

| File | Purpose |
|---|---|
| `agent.py` | Production agent — Cartesia (Arushi) TTS + Deepgram nova-3 STT + OpenAI gpt-4o-mini LLM. ~1.5s avg turn latency. |
| `azure_agent.py` | Parallel build using Azure OpenAI (Central India region). Same Cartesia voice. Use when client provides Azure account with India quota. |
| `gemini_agent.py` | Experimental build using Gemini Live Realtime API. Different voice. Higher latency on free tier. |
| `tools.py` | LLM function tools — `end_call`, `transfer_to_human`. |
| `tts.py` | TTS factory (Cartesia for production, OpenAI fallback). |
| `ecosystem.config.js` | PM2 process config for VPS deployment. |
| `.env.example` | Template for environment variables. Copy to `.env` and fill in real keys. |

## Local development

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env
# fill in .env with real API keys
uv run python agent.py start
```

Worker registers with LiveKit Cloud and waits for inbound calls. Place a call to your LiveKit SIP DID to test.

## VPS deployment (see steps in deployment notes)

```bash
# On VPS, after Python/uv/pm2 installed:
git clone https://github.com/udayraj0101/call-agent-livekit.git
cd call-agent-livekit
uv venv
uv pip install -r requirements.txt
# copy .env over (scp from local, do NOT commit)
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # auto-restart on reboot
```

## Architecture notes

- LiveKit agents are **outbound clients** — they open a WebSocket to LiveKit Cloud and accept dispatched jobs. No inbound HTTP, no nginx, no public IP needed.
- The agent registers under `agent_name: "inbound-caller"`. This must match the agent_name in your LiveKit SIP Dispatch Rule.
- Greeting audio is prerendered via Cartesia REST and cached in `.cache/` for instant first-word playback.
