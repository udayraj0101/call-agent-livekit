"""
VPS-optimised inbound agent — deploy this on a Linux server (Mumbai region).

Identical to agent.py in every way EXCEPT turn detection:
- Uses MultilingualModel semantic endpointing instead of pure silence threshold
- min_delay=0.1 (safe because the model guards false triggers)
- endpointing_ms=300 (Deepgram; MultilingualModel fires before this anyway)

On Linux the MultilingualModel runs in ~50ms (vs ~220ms on Windows), so it
saves 150-350ms per turn compared to agent.py's silence-only approach.

Expected agent-side latency on Mumbai VPS: 500-650ms per turn
(vs 750-1000ms on Windows with silence endpointing)

Run with:
    uv run python vps_agent.py start

Same .env as agent.py — no extra env vars needed.
"""

import asyncio
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from typing import AsyncGenerator
from dotenv import load_dotenv
import openai

from livekit import agents, api, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import deepgram, noise_cancellation, silero
from livekit.plugins import openai as lk_openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tts import build_tts
from tools import create_tools

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vps-agent")


# ---------------------------------------------------------------------------
# Config — identical to agent.py
# ---------------------------------------------------------------------------

AGENT_NAME                = "inbound-caller"

LLM_TEMPERATURE           = 0.3
GROQ_API_KEY              = os.getenv("GROQ_API_KEY")
GROQ_MODEL                = "llama-3.3-70b-versatile"
STT_MODEL                 = "nova-3"
STT_LANGUAGE              = "multi"
TTS_LANGUAGE              = "hi"
MAX_CALL_DURATION_SECONDS = 600

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "cartesia").lower()
if TTS_PROVIDER == "openai":
    TTS_MODEL = "tts-1"
    TTS_VOICE = "nova"
else:
    TTS_MODEL = "sonic-3.5"
    TTS_VOICE = "95d51f79-c397-46f9-b49a-23763d3eaa2d"  # Arushi — Hinglish Speaker

FIRST_MESSAGE = (
    "नमस्ते, मैं आरुषि बोल रही हूँ Aspirantive से। "
    "क्या आपके पास एक quick minute है?"
)

_GREETING_SAMPLE_RATE = 24000

SYSTEM_PROMPT = (
    "You are Arushi from Aspirantive, a friendly sales rep calling pharma/distribution businesses in India. "
    "Speak in natural Hinglish (Hindi + English mix) — warm, conversational, phone-call style.\n"
    "Goal: book a 15-minute discovery call about CRM/ERP automation.\n"
    "Reply in 2-3 sentences max. Explain clearly but stay concise — this is a phone call, not a presentation. "
    "Guide conversation: understand their order process → explain how automation saves time and reduces errors → offer a 15-min call to show it live.\n"
    "If they ask about Aspirantive: explain in 1-2 lines — 'Hum pharma aur distribution businesses ke liye CRM automation provide karte hain. "
    "Aapke orders, inventory, aur billing sab ek jagah automatic ho jaata hai.' Then steer to booking.\n"
    "If you don't understand what they said: 'Sorry, mujhe clearly nahi suna — kya aap phir se bol sakte hain?'"
)


class SampleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

async def _end_call_on_llm_error(session: AgentSession, ctx: agents.JobContext, err: Exception) -> None:
    logger.error(f"LLM error during call — hanging up gracefully: {err}")
    try:
        await session.say(
            "I'm sorry, I'm having a technical issue. Please try again in a few minutes. Goodbye!"
        )
        await asyncio.sleep(4)
    except Exception:
        pass
    try:
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception:
        pass
    ctx.shutdown()


async def _enforce_max_duration(session: AgentSession, ctx: agents.JobContext, max_seconds: int) -> None:
    await asyncio.sleep(max_seconds)
    logger.warning(f"Max call duration ({max_seconds}s) reached — ending call")
    try:
        await session.say("I need to wrap up our call now. Thank you for your time. Goodbye!")
        await asyncio.sleep(4)
    except Exception:
        pass
    try:
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception as e:
        logger.warning(f"Could not delete room on timeout: {e}")
    ctx.shutdown()


# ---------------------------------------------------------------------------
# Greeting prerender
# ---------------------------------------------------------------------------

def _prerender_greeting() -> bytes | None:
    if TTS_PROVIDER != "cartesia":
        return None
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        return None

    body = json.dumps({
        "model_id": TTS_MODEL,
        "transcript": FIRST_MESSAGE,
        "voice": {"mode": "id", "id": TTS_VOICE},
        "language": TTS_LANGUAGE,
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": _GREETING_SAMPLE_RATE,
        },
    }).encode()
    req = urllib.request.Request(
        "https://api.cartesia.ai/tts/bytes",
        data=body,
        headers={
            "X-API-Key": api_key,
            "Cartesia-Version": "2025-04-16",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    time.sleep(random.uniform(0, 1.5))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                seconds = len(data) / (_GREETING_SAMPLE_RATE * 2)
                logger.info(f"Greeting prerendered: {len(data)} bytes (~{seconds:.1f}s of audio)")
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                wait = 1.5 ** (attempt + 1) + random.uniform(0, 0.5)
                logger.info(f"Cartesia 429 during prerender, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            logger.warning(f"Greeting prerender failed (will fall back to live TTS): HTTP {e.code} {e.reason}")
            return None
        except Exception as e:
            logger.warning(f"Greeting prerender failed (will fall back to live TTS): {e}")
            return None
    return None


async def _warm_llm() -> None:
    try:
        client = openai.AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
        )
        await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        logger.info("Groq LLM connection warmed")
    except Exception as e:
        logger.warning(f"Groq LLM warmup failed (non-fatal): {e}")


async def _stream_cached_greeting(pcm_bytes: bytes) -> AsyncGenerator[rtc.AudioFrame, None]:
    samples_per_frame = int(_GREETING_SAMPLE_RATE * 0.02)
    bytes_per_frame = samples_per_frame * 2
    for i in range(0, len(pcm_bytes), bytes_per_frame):
        chunk = pcm_bytes[i:i + bytes_per_frame]
        if len(chunk) < bytes_per_frame:
            chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
        yield rtc.AudioFrame(
            data=chunk,
            sample_rate=_GREETING_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=samples_per_frame,
        )


# ---------------------------------------------------------------------------
# Prewarm — MultilingualModel is instantiated here so it loads once per
# worker process, not once per call.
# ---------------------------------------------------------------------------

def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"]          = silero.VAD.load(min_silence_duration=0.4)
    proc.userdata["stt"]          = deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300, keyterm=["Aspirantive", "Arushi"])
    proc.userdata["llm"]          = lk_openai.LLM(
        model=GROQ_MODEL,
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=120,
    )
    proc.userdata["tts"]          = build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE)
    proc.userdata["turn_model"]   = MultilingualModel()
    proc.userdata["greeting_pcm"] = _prerender_greeting()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
    t0 = time.monotonic()
    def ts(label: str) -> None:
        logger.info(f"[+{(time.monotonic()-t0)*1000:.0f}ms] {label}")

    logger.info(f"Inbound call (VPS) — connecting to room: {ctx.room.name}")
    logger.info(
        f"Config — llm=groq/{GROQ_MODEL} tts={TTS_PROVIDER}/{TTS_MODEL}/{TTS_VOICE} "
        f"stt={STT_MODEL}/{STT_LANGUAGE} turn_detection=MultilingualModel"
    )

    ud = ctx.proc.userdata
    session = AgentSession(
        vad=ud.get("vad") or silero.VAD.load(min_silence_duration=0.4),
        stt=ud.get("stt") or deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300, keyterm=["Aspirantive", "Arushi"]),
        llm=ud.get("llm") or lk_openai.LLM(
            model=GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=120,
        ),
        tts=ud.get("tts") or build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE),
        tools=create_tools(ctx),
        turn_handling={
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
            "turn_detection": ud.get("turn_model") or MultilingualModel(),
            "endpointing": {"min_delay": 0.1, "max_delay": 4.0},
        },
    )
    greeting_pcm: bytes | None = ud.get("greeting_pcm")

    # STT gap detector — fires when Deepgram drops a user utterance entirely.
    _last_speech_end     = [0.0]
    _last_agent_thinking = [0.0]
    _agent_listening     = [True]

    @session.on("user_state_changed")
    def _on_user_state(ev):
        ts(f"user_state: {ev.old_state} → {ev.new_state}")
        if ev.old_state == "speaking" and ev.new_state == "listening":
            _last_speech_end[0] = time.monotonic()

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        ts(f"agent_state: {ev.old_state} → {ev.new_state}")
        _agent_listening[0] = (ev.new_state == "listening")
        if ev.new_state == "thinking":
            _last_agent_thinking[0] = time.monotonic()
        elif ev.new_state == "speaking":
            _last_speech_end[0] = 0.0

    @session.on("user_input_transcribed")
    def _on_transcribed(ev):
        tag = "final" if getattr(ev, "is_final", False) else "interim"
        text = getattr(ev, "transcript", "") or ""
        ts(f"stt ({tag}): {text!r}")

    async def _stt_gap_watchdog():
        while True:
            await asyncio.sleep(2)
            speech_end = _last_speech_end[0]
            if speech_end <= 0:
                continue
            gap = time.monotonic() - speech_end
            if gap > 2.5 and _last_agent_thinking[0] < speech_end and _agent_listening[0]:
                logger.warning(f"STT gap: {gap:.1f}s after user stopped, no agent response — prompting repeat")
                _last_speech_end[0] = 0.0
                try:
                    await session.say("Sorry, mujhe clearly nahi suna — kya aap phir se bol sakte hain?")
                except Exception:
                    pass

    ts("session.start begin")
    await session.start(
        room=ctx.room,
        agent=SampleAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )
    ts("session.start done")

    asyncio.create_task(_enforce_max_duration(session, ctx, MAX_CALL_DURATION_SECONDS))
    asyncio.create_task(_warm_llm())
    asyncio.create_task(_stt_gap_watchdog())

    last_activity = [time.monotonic()]

    @session.on("user_state_changed")
    def _track_user_activity(ev):
        last_activity[0] = time.monotonic()

    @session.on("agent_state_changed")
    def _track_agent_activity(ev):
        last_activity[0] = time.monotonic()

    async def _silence_watchdog():
        while True:
            await asyncio.sleep(5)
            idle = time.monotonic() - last_activity[0]
            if idle > 35:
                logger.warning(f"Both parties silent for {idle:.0f}s — auto-ending call")
                try:
                    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
                except Exception as e:
                    logger.warning(f"Could not delete room on idle: {e}")
                return

    asyncio.create_task(_silence_watchdog())

    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=10.0)
        ts(f"participant joined: {participant.identity}")
    except asyncio.TimeoutError:
        logger.error("No SIP participant joined within 10s — aborting call.")
        try:
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:
            pass
        ctx.shutdown()
        return
    except Exception as e:
        logger.warning(f"wait_for_participant failed, greeting anyway: {e}")

    try:
        if greeting_pcm:
            ts(f"session.say begin (cached, {len(greeting_pcm)} bytes)")
            await session.say(FIRST_MESSAGE, audio=_stream_cached_greeting(greeting_pcm))
        else:
            ts("session.say begin (live TTS)")
            await session.say(FIRST_MESSAGE)
        ts("session.say done")
    except openai.RateLimitError as e:
        await _end_call_on_llm_error(session, ctx, e)
        return

    _room_closed = asyncio.Event()

    @ctx.room.on("disconnected")
    def _on_room_disconnected(*_):
        _room_closed.set()

    try:
        await asyncio.wait_for(_room_closed.wait(), timeout=MAX_CALL_DURATION_SECONDS + 120)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
            port=int(os.getenv("AGENT_HTTP_PORT", "8081")),
        )
    )
