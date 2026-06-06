"""
Azure OpenAI (Central India) voice agent — parallel to agent.py.

Identical to the production agent.py in every way EXCEPT the LLM source:
- Voice: same Cartesia Arushi (sonic-3.5)        ← unchanged
- STT:   same Deepgram nova-3 / Hindi             ← unchanged
- VAD:   same Silero (prewarmed)                  ← unchanged
- Prompt: same Aspirantive Hindi/Hinglish script  ← unchanged
- Tools, watchdog, greeting cache                 ← unchanged
- LLM:   Azure OpenAI gpt-4o-mini in Central India ← NEW

Expected impact: 500-1000 ms saved per turn vs OpenAI direct, because the
HTTPS round-trip stays inside India instead of crossing to US East.

Run with:
    uv run python azure_agent.py start

Required .env vars (set these from the Azure portal):
    AZURE_OPENAI_ENDPOINT     = https://<your-resource-name>.openai.azure.com/
    AZURE_OPENAI_API_KEY      = <key 1 from "Keys and Endpoint" tab>
    AZURE_OPENAI_DEPLOYMENT   = <name you chose when deploying gpt-4o-mini>
    OPENAI_API_VERSION        = 2024-10-21  (or newer)
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
logger = logging.getLogger("azure-agent")


# ---------------------------------------------------------------------------
# Hardcoded sample config (mirrors agent.py for fair comparison)
# ---------------------------------------------------------------------------

AGENT_NAME                = "inbound-caller"

LLM_TEMPERATURE           = 0.7
STT_MODEL                 = "nova-3"
# "multi" handles code-switched Hindi+English (Hinglish) better than "hi" —
# observed 2026-06-02 that nova-3 + "hi" often misses finals on short
# utterances (<2s) and mistranscribes short Hindi like "कल सुबह 10 बजे"
# as "Hello?". multi is Deepgram's code-switching model and is more robust.
STT_LANGUAGE              = "multi"
# Cartesia / build_tts() need a real language code, not "multi" — keep Hindi
# explicitly for the greeting prerender and TTS pipeline.
TTS_LANGUAGE              = "hi"
MAX_CALL_DURATION_SECONDS = 600

# Azure OpenAI config — read from .env
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION  = os.getenv("OPENAI_API_VERSION", "2024-10-21")

# Cartesia TTS — same as agent.py
TTS_PROVIDER = "cartesia"
TTS_MODEL    = "sonic-3.5"
TTS_VOICE    = "95d51f79-c397-46f9-b49a-23763d3eaa2d"  # Arushi — Hinglish Speaker

FIRST_MESSAGE = (
    "नमस्ते, मैं आरुषि बोल रही हूँ Aspirantive से। "
    "क्या आपके पास एक quick minute है?"
)

_GREETING_SAMPLE_RATE = 24000

# Trimmed from ~250 tokens to ~100 tokens to cut LLM prefix-processing time
# (~150-300ms saved per turn). Every word here ships on every request.
SYSTEM_PROMPT = (
    "You are Arushi from Aspirantive, calling pharma/distribution businesses in India about CRM/ERP automation. "
    "Goal: book a 15-minute discovery call.\n\n"
    "Reply in natural Hinglish (Hindi + English). "
    "MAX 2 short sentences per reply — this is a phone call, be brief. "
    "Guide the conversation: understand how they currently manage orders, explain how automation helps, then offer to schedule a quick call."
)


class SampleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Azure LLM factory — single place we call lk_openai.LLM.with_azure(...)
# so the entrypoint + prewarm + fallback all share identical construction.
# ---------------------------------------------------------------------------

def _build_azure_llm():
    # max_completion_tokens caps reply length — TTS starts streaming sooner and
    # prevents the model from rambling past ~2 phone sentences.
    return lk_openai.LLM.with_azure(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=60,
    )


# ---------------------------------------------------------------------------
# Lifecycle helpers (same as agent.py)
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
# Greeting prerender — IDENTICAL to agent.py (uses Cartesia TTS REST).
# Voice and audio path unchanged; only the LLM is swapped to Azure.
# ---------------------------------------------------------------------------

def _prerender_greeting() -> bytes | None:
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
    """Warm the Azure HTTPS+TLS connection pool during greeting playback."""
    try:
        client = openai.AsyncAzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        await client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,  # Azure uses deployment name here
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        logger.info("Azure LLM connection warmed")
    except Exception as e:
        logger.warning(f"Azure LLM warmup failed (non-fatal): {e}")


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
# Prewarm — Silero VAD + Deepgram STT + Azure LLM + Cartesia TTS instances,
# plus greeting prerender. Only the LLM construction differs from agent.py.
# ---------------------------------------------------------------------------

def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)
    proc.userdata["stt"] = deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300)
    proc.userdata["llm"] = _build_azure_llm()
    proc.userdata["tts"] = build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE)
    proc.userdata["greeting_pcm"] = _prerender_greeting()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
    # Fail fast if Azure env vars are missing — better than a confusing 401
    # mid-call.
    missing = [
        name for name, value in [
            ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
            ("AZURE_OPENAI_DEPLOYMENT", AZURE_OPENAI_DEPLOYMENT),
        ] if not value
    ]
    if missing:
        logger.error(f"Missing required Azure env vars: {missing} — aborting call")
        ctx.shutdown()
        return

    t0 = time.monotonic()
    def ts(label: str) -> None:
        logger.info(f"[+{(time.monotonic()-t0)*1000:.0f}ms] {label}")

    logger.info(f"Inbound call (Azure India) — connecting to room: {ctx.room.name}")
    logger.info(
        f"Config — llm=azure/{AZURE_OPENAI_DEPLOYMENT} "
        f"tts={TTS_PROVIDER}/{TTS_MODEL}/{TTS_VOICE} stt={STT_MODEL}/{STT_LANGUAGE}"
    )

    ud = ctx.proc.userdata
    session = AgentSession(
        vad=ud.get("vad") or silero.VAD.load(min_silence_duration=0.4),
        stt=ud.get("stt") or deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300),
        llm=ud.get("llm") or _build_azure_llm(),
        tts=ud.get("tts") or build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE),
        tools=create_tools(ctx),
        turn_handling={
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
            # MultilingualModel predicts semantic end-of-turn before Deepgram's
            # 300ms silence expires — saves ~200-300ms vs pure silence endpointing.
            # min_delay=0.1 is safe because the model guards against mid-sentence
            # false triggers (the comment-out 0.3s attempt broke Hindi pauses).
            "turn_detection": MultilingualModel(),
            "endpointing": {"min_delay": 0.1, "max_delay": 4.0},
        },
    )
    greeting_pcm: bytes | None = ud.get("greeting_pcm")

    @session.on("user_state_changed")
    def _on_user_state(ev):
        ts(f"user_state: {ev.old_state} → {ev.new_state}")

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        ts(f"agent_state: {ev.old_state} → {ev.new_state}")

    @session.on("user_input_transcribed")
    def _on_transcribed(ev):
        tag = "final" if getattr(ev, "is_final", False) else "interim"
        text = getattr(ev, "transcript", "") or ""
        ts(f"stt ({tag}): {text!r}")

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

    last_activity = [time.monotonic()]

    # Bump on EVERY state change — both speak-start and speak-end count as
    # activity. Earlier version only bumped on speak-start, which meant a long
    # agent response (e.g. 28s) would be counted as silence and the watchdog
    # would fire the moment the agent stopped talking. Fixed 2026-06-02.
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

    participant = None
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
        )
    )
