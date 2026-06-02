"""
SaaS-LiveKit-compatible Azure India inbound agent.

Mirrors azure_agent.py features exactly (Azure gpt-4o India + Cartesia Arushi +
Deepgram nova-3 Hindi + greeting prerender + LLM warmup + silence watchdog +
trimmed prompt + max_tokens cap) but registers on the call-agent-saas LiveKit
project using a SEPARATE set of credentials.

Goal: validate that the same agent quality + latency hold when calls are routed
through the SaaS project's existing SIP trunk and dispatch rule.

Supports both inbound and outbound (SaaS-style):
- Inbound: SaaS SIP dispatch rule routes incoming call → this agent picks up
- Outbound: dispatched with metadata {"phone_number": "+91...", "trunk_id": ...}
  → this agent places the call via the SaaS trunk

Intentionally OMITS the SaaS internal-API integrations (KB fetch, transcript
POST, inbound config lookup) — we don't have access to those endpoints from
this repo. The agent config is hardcoded Aspirantive (same as azure_agent.py);
job metadata is logged but not used to override config.

============================================================================
Required env vars — add these to .env alongside the existing aspirantive ones:
============================================================================

    # SaaS LiveKit project credentials (DIFFERENT from LIVEKIT_URL/KEY/SECRET
    # used by agent.py and azure_agent.py — those point at ai-integration)
    LIVEKIT_SAAS_URL          = wss://<saas-project>.livekit.cloud
    LIVEKIT_SAAS_API_KEY      = <key from SaaS LiveKit project>
    LIVEKIT_SAAS_API_SECRET   = <secret from SaaS LiveKit project>

    # agent_name to register as — MUST match the agent_name in the SaaS
    # project's SIP Dispatch Rule. SaaS default is "outbound-caller".
    SAAS_AGENT_NAME           = outbound-caller

Cartesia, Deepgram, and Azure OpenAI keys are SHARED with the existing setup
— reads CARTESIA_API_KEY, DEEPGRAM_API_KEY, AZURE_OPENAI_* from .env directly.

Run with:
    uv run python saas_inbound_agent.py start
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
logger = logging.getLogger("saas-azure-agent")


# ---------------------------------------------------------------------------
# Hardcoded Aspirantive config — mirrors azure_agent.py exactly
# ---------------------------------------------------------------------------

# Must match the agent_name in the SaaS project's SIP Dispatch Rule.
AGENT_NAME = os.getenv("SAAS_AGENT_NAME", "outbound-caller")

# SaaS LiveKit project credentials — read from separate env vars so this file
# can run alongside agent.py/azure_agent.py (which use the aspirantive
# LIVEKIT_URL/KEY/SECRET) without env conflicts.
SAAS_LIVEKIT_URL        = os.getenv("LIVEKIT_SAAS_URL")
SAAS_LIVEKIT_API_KEY    = os.getenv("LIVEKIT_SAAS_API_KEY")
SAAS_LIVEKIT_API_SECRET = os.getenv("LIVEKIT_SAAS_API_SECRET")

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

# Azure OpenAI config — same as azure_agent.py
AZURE_OPENAI_ENDPOINT     = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY      = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT   = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION  = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

# Cartesia TTS — same Arushi voice
TTS_PROVIDER = "cartesia"
TTS_MODEL    = "sonic-3.5"
TTS_VOICE    = "95d51f79-c397-46f9-b49a-23763d3eaa2d"  # Arushi — Hinglish Speaker

FIRST_MESSAGE = (
    "नमस्ते, मैं आरुषि बोल रही हूँ Aspirantive से। "
    "क्या आपके पास एक quick minute है?"
)

_GREETING_SAMPLE_RATE = 24000

# Same trimmed prompt as azure_agent.py — ~100 tokens, prefix-processing optimized.
SYSTEM_PROMPT = (
    "You are Arushi from Aspirantive, calling Indian pharma/distribution "
    "businesses about CRM/ERP automation. Book a 15-minute discovery slot.\n\n"
    "Hinglish (Hindi+English), MAXIMUM 2 sentences and 20 words per reply — phone call, be brief. "
    "Ask if reps manage orders manually. Briefly pitch automation. Offer the slot."
)


class SampleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------

def _build_azure_llm():
    return lk_openai.LLM.with_azure(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=LLM_TEMPERATURE,
        max_completion_tokens=40,
    )

def _build_openai_llm():
    # gpt-4o-mini direct — ~400-600ms TTFB even from India (vs 900-1400ms for
    # Azure gpt-4o full). Use this until Azure gpt-4o-mini India quota is approved.
    return lk_openai.LLM(model="gpt-4o-mini", temperature=LLM_TEMPERATURE, max_completion_tokens=40)


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
# Greeting prerender — same as azure_agent.py
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
    try:
        client = openai.AsyncOpenAI()
        await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        logger.info("OpenAI LLM connection warmed")
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
# Prewarm
# ---------------------------------------------------------------------------

def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)
    proc.userdata["stt"] = deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300)
    proc.userdata["llm"] = _build_openai_llm()
    proc.userdata["tts"] = build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE)
    proc.userdata["greeting_pcm"] = _prerender_greeting()


# ---------------------------------------------------------------------------
# Entrypoint — handles both inbound and outbound (SaaS-style metadata)
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
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

    # Parse SaaS-style job metadata. We log it but don't use it to override the
    # hardcoded Aspirantive config — this agent is a fixed-config testbed for
    # validating azure_agent.py features through SaaS infrastructure.
    metadata: dict = {}
    try:
        if ctx.job.metadata:
            metadata = json.loads(ctx.job.metadata)
    except Exception:
        logger.warning("Could not parse job metadata — proceeding with hardcoded config.")

    phone_number: str | None = metadata.get("phone_number")
    trunk_id:     str | None = metadata.get("trunk_id") or os.getenv("OUTBOUND_TRUNK_ID")
    direction = "outbound" if phone_number else "inbound"

    logger.info(
        f"{direction.title()} call (SaaS-LiveKit + OpenAI gpt-4o-mini) — "
        f"room: {ctx.room.name}"
    )
    logger.info(
        f"Config — llm=openai/gpt-4o-mini "
        f"tts={TTS_PROVIDER}/{TTS_MODEL}/{TTS_VOICE} stt={STT_MODEL}/{STT_LANGUAGE} "
        f"direction={direction}"
    )
    if metadata:
        logger.info(f"Metadata (logged, not applied): {metadata}")

    ud = ctx.proc.userdata
    session = AgentSession(
        vad=ud.get("vad") or silero.VAD.load(min_silence_duration=0.4),
        stt=ud.get("stt") or deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300),
        llm=ud.get("llm") or _build_openai_llm(),
        tts=ud.get("tts") or build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE),
        tools=create_tools(ctx),
        turn_handling={
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
            # MultilingualModel predicts end-of-turn semantically so the agent
            # can commit before Deepgram's 300ms silence wait expires (~200-300ms
            # saved). min_delay lowered from default 0.5s to 0.1s — the model
            # acts as the semantic guard; Deepgram's 300ms is still the floor.
            "turn_detection": MultilingualModel(),
            "endpointing": {"min_delay": 0.1, "max_delay": 2.0},
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

    # ── Outbound branch — place a SIP call via the SaaS trunk ──────────────
    if direction == "outbound":
        if not trunk_id:
            logger.error("Outbound call but no trunk_id in metadata or env — aborting.")
            ctx.shutdown()
            return
        logger.info(f"Placing outbound call → {phone_number} via trunk {trunk_id}")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=f"sip_{phone_number}",
                    wait_until_answered=True,
                )
            )
            ts(f"outbound call answered: {phone_number}")
        except Exception as e:
            logger.error(f"Outbound call failed: {e}")
            try:
                await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
            except Exception:
                pass
            ctx.shutdown()
            return

    # ── Inbound branch — wait for SIP caller to subscribe ──────────────────
    else:
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

    # ── Speak greeting (cached PCM bypasses TTS for instant first-word) ────
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
    # Fail fast if the SaaS LiveKit creds are missing — without them the worker
    # would silently fall back to the aspirantive LiveKit project and register
    # there, which would collide with the agent.py/azure_agent.py worker.
    missing = [
        name for name, value in [
            ("LIVEKIT_SAAS_URL", SAAS_LIVEKIT_URL),
            ("LIVEKIT_SAAS_API_KEY", SAAS_LIVEKIT_API_KEY),
            ("LIVEKIT_SAAS_API_SECRET", SAAS_LIVEKIT_API_SECRET),
        ] if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing required SaaS LiveKit env vars: {missing}. "
            f"Add them to .env — see the file header for the full list."
        )

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
            ws_url=SAAS_LIVEKIT_URL,
            api_key=SAAS_LIVEKIT_API_KEY,
            api_secret=SAAS_LIVEKIT_API_SECRET,
            port=int(os.getenv("AGENT_HTTP_PORT", "8083")),
        )
    )
