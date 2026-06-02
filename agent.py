"""
Inbound LiveKit voice agent — hardcoded sample config.

Stripped-down version of the call-agent-saas worker. The LiveKit-side inbound
SIP trunk + dispatch rule should be configured to route incoming calls into a
room and dispatch the agent named below. Everything else (persona, models,
voice, greeting) is hardcoded.
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
logger = logging.getLogger("call-agent")


# ---------------------------------------------------------------------------
# Hardcoded sample config — edit here to tweak persona / models / voice.
# ---------------------------------------------------------------------------

# IMPORTANT: must match the agent_name in your LiveKit inbound SIP dispatch rule.
AGENT_NAME                = "inbound-caller"

LLM_MODEL                 = "gpt-4o-mini"
LLM_TEMPERATURE           = 0.7
STT_MODEL                 = "nova-3"          # upgraded from nova-2 — better Hinglish accuracy
STT_LANGUAGE              = "multi"            # Deepgram code-switching model
                                               # (better than "hi" for Hinglish)
TTS_LANGUAGE              = "hi"               # Cartesia / build_tts() need a real
                                               # language code, not "multi"
MAX_CALL_DURATION_SECONDS = 600               # 10 minutes per CRM config

# TTS — locked to Cartesia for low TTFB and the Arushi Hinglish voice.
# Provider can still be overridden via TTS_PROVIDER=openai env var as a fallback
# if Cartesia is unavailable (e.g. quota exhausted).
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "cartesia").lower()
if TTS_PROVIDER == "openai":
    TTS_MODEL = "tts-1"
    TTS_VOICE = "nova"
else:
    # sonic-3.5 is the unified multi-language successor to sonic-2 /
    # sonic-multilingual / sonic-turbo (all deprecated and removed by Cartesia
    # on 2026-06-01). It supports Hindi natively, so no model auto-switch needed.
    TTS_MODEL = "sonic-3.5"
    # "Arushi — Hinglish Speaker (Hindi)" — same voice the SaaS dashboard ships.
    TTS_VOICE = "95d51f79-c397-46f9-b49a-23763d3eaa2d"

FIRST_MESSAGE = (
    "नमस्ते, मैं आरुषि बोल रही हूँ Aspirantive से। "
    "क्या आपके पास एक quick minute है?"
)

# Sample rate for the prerendered greeting (Cartesia REST PCM s16le mono).
_GREETING_SAMPLE_RATE = 24000

# Trimmed from ~250 tokens to ~100 tokens to cut LLM prefix-processing time
# (~500ms saved per turn — measured on azure_agent.py 2026-06-01 test). Every
# word here ships on every request.
SYSTEM_PROMPT = (
    "You are Arushi from Aspirantive, calling Indian pharma/distribution "
    "businesses about CRM/ERP automation. Book a 15-minute discovery slot.\n\n"
    "Hinglish (Hindi+English), MAXIMUM 2 sentences and 20 words per reply — phone call, be brief. "
    "Ask if reps manage orders manually. Briefly pitch automation. Offer the slot."
)


class SampleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


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


def _prerender_greeting() -> bytes | None:
    """
    Synthesize FIRST_MESSAGE once via Cartesia's REST /tts/bytes endpoint and
    return raw PCM s16le mono bytes. Called from prewarm so the greeting plays
    instantly on call pickup without paying TTS TTFB.

    Returns None if TTS_PROVIDER isn't cartesia, the API key is missing, or
    Cartesia returns an error — the entrypoint falls back to live TTS in that
    case so a prerender failure never breaks the call.
    """
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

    # All N worker processes race to prerender simultaneously at startup, which
    # trips Cartesia's burst limit. A small random offset before the first
    # attempt + exponential backoff on 429 lets everyone through.
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
    """
    Issue a 1-token completion against the same model/endpoint the LLM plugin
    uses, so the OpenAI HTTPS connection pool and TLS session are established
    before the caller's first response lands.

    Runs fire-and-forget during greeting playback (~4s window). Total cost:
    a couple of tokens per call. Never re-raised — warmup failure must not
    affect the actual call.
    """
    try:
        client = openai.AsyncOpenAI()
        await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        logger.info("LLM connection warmed")
    except Exception as e:
        logger.warning(f"LLM warmup failed (non-fatal): {e}")


async def _stream_cached_greeting(pcm_bytes: bytes) -> AsyncGenerator[rtc.AudioFrame, None]:
    """Yield 20ms AudioFrames from the prerendered PCM. Bypasses TTS entirely."""
    samples_per_frame = int(_GREETING_SAMPLE_RATE * 0.02)  # 480 samples = 20ms
    bytes_per_frame = samples_per_frame * 2                 # s16le = 2 bytes/sample
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
# Prewarm — runs once per worker process at startup. Loads Silero VAD,
# pre-builds the STT/LLM/TTS plugin instances so their connection pools start
# opening websockets in the background, AND prerenders the greeting audio so
# the first word plays without paying any TTS TTFB at call time.
# ---------------------------------------------------------------------------

def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.4)
    proc.userdata["stt"] = deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300)
    proc.userdata["llm"] = lk_openai.LLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_completion_tokens=40)
    proc.userdata["tts"] = build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE)
    proc.userdata["greeting_pcm"] = _prerender_greeting()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
    t0 = time.monotonic()
    def ts(label: str) -> None:
        logger.info(f"[+{(time.monotonic()-t0)*1000:.0f}ms] {label}")

    logger.info(f"Inbound call — connecting to room: {ctx.room.name}")
    logger.info(
        f"Config — llm={LLM_MODEL} tts={TTS_PROVIDER}/{TTS_MODEL}/{TTS_VOICE} "
        f"stt={STT_MODEL}/{STT_LANGUAGE}"
    )

    ud = ctx.proc.userdata
    session = AgentSession(
        vad=ud.get("vad") or silero.VAD.load(min_silence_duration=0.4),
        stt=ud.get("stt") or deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, endpointing_ms=300),
        llm=ud.get("llm") or lk_openai.LLM(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_completion_tokens=40),
        tts=ud.get("tts") or build_tts(TTS_PROVIDER, TTS_MODEL, TTS_VOICE, TTS_LANGUAGE),
        tools=create_tools(ctx),
        turn_handling={
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
            "turn_detection": MultilingualModel(),
            "endpointing": {"min_delay": 0.1, "max_delay": 2.0},
        },
    )
    greeting_pcm: bytes | None = ud.get("greeting_pcm")

    # Per-turn diagnostic logging — shows where each turn's latency goes.
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
    # Fire-and-forget LLM warmup — runs in parallel with the greeting playback
    # so the OpenAI HTTPS+TLS connection pool is warm by the time the caller
    # finishes their first response. Knocks ~1-2s off the first LLM TTFT.
    asyncio.create_task(_warm_llm())

    # Silence watchdog — if neither party has spoken for >35s, force-end the call.
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

    # Wait for the SIP caller to actually subscribe to the agent's track before
    # speaking — otherwise the first TTS chunks are published into a track no
    # one is listening to yet. 10s timeout: if the SIP bridge doesn't establish
    # in that time, the call is almost certainly broken on the carrier side
    # (LiveKit trunk down, Vobiz balance, dispatch rule, etc.) — bail cleanly
    # instead of publishing the greeting into a dead room.
    participant = None
    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=10.0)
        ts(f"participant joined: {participant.identity}")
    except asyncio.TimeoutError:
        logger.error("No SIP participant joined within 10s — aborting call. "
                     "Check LiveKit SIP trunk and Vobiz status.")
        try:
            await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
        except Exception:
            pass
        ctx.shutdown()
        return
    except Exception as e:
        logger.warning(f"wait_for_participant failed, greeting anyway: {e}")

    # Speak the canned first line directly — no LLM round-trip = faster greeting.
    # If we have prerendered audio in userdata, stream it as AudioFrames (bypasses
    # TTS entirely → first word plays immediately). Falls back to live Cartesia
    # TTS if prerender failed at boot time.
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

    # Block here until the room disconnects so the worker keeps the session alive.
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
            # Internal HTTP server port (health/metrics). Default 8081; override
            # on the VPS via AGENT_HTTP_PORT env var because port 8081 is taken
            # by the call-agent-saas worker.
            port=int(os.getenv("AGENT_HTTP_PORT", "8081")),
        )
    )
