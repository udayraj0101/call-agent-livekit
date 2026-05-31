"""
Gemini Live Realtime voice agent — parallel to agent.py.

Same Aspirantive persona, same SIP wiring, same silence-watchdog + tool setup —
but the STT + LLM + TTS pipeline is replaced with a single Gemini Live Realtime
session (multimodal speech-to-speech).

Why a separate file:
- Keeps the proven Cartesia/Arushi production agent (agent.py) intact.
- Lets you A/B test latency & voice quality side-by-side: kill one worker, run
  the other. Both use agent_name="inbound-caller" so no LiveKit dispatch-rule
  change is needed when switching.

Run with:
    uv run python gemini_agent.py start
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import AsyncGenerator
from dotenv import load_dotenv

from livekit import agents, api, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google, noise_cancellation

from tools import create_tools

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-agent")


# ---------------------------------------------------------------------------
# Hardcoded sample config (mirrors agent.py for fair comparison)
# ---------------------------------------------------------------------------

# Same name as agent.py so the existing LiveKit dispatch rule still routes to
# whichever worker is running.
AGENT_NAME                = "inbound-caller"

# Gemini Live Realtime model. "preview-12-2025" is the current native-audio
# model on the public Gemini API; "gemini-3.1-flash-live-preview" is also
# available if you want to test the newer one.
GEMINI_MODEL              = "gemini-2.5-flash-native-audio-preview-12-2025"

# Voice persona for Arushi (Aspirantive sales). Candidates (description from
# Google's voice catalog):
#   Aoede     — Breezy        ← picked: closest to current Arushi feel
#   Sulafat   — Warm
#   Despina   — Smooth
#   Achird    — Friendly
#   Leda      — Youthful
GEMINI_VOICE              = "Aoede"

# BCP-47 language tag — Hindi (India). Gemini Live supports code-switching, so
# the model will still handle English/Hinglish inside this setting.
GEMINI_LANGUAGE           = "hi-IN"

GEMINI_TEMPERATURE        = 0.7
MAX_CALL_DURATION_SECONDS = 600

# Gemini's separate TTS endpoint (used for prerendering the greeting). Returns
# raw PCM s16le mono at 24 kHz, base64-encoded inside the JSON response.
GEMINI_TTS_MODEL          = "gemini-2.5-flash-preview-tts"
_GREETING_SAMPLE_RATE     = 24000

FIRST_MESSAGE = (
    "नमस्ते, मैं आरुषि बोल रही हूँ Aspirantive से। "
    "क्या आपके पास एक quick minute है?"
)

SYSTEM_PROMPT = (
    # Put the language rule FIRST, in Hindi, and unmissable — Gemini Live
    # tends to mirror the language of its instructions, so this anchors it.
    "भाषा का सबसे ज़रूरी नियम (LANGUAGE — MOST IMPORTANT RULE):\n"
    "• आप हर जवाब हिंदी या Hinglish में देंगी — कभी भी पूरी English में नहीं।\n"
    "• अगर सामने वाला English बोले, तब भी आप Hindi/Hinglish में ही reply करोगी।\n"
    "• Technical words जैसे CRM, ERP, software, spreadsheet, order, manage, "
    "minute — English में रख सकती हैं। बाकी sentence Hindi में हो।\n"
    "• आपकी आवाज़ का तरीका Indian, natural, professional और conversational हो।\n\n"
    "YOU MUST ALWAYS RESPOND IN HINDI OR HINGLISH. NEVER REPLY IN PURE ENGLISH.\n\n"
    "---\n"
    "ROLE:\n"
    "आप Arushi हैं, Aspirantive की outbound sales agent। Aspirantive pharmaceutical "
    "और distribution businesses के लिए custom CRM और ERP systems बनाती है।\n\n"
    "GOAL: prospect को qualify करना और 15-minute discovery call book करना।\n\n"
    "CONVERSATION FLOW:\n"
    "1. गर्मजोशी से greet करो, बताओ कि आप Aspirantive से बोल रही हैं।\n"
    "2. पूछो: \"क्या आपकी field team अभी भी orders manually या spreadsheets से "
    "manage करती है?\"\n"
    "3. अगर हाँ — समझाओ कि Aspirantive कैसे sales और inventory operations "
    "automate करती है।\n"
    "4. Objections को politely handle करो।\n"
    "5. 15-minute discovery call schedule करने का offer दो।\n"
    "6. अगर हाँ बोले, उनकी availability confirm करो और धन्यवाद कहो।\n\n"
    "STYLE:\n"
    "• Responses छोटे रखो — यह phone call है, email नहीं। एक-दो sentence में बात पूरी हो।\n"
    "• Sound natural और warm। Script padhne जैसा मत बोलो।\n"
    "• बातचीत finish हो जाए तो politely end करो।\n\n"
    "---\n"
    "CALL BEHAVIOUR (always follow):\n"
    "• CRITICAL: सिर्फ 'अलविदा' बोलने से call end नहीं होती — आपको end_call "
    "function ज़रूर call करना है। जब भी farewell बोलो, उसी turn में end_call invoke करो।\n"
    "• अगर caller 10 second से ज़्यादा silent रहे, छोटा prompt दो जैसे \"क्या आप अभी भी "
    "line पर हैं?\" और फिर call end करो।"
)


class SampleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Greeting prerender — uses Gemini's separate TTS endpoint (not the Realtime
# session) so the first audio plays instantly without paying any inference
# round-trip at call time. Same trick as agent.py uses with Cartesia.
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent / ".cache"


def _greeting_cache_path() -> Path:
    """
    Disk cache path keyed by (voice, model, message). If any of those change,
    the hash changes and we re-render.
    """
    key = f"{GEMINI_VOICE}|{GEMINI_TTS_MODEL}|{FIRST_MESSAGE}".encode()
    digest = hashlib.sha1(key).hexdigest()[:12]
    return _CACHE_DIR / f"greeting_{digest}.pcm"


def _prerender_greeting() -> bytes | None:
    """
    Synthesize FIRST_MESSAGE via Gemini TTS REST and return raw PCM bytes.

    Disk-cached: first worker process to hit this writes the PCM to .cache/;
    subsequent processes read from disk. Crucial for the Gemini free tier
    which only allows ~3 TTS requests/minute and was 429-ing on 4-way startup.
    """
    cache_path = _greeting_cache_path()
    if cache_path.exists():
        try:
            pcm = cache_path.read_bytes()
            seconds = len(pcm) / (_GREETING_SAMPLE_RATE * 2)
            logger.info(
                f"Greeting loaded from disk cache: {len(pcm)} bytes "
                f"(~{seconds:.1f}s of audio, voice={GEMINI_VOICE})"
            )
            return pcm
        except Exception as e:
            logger.warning(f"Disk cache read failed, will re-render: {e}")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_TTS_MODEL}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": FIRST_MESSAGE}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": GEMINI_VOICE},
                },
            },
        },
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Stagger startup requests across worker processes so they don't all race
    # the Gemini free-tier rate limit at the same time. Capped at 2 attempts —
    # the disk cache means at most one process needs to succeed per startup.
    time.sleep(random.uniform(0, 1.5))
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read())
            audio_b64 = (
                payload["candidates"][0]["content"]["parts"][0]
                ["inlineData"]["data"]
            )
            pcm = base64.b64decode(audio_b64)
            seconds = len(pcm) / (_GREETING_SAMPLE_RATE * 2)
            try:
                _CACHE_DIR.mkdir(exist_ok=True)
                cache_path.write_bytes(pcm)
                logger.info(
                    f"Greeting prerendered + cached to disk: {len(pcm)} bytes "
                    f"(~{seconds:.1f}s of audio, voice={GEMINI_VOICE}, path={cache_path.name})"
                )
            except Exception as e:
                logger.warning(f"Could not write cache file: {e}")
            return pcm
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 1:
                wait = 2.0 + random.uniform(0, 1.0)
                logger.info(f"Gemini TTS 429 during prerender, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            logger.warning(
                f"Greeting prerender failed (will fall back to live greeting): "
                f"HTTP {e.code} {e.reason}"
            )
            return None
        except Exception as e:
            logger.warning(f"Greeting prerender failed (will fall back to live greeting): {e}")
            return None
    return None


async def _stream_cached_greeting(pcm_bytes: bytes) -> AsyncGenerator[rtc.AudioFrame, None]:
    """Yield 20ms AudioFrames from cached PCM. Bypasses Realtime TTS entirely."""
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
# Lifecycle helpers (mirrors agent.py)
# ---------------------------------------------------------------------------

async def _end_call_on_error(session: AgentSession, ctx: agents.JobContext, err: Exception) -> None:
    logger.error(f"Realtime error during call — hanging up gracefully: {err}")
    try:
        await session.say(
            "मुझे एक तकनीकी समस्या हो रही है। कृपया कुछ देर बाद फिर से कॉल करें। धन्यवाद।"
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
        await session.say("मुझे अब कॉल समाप्त करनी होगी। आपके समय के लिए धन्यवाद। नमस्ते!")
        await asyncio.sleep(4)
    except Exception:
        pass
    try:
        await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
    except Exception as e:
        logger.warning(f"Could not delete room on timeout: {e}")
    ctx.shutdown()


# ---------------------------------------------------------------------------
# Prewarm — Gemini Realtime doesn't have a heavy VAD/STT/TTS chain to
# prewarm; the only artefact worth caching is the greeting audio.
# ---------------------------------------------------------------------------

def prewarm(proc: agents.JobProcess):
    proc.userdata["greeting_pcm"] = _prerender_greeting()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
    t0 = time.monotonic()
    def ts(label: str) -> None:
        logger.info(f"[+{(time.monotonic()-t0)*1000:.0f}ms] {label}")

    logger.info(f"Inbound call (Gemini) — connecting to room: {ctx.room.name}")
    logger.info(
        f"Config — provider=gemini model={GEMINI_MODEL} voice={GEMINI_VOICE} "
        f"language={GEMINI_LANGUAGE}"
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No GEMINI_API_KEY / GOOGLE_API_KEY in env — cannot start Gemini Realtime")
        ctx.shutdown()
        return

    ud = ctx.proc.userdata
    greeting_pcm: bytes | None = ud.get("greeting_pcm")

    session = AgentSession(
        # Gemini Live Realtime replaces the entire STT + LLM + TTS chain with a
        # single bidirectional audio session. No VAD, STT, or TTS plugins
        # needed alongside it — the model handles turn-taking server-side.
        llm=google.realtime.RealtimeModel(
            model=GEMINI_MODEL,
            voice=GEMINI_VOICE,
            language=GEMINI_LANGUAGE,
            temperature=GEMINI_TEMPERATURE,
            api_key=api_key,
            instructions=SYSTEM_PROMPT,
        ),
        tools=create_tools(ctx),
    )

    # Per-turn diagnostic logging — matches agent.py so you can compare logs
    # apples-to-apples between the two builds.
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

    # Silence watchdog (same as agent.py) — backup in case Gemini Live's
    # built-in turn detection misses a "you should hang up" cue.
    last_activity = [time.monotonic()]

    @session.on("user_state_changed")
    def _track_user_activity(ev):
        if ev.new_state == "speaking":
            last_activity[0] = time.monotonic()

    @session.on("agent_state_changed")
    def _track_agent_activity(ev):
        if ev.new_state == "speaking":
            last_activity[0] = time.monotonic()

    async def _silence_watchdog():
        while True:
            await asyncio.sleep(5)
            idle = time.monotonic() - last_activity[0]
            if idle > 25:
                logger.warning(f"Both parties silent for {idle:.0f}s — auto-ending call")
                try:
                    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
                except Exception as e:
                    logger.warning(f"Could not delete room on idle: {e}")
                return

    asyncio.create_task(_silence_watchdog())

    # Wait for the SIP caller to actually subscribe before publishing audio.
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

    # Greet immediately:
    # - cached path: stream prerendered Gemini-TTS audio frames (no inference)
    # - fallback: tell the realtime model to generate the greeting itself
    try:
        if greeting_pcm:
            ts(f"session.say begin (cached, {len(greeting_pcm)} bytes)")
            await session.say(FIRST_MESSAGE, audio=_stream_cached_greeting(greeting_pcm))
        else:
            ts("session.say begin (live realtime)")
            await session.generate_reply(
                instructions=f"Greet the caller by saying exactly: {FIRST_MESSAGE}"
            )
        ts("session.say done")
    except Exception as e:
        await _end_call_on_error(session, ctx, e)
        return

    # Block here until the room disconnects.
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
