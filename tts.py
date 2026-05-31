import logging
from livekit.plugins import cartesia
from livekit.plugins import openai as lk_openai

logger = logging.getLogger("call-agent")

# STT language codes → Cartesia TTS language codes (Cartesia sonic-3.5 is
# multilingual and accepts these directly).
_STT_TO_TTS_LANG: dict[str, str] = {
    "hi": "hi", "ta": "ta", "te": "te", "mr": "mr", "bn": "bn",
    "es": "es", "fr": "fr", "de": "de", "pt": "pt", "ar": "ar",
    "ja": "ja", "ko": "ko", "zh": "zh", "en": "en",
    "multi": "en",
}

# Cartesia restricts word_timestamps to these languages on all sonic-* models
# (including sonic-3.5). Other languages get word_timestamps=False implicitly.
_WORD_TIMESTAMPS_LANGS = {"en", "de", "es", "fr"}


def build_tts(provider: str, model: str, voice: str, language: str = "en"):
    """
    Build a TTS instance from the configured provider.

    Cartesia: sonic-3.5 is the only supported/recommended model as of
    2026-06-01 (sonic, sonic-2, sonic-multilingual, sonic-turbo were all
    deprecated and removed). It's a single multi-language model — no model
    switching needed for non-English.
    """
    if provider == "openai":
        logger.info(f"TTS: OpenAI model={model} voice={voice}")
        return lk_openai.TTS(model=model, voice=voice)

    tts_lang = _STT_TO_TTS_LANG.get(language, "en")
    word_ts = tts_lang in _WORD_TIMESTAMPS_LANGS

    logger.info(
        f"TTS: Cartesia model={model} voice={voice} language={tts_lang} "
        f"word_timestamps={word_ts}"
    )
    return cartesia.TTS(model=model, voice=voice, language=tts_lang, word_timestamps=word_ts)
