"""Voice I/O — talk to Daedalus and hear it answer, all locally.

Two local, $0 building blocks (both optional, in the ``[voice]`` extra):

  * **STT** — ``faster-whisper`` transcribes a WAV to text. Runs on CPU; the model
    downloads once and is cached.
  * **TTS** — ``piper`` synthesizes speech from text. Piper needs a voice model (an
    ``.onnx`` file you download once); point ``PIPER_VOICE`` at it.

Everything is lazy and guarded: importing this module never pulls the heavy deps, and
each method raises a clear :class:`VoiceUnavailable` (with the exact install hint) rather
than a cryptic ``ImportError`` when a piece is missing. Microphone capture/playback use
``sounddevice`` if present, so the live loop (``dae voice``) works on a real machine while
the core stays installable and testable without any audio stack.
"""

from __future__ import annotations

import wave
from pathlib import Path

from ..config import Settings


class VoiceUnavailable(RuntimeError):
    """Raised when a voice feature is requested but its dependency isn't installed."""


class VoiceIO:
    """Speech in, speech out. Each capability fails loudly and helpfully if absent."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._whisper = None  # lazily loaded WhisperModel
        self._piper = None  # lazily loaded PiperVoice

    # ---- speech-to-text ------------------------------------------------------

    def _whisper_model(self):
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise VoiceUnavailable(
                    'speech-to-text needs faster-whisper:  uv pip install -e ".[voice]"'
                ) from exc
            # int8 on CPU keeps it light; the model name comes from config (default 'base').
            self._whisper = WhisperModel(
                self.settings.whisper_model, device="cpu", compute_type="int8"
            )
        return self._whisper

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe a WAV file to text (empty string if nothing was recognized)."""
        model = self._whisper_model()
        segments, _info = model.transcribe(str(wav_path))
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ---- text-to-speech ------------------------------------------------------

    def _piper_voice(self):
        if self._piper is None:
            try:
                from piper.voice import PiperVoice
            except ImportError as exc:
                raise VoiceUnavailable(
                    'text-to-speech needs piper-tts:  uv pip install -e ".[voice]"'
                ) from exc
            model_path = self.settings.piper_voice
            if not model_path or not Path(model_path).exists():
                raise VoiceUnavailable(
                    "set PIPER_VOICE to a downloaded Piper voice (.onnx). See "
                    "https://github.com/rhasspy/piper for voices."
                )
            self._piper = PiperVoice.load(model_path)
        return self._piper

    def speak_to_file(self, text: str, out_path: str | Path) -> Path:
        """Synthesize ``text`` to a WAV file and return its path."""
        voice = self._piper_voice()
        out = Path(out_path)
        with wave.open(str(out), "wb") as wav:
            voice.synthesize(text, wav)
        return out

    # ---- live microphone loop (needs sounddevice) ----------------------------

    @staticmethod
    def _audio_io():
        try:
            import sounddevice as sd  # noqa: F401
            import soundfile as sf  # noqa: F401
        except ImportError as exc:
            raise VoiceUnavailable(
                "the live mic loop needs sounddevice + soundfile:  "
                "uv pip install sounddevice soundfile"
            ) from exc
        return sd, sf

    def record(self, out_path: str | Path, seconds: float = 5.0, samplerate: int = 16000) -> Path:
        """Record ``seconds`` of mono audio from the default mic to a WAV file."""
        sd, sf = self._audio_io()
        frames = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        out = Path(out_path)
        sf.write(str(out), frames, samplerate)
        return out

    def play(self, wav_path: str | Path) -> None:
        """Play a WAV file through the default output device (blocking)."""
        sd, sf = self._audio_io()
        data, samplerate = sf.read(str(wav_path), dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
