"""Voice I/O: speech-to-text (faster-whisper) and text-to-speech (Piper)."""

from .io import VoiceIO, VoiceUnavailable

__all__ = ["VoiceIO", "VoiceUnavailable"]
