import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from app.core.config import get_settings


class VADStream:
    """
    Streaming voice activity detector using Silero-VAD.

    Uses VADIterator rather than get_speech_timestamps to support real-time
    chunk-based processing, where audio arrives incrementally from the
    microphone rather than as a complete buffer. Each 512-sample chunk is
    fed to the iterator which maintains internal state between calls and
    emits events on speech onset and offset.
    """

    def __init__(self):
        settings = get_settings()

        self.sr = settings.sample_rate
        self.CHUNK_SIZE = 512
        self.return_seconds = True

        self.model = load_silero_vad()
        self.model.eval()
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=settings.vad_threshold,
            sampling_rate=settings.sample_rate,
            min_silence_duration_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
        )

    def reset(self) -> None:
        """Reset VAD state (use this when starting a new stream/file)."""
        self._buf = np.zeros((0,), dtype=np.float32)
        self.vad_iterator.reset_states()

    def feed(self, x: np.ndarray) -> dict | None:
        """Process a single 512-sample chunk and return a VAD event if a
        speech boundary is detected.

        Returns:
            ``{'start': float}`` on speech onset, ``{'end': float}`` on speech
            offset, or ``None`` if no transition occurred. Timestamps are in
            seconds.
        """
        x_tensor = torch.from_numpy(x)
        vad_event = self.vad_iterator(x=x_tensor, return_seconds=self.return_seconds)
        return vad_event if vad_event else None
