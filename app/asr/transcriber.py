import re

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import get_settings


class Transcriber:
    """
    Local ASR using faster-whisper with CTranslate2 int8 inference.

    vad_filter is disabled because voice activity detection is handled
    upstream by VADStream before audio reaches the transcriber.
    """

    def __init__(self):
        settings = get_settings()
        self.whisper_language = settings.whisper_language
        self.whisper_beam_size = settings.whisper_beam_size
        self.whisper_task = settings.whisper_task

        self.model = WhisperModel(
            model_size_or_path=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def _collapse_repeated_words(self, words: list[str]) -> list[str]:
        out = []
        i = 0
        # Collapse repeated single tokens (e.g. woodchop woodchop ...)
        while i < len(words):
            j = i + 1
            while j < len(words) and words[j] == words[i]:
                j += 1
            repeat_count = j - i
            if repeat_count > 3:
                out.append(words[i])
            else:
                out.extend(words[i:j])
            i = j

        # Collapse repeated contiguous n-grams for short n (2-4) repeated at least 2 times.
        words = out
        out = []
        i = 0
        while i < len(words):
            matched = False
            for span in range(min(4, len(words) - i), 1, -1):
                block = words[i : i + span]
                j = i + span
                repeats = 1
                while j + span <= len(words) and words[j : j + span] == block:
                    repeats += 1
                    j += span
                if repeats >= 2:
                    out.extend(block)
                    i = j
                    matched = True
                    break
            if not matched:
                out.append(words[i])
                i += 1
        return out

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        # Flatten repeated punctuation and whitespace
        text = re.sub(r"[\s\n]+", " ", text)
        text = re.sub(r"([.,!?;:])\1+", r"\1", text)
        return text

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a pre-processed audio segment to text.

        Runs faster-whisper inference and applies three layers of repetition
        cleanup to guard against Whisper hallucination loops: single-token
        deduplication, n-gram deduplication, and a final token-frequency
        safeguard. Returns an empty string if no speech is detected.

        Args:
            audio: float32 mono PCM array at the configured sample rate,
                   already processed by the audio pipeline.

        Returns:
            Cleaned transcript string, or ``""`` if the segment is silent.
        """
        segments, _ = self.model.transcribe(
            audio=audio,
            vad_filter=False,
            language=self.whisper_language,
            task=self.whisper_task,
            beam_size=self.whisper_beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        # Collect non-empty segment text.
        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        if not texts:
            return ""

        normalized = self._normalize_text(" ".join(texts))
        words = normalized.split()

        # Collapse repeated words and repeated phrase chunks.
        words = self._collapse_repeated_words(words)
        cleaned = " ".join(words).strip()
        if len(cleaned) <= 1:
            return normalized

        # Final safe guard: if output is mostly repeating a single token, collapse.
        toks = cleaned.split()
        if (
            len(toks) > 10
            and (max(toks.count(t) for t in set(toks)) / len(toks)) > 0.45
        ):
            # keep first unique 20 tokens as fallback
            seen = set()
            dedup = []
            for t in toks:
                if t not in seen:
                    dedup.append(t)
                    seen.add(t)
                if len(dedup) >= 20:
                    break
            cleaned = " ".join(dedup)

        return cleaned
