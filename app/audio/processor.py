import noisereduce as nr
import numpy as np

from app.core.config import get_settings


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def apply_noise_gate(audio: np.ndarray, threshold_db: float) -> np.ndarray:
    """Zero out samples below the amplitude threshold.

    Prevents low-level background noise and silence from reaching the ASR
    engine, where it can cause hallucinations in models like Whisper.
    """
    threshold = db_to_linear(threshold_db)
    return np.where(np.abs(audio) >= threshold, audio, 0.0)


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtract the mean to eliminate DC bias from the signal."""
    return audio - np.mean(audio)


def normalize_rms(audio: np.ndarray, target_db: float) -> np.ndarray:
    """RMS-normalize audio to a target level in dBFS.

    Whisper performs most consistently when input levels are in a predictable
    range. Avoids the extremes of whisper-quiet recordings and clipping.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-9:
        return audio  # silence — nothing to normalize
    target_linear = db_to_linear(target_db)
    return audio * (target_linear / rms)


def reduce_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply spectral gating noise reduction via the noisereduce library.

    Uses a statistical model of background noise to attenuate non-speech
    frequency content. Provides the most meaningful accuracy improvement in
    noisy recording environments (open offices, cafés, etc.).
    """
    return nr.reduce_noise(y=audio, sr=sample_rate, stationary=False)


def compute_stats(audio: np.ndarray) -> dict:
    if audio.size == 0:
        return {"rms": 0.0, "peak": 0.0, "min": 0.0, "max": 0.0}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return {
        "rms": rms,
        "peak": float(np.max(np.abs(audio))),
        "min": float(np.min(audio)),
        "max": float(np.max(audio)),
    }


def process(audio: np.ndarray) -> np.ndarray:
    """Full pre-processing pipeline applied before ASR transcription.

    Order of operations:
        1. DC offset removal  — eliminates low-frequency drift artifacts
        2. Noise gate         — hard-zeros sub-threshold samples
        3. Spectral reduction — attenuates stationary and transient noise
        4. RMS normalization  — brings signal to a consistent level

    Args:
        audio: float32 mono PCM array at the configured sample rate.

    Returns:
        Processed float32 mono PCM array, ready for transcription.
    """
    settings = get_settings()

    # Cast and preserve float32
    audio = audio.astype(np.float32)
    pre_stats = compute_stats(audio)

    audio = remove_dc_offset(audio)
    audio = apply_noise_gate(audio, settings.noise_gate_db)
    if settings.apply_noise_reduction:
        audio = reduce_noise(audio, settings.sample_rate)
    audio = normalize_rms(audio, settings.target_db)

    max_abs = np.max(np.abs(audio)) if audio.size else 0.0
    if max_abs > 1.0:
        audio = audio / max_abs
        print(f"Audio debug: scaled down to avoid clipping (max_abs={max_abs:.6f})")

    post_stats = compute_stats(audio)
    print(
        f"Audio debug pre: rms={pre_stats['rms']:.6f}, peak={pre_stats['peak']:.6f}, min={pre_stats['min']:.6f}, max={pre_stats['max']:.6f}"
    )
    print(
        f"Audio debug post: rms={post_stats['rms']:.6f}, peak={post_stats['peak']:.6f}, min={post_stats['min']:.6f}, max={post_stats['max']:.6f}"
    )

    return audio.astype(np.float32)
