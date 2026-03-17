import numpy as np
import pytest

from app.audio.processor import (apply_noise_gate, db_to_linear, normalize_rms,
                                 remove_dc_offset)


def sine_wave(
    frequency: float = 440.0, duration: float = 0.1, sample_rate: int = 16000
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


class TestDbToLinear:
    def test_zero_db_is_unity(self):
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_minus_20_db(self):
        assert db_to_linear(-20.0) == pytest.approx(0.1)

    def test_minus_40_db(self):
        assert db_to_linear(-40.0) == pytest.approx(0.01)


class TestRemoveDcOffset:
    def test_removes_constant_bias(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = remove_dc_offset(audio)
        assert np.mean(result) == pytest.approx(0.0, abs=1e-6)

    def test_zero_mean_signal_unchanged(self):
        audio = sine_wave()
        result = remove_dc_offset(audio)
        np.testing.assert_allclose(result, audio, atol=1e-5)


class TestApplyNoiseGate:
    def test_zeros_samples_below_threshold(self):
        audio = np.array([0.001, 0.5, -0.001, -0.5], dtype=np.float32)
        # threshold at -6 dB ≈ 0.5; samples below that magnitude should be zeroed
        result = apply_noise_gate(audio, threshold_db=-20.0)  # threshold ≈ 0.1
        assert result[0] == 0.0  # 0.001 < 0.1
        assert result[1] == pytest.approx(0.5)  # 0.5 >= 0.1
        assert result[2] == 0.0  # -0.001, abs < 0.1
        assert result[3] == pytest.approx(-0.5)  # -0.5, abs >= 0.1

    def test_silent_audio_stays_silent(self):
        audio = np.zeros(512, dtype=np.float32)
        result = apply_noise_gate(audio, threshold_db=-40.0)
        np.testing.assert_array_equal(result, audio)


class TestNormalizeRms:
    def test_output_rms_matches_target(self):
        audio = sine_wave()
        target_db = -20.0
        result = normalize_rms(audio, target_db)
        rms = float(np.sqrt(np.mean(result**2)))
        assert rms == pytest.approx(db_to_linear(target_db), rel=1e-3)

    def test_silent_audio_unchanged(self):
        audio = np.zeros(512, dtype=np.float32)
        result = normalize_rms(audio, target_db=-20.0)
        np.testing.assert_array_equal(result, audio)

    def test_idempotent_when_already_at_target(self):
        audio = sine_wave()
        target_db = -20.0
        once = normalize_rms(audio, target_db)
        twice = normalize_rms(once, target_db)
        np.testing.assert_allclose(once, twice, rtol=1e-4)
