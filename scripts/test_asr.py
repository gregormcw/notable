import sys
import numpy as np
from scipy.io import wavfile
from app.asr.transcriber import Transcriber


def run_test(wav_path: str):
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    if audio.dtype == np.int16:
        audio /= 32768.0
    elif audio.dtype == np.int32:
        audio /= 2147483648.0
    transcriber = Transcriber()
    print(f"Loaded {wav_path} sr={sr} len={len(audio)}")
    text = transcriber.transcribe(audio)
    print("Transcription:")
    print(text)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_asr.py <path/to/test.wav>")
        sys.exit(1)
    run_test(sys.argv[1])
