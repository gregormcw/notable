import numpy as np
import sounddevice as sd

from app.asr.transcriber import Transcriber
from app.audio.processor import process
from app.audio.vad import VADStream
from app.core.config import get_settings

settings = get_settings()
vad_stream = VADStream()
transcriber = Transcriber()

TEST_LENGTH = 5 * 1000  # 5 seconds
is_speech = False
chunks = []


def callback(indata: np.ndarray, frames: int, time: int, status: str | None) -> None:
    global is_speech, chunks

    audio_chunk = indata.flatten().copy()
    vad_status = vad_stream.feed(indata.flatten())

    if is_speech:
        chunks.append(audio_chunk)

    if vad_status:
        if "start" in vad_status:
            is_speech = True
            chunks.append(audio_chunk)
        elif "end" in vad_status:
            is_speech = False
            audio = np.concatenate(chunks)
            chunks.clear()
            transcription = transcriber.transcribe(process(audio))
            print(transcription)


def test_vad() -> None:
    print(f"Listening for {TEST_LENGTH / 1000} seconds... speak now.")
    with sd.InputStream(
        samplerate=settings.sample_rate,
        channels=1,
        blocksize=vad_stream.CHUNK_SIZE,
        callback=callback,
    ):
        sd.sleep(TEST_LENGTH)
    print("Done.")


if __name__ == "__main__":
    test_vad()
