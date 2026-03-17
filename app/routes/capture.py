import json
from math import gcd

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from scipy.signal import resample_poly

from app.audio.vad import VADStream
from app.core.config import get_settings
from app.services.capture import Capture

settings = get_settings()
router = APIRouter(prefix="/ws", tags=["ws"])


_vad_stream = None
_capture = None


def _get_instances():
    global _vad_stream, _capture
    if _vad_stream is None:
        _vad_stream = VADStream()
    if _capture is None:
        _capture = Capture()
    return _vad_stream, _capture


@router.websocket("/ws")
async def capture_ws(websocket: WebSocket):
    vad_stream, capture = _get_instances()
    vad_stream.reset()
    vad_buffer = []
    segment_buffer = []
    is_speech = False
    resample = False

    await websocket.accept()
    print("WebSocket accepted")

    async def process_segment_and_send():
        nonlocal segment_buffer
        if not segment_buffer:
            return
        segment_audio = np.concatenate(segment_buffer)
        segment_buffer = []
        try:
            note = capture.process_segment(segment_audio)
            print(f"Note saved: {note.utterance_text}")
            await websocket.send_json({"transcript": note.utterance_text})
            print("Transcript sent")
        except Exception as e:
            print(f"Error in process segment: {e}")

    def flush_vad_buffer():
        nonlocal vad_buffer, segment_buffer, is_speech
        segment_closed = False
        while len(vad_buffer) >= vad_stream.CHUNK_SIZE:
            chunk = np.array(vad_buffer[: vad_stream.CHUNK_SIZE], dtype=np.float32)
            vad_buffer = vad_buffer[vad_stream.CHUNK_SIZE :]
            vad_status = vad_stream.feed(chunk)
            if is_speech:
                segment_buffer.append(chunk)
            if vad_status:
                print(f"VAD event: {vad_status}")
                if "start" in vad_status:
                    is_speech = True
                    segment_buffer.append(chunk)
                elif "end" in vad_status:
                    is_speech = False
                    segment_closed = True
        return segment_closed

    try:
        metadata = await websocket.receive_json()
        sample_rate_in = metadata.get("sr", settings.sample_rate)
        if sample_rate_in != settings.sample_rate:
            resample = True

        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                print("WebSocket disconnected")
                break

            if msg_type == "websocket.receive":
                if "text" in message and message["text"]:
                    try:
                        payload = json.loads(message["text"])
                        if payload.get("cmd") == "stop":
                            print("Stop command received")
                            # Flush any pending VAD chunks and process the latest segment.
                            segment_closed = flush_vad_buffer()
                            if segment_closed or segment_buffer:
                                await process_segment_and_send()
                            break
                    except Exception:
                        pass

                if "bytes" not in message or not message["bytes"]:
                    continue

                incoming = np.frombuffer(message["bytes"], dtype=np.float32)
                if resample and len(incoming):
                    incoming = resample_audio(
                        audio=incoming,
                        orig_sr=sample_rate_in,
                        target_sr=settings.sample_rate,
                    )

                vad_buffer.extend(incoming)
                segment_closed = flush_vad_buffer()
                if is_speech and vad_buffer:
                    segment_buffer.append(np.array(vad_buffer, dtype=np.float32))
                    vad_buffer = []
                if segment_closed:
                    await process_segment_and_send()

    except WebSocketDisconnect:
        print("WebSocket disconnected exception")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        try:
            if is_speech:
                await process_segment_and_send()
        except Exception as e:
            print(f"Error final flush: {e}")

        try:
            await websocket.close()
        except RuntimeError:
            pass


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g)
