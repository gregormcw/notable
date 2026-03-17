import os
import uuid
from datetime import datetime, timezone

import numpy as np
import tiktoken
from scipy.io import wavfile

from app.asr.transcriber import Transcriber
from app.audio.processor import process
from app.core.config import get_settings
from app.storage.database import Note, session_local
from app.storage.vector_store import VectorStore


class VectorStoreError(Exception):
    pass


class DatabaseError(Exception):
    pass


class Capture:
    """Orchestrates the full note capture pipeline: audio → ASR → storage.

    Coordinates the transcriber, audio processor, vector store, and
    database to convert a raw speech segment into a persisted, searchable
    note.
    """

    def __init__(self):
        self.transcriber = Transcriber()
        self.vector_store = VectorStore()
        self.settings = get_settings()

    def process_segment(self, audio: np.ndarray) -> Note:
        """Process a raw speech segment into a persisted note.

        Runs the audio through the pre-processing pipeline, transcribes it
        with Whisper, generates an embedding via the OpenAI API, and writes
        both the metadata and embedding to their respective stores.

        Args:
            audio: float32 mono PCM array at the configured sample rate,
                   covering a single speech segment as detected by VADStream.

        Returns:
            The newly created Note ORM instance.

        Raises:
            VectorStoreError: If the ChromaDB write fails.
            DatabaseError: If the SQLite write fails.
        """
        # Pre-process audio
        audio_processed = process(audio)

        # Generate note_id and created_at timestamp
        note_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        if self.settings.debug_save_segments:
            os.makedirs(self.settings.debug_segment_dir, exist_ok=True)
            debug_path = os.path.join(
                self.settings.debug_segment_dir, f"segment_{note_id}.wav"
            )
            wavfile.write(debug_path, self.settings.sample_rate, audio_processed)
            print(f"Saved debug segment WAV: {debug_path}")

        # Transcribe audio
        utterance_text = self.transcriber.transcribe(audio_processed)

        # Compute utterance_num_tokens and utterance_duration
        enc = tiktoken.encoding_for_model(self.settings.embedding_model)
        utterance_num_tokens = len(enc.encode(utterance_text))

        # Compute utterance_duration
        utterance_duration = len(audio_processed) / self.settings.sample_rate
        print(
            f"transcription debug: text={utterance_text!r}, duration={utterance_duration:.2f}s"
        )

        # Initialize database variables
        db = session_local()
        note = None

        try:
            # Generate embedding and save to ChromaDB
            self.vector_store.add(note_id=note_id, text=utterance_text)
        except Exception as e:
            raise VectorStoreError(f"Failed to save embedding to ChromaDB: {e}") from e

        try:
            # Save to SQLite
            note = Note(
                note_id=note_id,
                created_at=created_at,
                utterance_text=utterance_text,
                utterance_num_tokens=utterance_num_tokens,
                utterance_duration=utterance_duration,
            )
            db.add(note)
            db.commit()
            db.expunge(note)

        except Exception as e:
            db.rollback()
            raise DatabaseError(f"Failed to save note to database: {e}") from e

        finally:
            db.close()

        if note is None:
            raise NameError("Note never initialized. Nothing returned.")
        return note
