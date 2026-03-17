from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Notable"
    description: str = "Voice-first note capture and semantic retrieval."
    version: str = "0.1.0"
    debug: bool = False

    # OpenAI
    openai_api_key: str
    openai_model_name: str = ""

    # ASR
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "en"
    whisper_task: str = "transcribe"
    whisper_beam_size: int = 5

    # Audio processing
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 50
    noise_gate_db: float = -40.0
    target_db: float = -20.0
    apply_noise_reduction: bool = True
    debug_save_segments: bool = False
    debug_segment_dir: str = "./debug_segments"

    # Storage
    database_url: str = "sqlite:///./notable.db"
    chroma_path: str = "./chroma"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
