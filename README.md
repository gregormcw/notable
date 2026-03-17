# Notable

**Voice-first note capture and semantic retrieval.**

Notable lets you speak a thought and retrieve it later in natural language. Press record, say something, stop — your words are transcribed, embedded, and stored locally. Later, ask "what did I say about X?" and Notable surfaces the closest matches semantically, not just by keyword.

Everything runs on your machine. No cloud ASR, no subscription, no data leaving the device except for OpenAI embeddings generation.

---

## Architecture

```
Browser mic (AudioWorklet)
        │  Float32 PCM @ 48kHz via WebSocket
        ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI WebSocket handler                          │
│                                                     │
│  resample_poly (48kHz → 16kHz)                      │
│        │                                            │
│        ▼                                            │
│  Silero-VAD (VADIterator, 512-sample chunks)        │
│        │  speech segment                            │
│        ▼                                            │
│  Pre-processing pipeline                            │
│    · DC offset removal                              │
│    · Noise gate (–40 dBFS floor)                   │
│    · Spectral noise reduction (noisereduce)         │
│    · RMS normalisation (–20 dBFS target)            │
│        │                                            │
│        ▼                                            │
│  faster-whisper (local, int8 quantised)             │
│        │  transcript text                           │
│        ▼                                            │
│  SQLite  ←──────────────────────────────────────┐  │
│  (note_id, text, timestamp, duration, tokens)   │  │
│                                                  │  │
│  OpenAI text-embedding-3-small                   │  │
│        │  1536-dim vector                         │  │
│        ▼                                          │  │
│  ChromaDB (cosine similarity, persistent)        ─┘  │
└─────────────────────────────────────────────────────┘
        │  transcript via WebSocket
        ▼
  Browser UI (captures notes, semantic search)
```

---

## Technical decisions

**Why Silero-VAD over WebRTC VAD?**
Silero is a lightweight neural VAD model (~2MB) that significantly outperforms energy-based approaches in realistic recording conditions — laptop microphones, background noise, soft speech. `VADIterator` is used rather than `get_speech_timestamps` to support chunk-based streaming, where audio arrives incrementally from the browser rather than as a complete file.

**Why faster-whisper locally?**
On-device ASR means no audio leaves the machine and there is no per-request API cost. faster-whisper uses CTranslate2 with int8 quantisation, making CPU inference practical. The `base` model offers a reasonable accuracy/latency tradeoff; `small` is available in config for better accuracy.

**Why AudioWorklet over ScriptProcessorNode?**
`ScriptProcessorNode` runs on the main browser thread and is deprecated. It introduces glitches and timing artifacts under load. `AudioWorklet` processes audio on a dedicated thread, producing a clean signal at the cost of a slightly more complex setup.

**Why separate VAD and pre-processing?**
VAD determines *when* speech occurs. Pre-processing improves the *quality* of the signal fed to ASR. Keeping them separate makes each independently tunable — VAD sensitivity, noise gate threshold, and normalisation target are all configurable via `.env` without touching code.

**Dual-store design (SQLite + ChromaDB)**
SQLite stores structured note metadata and is the source of truth for note content. ChromaDB stores embedding vectors for similarity search. The two stores are linked by a shared `note_id` (UUID). This separation keeps concerns clean: structured queries against SQLite, vector search against ChromaDB.

---

## Stack

| Component | Technology |
|---|---|
| Backend | FastAPI, Python 3.12 |
| WebSocket audio | AudioWorklet → FastAPI WebSocket |
| VAD | Silero-VAD (VADIterator) |
| ASR | faster-whisper (CTranslate2, int8) |
| Pre-processing | scipy, noisereduce |
| Resampling | scipy.signal.resample_poly |
| Note storage | SQLite via SQLAlchemy |
| Vector store | ChromaDB (cosine distance) |
| Embeddings | OpenAI text-embedding-3-small |
| Token counting | tiktoken |
| Frontend | Vanilla HTML/CSS/JS |

---

## Setup

### Prerequisites

- Python 3.12
- An OpenAI API key (for embeddings only — ASR runs locally)

### Installation

```bash
git clone https://github.com/yourusername/notable.git
cd notable

# Install torch CPU build first to avoid pulling the CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/static/index.html`.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | Used for embedding generation only |
| `WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantisation type |
| `VAD_THRESHOLD` | `0.5` | Silero-VAD speech probability threshold (0–1) |
| `VAD_MIN_SPEECH_MS` | `250` | Minimum speech segment duration |
| `VAD_MIN_SILENCE_MS` | `500` | Silence duration to trigger end event |
| `NOISE_GATE_DB` | `-40.0` | Amplitude floor below which samples are zeroed |
| `TARGET_DB` | `-20.0` | RMS normalisation target level |

---

## Usage

**Capturing a note**
Click the record button and speak. Notable listens continuously — speech is detected automatically by the VAD. Pause for a moment after speaking; the end-of-speech event triggers transcription. The transcript appears in the captured notes log on the left. You can speak multiple notes in a single session.

**Searching**
Type a natural language query in the search box on the right and press Enter or click the search icon. Results are ranked by semantic similarity — you do not need to use the exact words from your note.

---

## Known limitations

- **macOS Intel (OpenMP conflict):** faster-whisper and torch both bundle `libiomp5`, which conflicts on Intel Macs. Set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` in your environment before starting the server. Apple Silicon is unaffected.
- **Single user:** There is no authentication. The app is designed for local, single-user use. Multi-user support would require adding `user_id` to the note schema and a session layer.
- **No persistent session UI:** Notes captured in previous sessions are stored in the database and searchable, but do not populate the capture panel log on reload. The log is session-scoped.
- **Embedding cost:** Each note generates one OpenAI API call for embedding. For typical voice note volumes this is negligible, but the app is not suitable for bulk import of large note collections without rate-limit handling.

---

## Future directions

- Async background embedding to reduce post-speech latency
- Speaker diarisation for multi-speaker recordings
- LLM-powered note summarisation and theme extraction
- Export to Markdown / Obsidian vault
- REST API for third-party integrations