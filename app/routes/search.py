from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import get_settings
from app.storage.database import Note, session_local
from app.storage.vector_store import VectorStore, get_vector_store


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/")
async def search(
    request: SearchRequest, vector_store: VectorStore = Depends(get_vector_store)
) -> list[dict]:
    results = vector_store.query(text=request.query, n_results=request.n_results)
    note_ids = [r["note_id"] for r in results]

    db = session_local()
    try:
        notes = db.query(Note).filter(Note.note_id.in_(note_ids)).all()
        notes_by_id = {note.note_id: note for note in notes}
    finally:
        db.close()

    enriched = []
    for r in results:
        note = notes_by_id.get(r["note_id"])
        created_at = note.created_at.isoformat() if note else None
        enriched.append(
            {
                "note_id": r["note_id"],
                "text": r["text"],
                "distance": r["distance"],
                "created_at": created_at,
            }
        )
    return enriched
