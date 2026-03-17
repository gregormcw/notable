import chromadb
from openai import OpenAI

from app.core.config import get_settings


class VectorStore:
    """Persists and queries note embeddings using ChromaDB and the OpenAI API.

    Embeddings are generated with the configured OpenAI model
    (default: text-embedding-3-small) and stored in a ChromaDB collection
    with cosine similarity. Each note is keyed by its UUID note_id.
    """

    def __init__(self):
        settings = get_settings()
        self.embedding_model = settings.embedding_model
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="notes", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> list[float]:
        return (
            self.client.embeddings.create(input=text, model=self.embedding_model)
            .data[0]
            .embedding
        )

    def add(self, note_id: str, text: str) -> None:
        """Generate an embedding for text and store it in the collection.

        Args:
            note_id: UUID string used as the ChromaDB document ID.
            text: The transcribed utterance to embed and store.
        """
        embedding = self._get_embedding(text)
        self.collection.add(
            ids=[note_id],
            embeddings=[embedding],
            documents=[text],
        )

    def query(self, text: str, n_results: int) -> list[dict]:
        """
        Query the collection by semantic similarity.

        Returns a ChromaDB result dict with keys 'ids', 'documents', and
        'distances', each containing a list of lists (one per query).
        Results are ordered by similarity, closest first.
        """
        query_embedding = self._get_embedding(text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        results_reshaped = [
            {"note_id": id, "text": doc, "distance": dist}
            for id, doc, dist in zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
            )
        ]
        return results_reshaped


def get_vector_store() -> VectorStore:
    return VectorStore()
