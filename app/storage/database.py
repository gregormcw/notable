from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.core.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url)
session_local = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Note(Base):
    __tablename__ = "note"

    note_id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column()
    utterance_text: Mapped[str] = mapped_column()
    utterance_num_tokens: Mapped[int] = mapped_column()
    utterance_duration: Mapped[float] = mapped_column()

    def __repr__(self) -> str:
        return f"Note(note_id={self.note_id!r}, created_at={self.created_at!r}, utterance_text={self.utterance_text!r}, utterance_num_tokens={self.utterance_num_tokens!r}, utterance_duration={self.utterance_duration!r})"


Base.metadata.create_all(engine)


def get_db():
    db = session_local()
    try:
        yield db
    finally:
        db.close()
