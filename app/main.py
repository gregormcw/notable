from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.routes.capture import router as capture_router
from app.routes.search import router as search_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


settings = get_settings()
app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    description=settings.description,
    version=settings.version,
    debug=settings.debug,
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(capture_router)
app.include_router(search_router)
