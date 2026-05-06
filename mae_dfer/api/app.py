from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env_file() -> None:
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break


_load_env_file()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mae_dfer.api.controllers.health_controller import router as health_router
from mae_dfer.api.controllers.predict_controller import router as predict_router
from mae_dfer.api.controllers.websocket_controller import router as websocket_router
from mae_dfer.api.services.model_registry import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="MAE-DFER emotion inference", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(websocket_router)
    return app

