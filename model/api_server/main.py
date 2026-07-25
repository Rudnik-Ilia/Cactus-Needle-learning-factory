import uvicorn
from fastapi import FastAPI

from .api.v1.router import router as tool_router

app = FastAPI()

app.include_router(tool_router, prefix="/api/v1")


def run() -> None:
    uvicorn.run("model.api_server.main:app", host="127.0.0.1", port=8000)
