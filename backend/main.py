from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.health_check import router as health_check_router
from routers.client import router as client_router
from pydantic import BaseModel

app = FastAPI(
    title="DataDetox",
    description="DataDetox is a platform for interpretable fine-tuning and model / data trees.",
    version="0.0.1",
)

origins = [
    "http://localhost:3000⁠",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(client_router)
app.include_router(health_check_router)
