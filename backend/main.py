from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.health_check import router as health_check_router
from routers.client import router as client_router
from pydantic import BaseModel


app = FastAPI(
    title="DataDetox API",
    description="API for DataDetox application",
    version="1.0.0"
)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # Allow all origins during development
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
