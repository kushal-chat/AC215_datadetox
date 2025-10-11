import asyncio
import base64
import io
import json
import logging
import os
import random
from re import search
from token import OP
import uuid
from typing import Any, AsyncIterator, Dict, List, Sequence

import fitz
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, PlainTextResponse
from PIL import Image
import pytesseract
from pydantic import BaseModel
from redis.asyncio import Redis
from starlette.status import HTTP_400_BAD_REQUEST

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    Runner,
    RunResult,
    RunResultStreaming,
    TResponseInputItem,
    custom_span,
    trace,
)
from agents.extensions.visualization import draw_graph
from agents.items import ItemHelpers
from agents.mcp import MCPServerSse
from context import ShinanContext
from openai.types.responses import (
    ResponseTextDeltaEvent,
)

from openai import OpenAI

router = APIRouter(prefix="/datadetox", tags=["client"])