from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str
    kind: str = "models"  # "models", "datasets", or "spaces"
    limit: int = 5
    include_cards: bool = False

class QueryResponse(BaseModel):
    answer: str
    search_results: List[Dict[str, Any]]
    cards: Optional[List[Dict[str, Any]]] = None
    sources: List[str]