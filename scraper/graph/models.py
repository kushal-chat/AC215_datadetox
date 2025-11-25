"""Data models for graph nodes and edges."""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ModelNode(BaseModel):
    """Represents a model node in the graph."""
    model_id: str
    author: Optional[str] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    tags: List[str] = []
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    private: bool = False
    url: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class DatasetNode(BaseModel):
    """Represents a dataset node in the graph."""
    dataset_id: str
    author: Optional[str] = None
    downloads: Optional[int] = None
    tags: List[str] = []

class Relationship(BaseModel):
    """Represents a relationship between nodes."""
    source: str
    target: str
    relationship_type: str  # e.g., "based_on", "trained_on", "fine_tuned_from"
    source_type: str  # "model" or "dataset"
    target_type: str  # "model" or "dataset"
    metadata: Optional[Dict[str, Any]] = None

class GraphData(BaseModel):
    """Complete graph data structure."""
    models: List[ModelNode]
    datasets: List[DatasetNode]
    relationships: List[Relationship]
    metadata: Optional[Dict[str, Any]] = None

