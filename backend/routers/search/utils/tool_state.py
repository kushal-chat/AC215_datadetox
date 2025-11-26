"""Request-scoped state for capturing tool results using FastAPI request state."""

from contextvars import ContextVar
from typing import Optional, Any
from fastapi import Request
import logging

# Store the current request in a context variable so tool functions can access it
_request_context: ContextVar[Optional[Request]] = ContextVar(
    "request_context", default=None
)


def set_request_context(request: Request) -> None:
    """Store the current request in the async context."""
    _request_context.set(request)


def get_request_context() -> Optional[Request]:
    """Get the current request from the async context."""
    return _request_context.get(None)


def set_tool_result(tool_name: str, result: Any) -> None:
    """Store a tool result in the current request's state."""
    request = get_request_context()
    if request:
        if not hasattr(request.state, "tool_results"):
            request.state.tool_results = {}
        request.state.tool_results[tool_name] = result
    else:
        # Fallback: log warning if no request context
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Attempted to store tool result '{tool_name}' but no request context available"
        )


def get_tool_result(tool_name: str, request: Optional[Request] = None) -> Optional[Any]:
    """Get a tool result from the request state."""
    if request is None:
        request = get_request_context()

    if request and hasattr(request.state, "tool_results"):
        return request.state.tool_results.get(tool_name)

    return None
