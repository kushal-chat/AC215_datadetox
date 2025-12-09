"""Tests for tool_state module."""

from unittest.mock import MagicMock
from fastapi import Request

from routers.search.utils.tool_state import (
    set_request_context,
    get_request_context,
    set_tool_result,
    get_tool_result,
)


def test_set_request_context():
    """Test setting request context."""
    request = MagicMock(spec=Request)
    set_request_context(request)

    # Verify context was set
    retrieved = get_request_context()
    assert retrieved == request


def test_get_request_context_no_context():
    """Test getting request context when none is set."""
    # Clear any existing context by setting None
    set_request_context(None)
    result = get_request_context()
    assert result is None


def test_set_tool_result_with_request():
    """Test setting tool result when request context exists."""
    from types import SimpleNamespace

    request = MagicMock()
    request.state = SimpleNamespace(tool_results={})

    set_request_context(request)

    # Verify context was set
    retrieved = get_request_context()
    assert retrieved == request

    test_result = {"test": "data"}
    set_tool_result("test_tool", test_result)

    assert request.state.tool_results["test_tool"] == test_result


def test_set_tool_result_without_request():
    """Test setting tool result when no request context (should log warning)."""
    set_request_context(None)

    # Should not raise, just log warning
    test_result = {"test": "data"}
    set_tool_result("test_tool", test_result)  # Should handle gracefully


def test_set_tool_result_initializes_state():
    """Test that set_tool_result initializes tool_results if missing."""
    from types import SimpleNamespace

    request = MagicMock()
    request.state = SimpleNamespace()  # No tool_results initially

    set_request_context(request)

    # Verify context was set
    retrieved = get_request_context()
    assert retrieved == request

    test_result = {"test": "data"}
    set_tool_result("test_tool", test_result)

    assert hasattr(request.state, "tool_results")
    assert request.state.tool_results["test_tool"] == test_result


def test_get_tool_result_with_request_param():
    """Test getting tool result using request parameter."""
    from types import SimpleNamespace

    request = MagicMock()
    request.state = SimpleNamespace(tool_results={"test_tool": {"data": "value"}})

    result = get_tool_result("test_tool", request)
    assert result == {"data": "value"}


def test_get_tool_result_from_context():
    """Test getting tool result from request context."""
    from types import SimpleNamespace

    request = MagicMock()
    request.state = SimpleNamespace(tool_results={"test_tool": {"data": "value"}})

    set_request_context(request)

    # Verify context was set
    retrieved = get_request_context()
    assert retrieved == request

    result = get_tool_result("test_tool")
    assert result == {"data": "value"}


def test_get_tool_result_not_found():
    """Test getting tool result that doesn't exist."""
    request = MagicMock(spec=Request)
    request.state.tool_results = {}

    result = get_tool_result("nonexistent", request)
    assert result is None


def test_get_tool_result_no_request():
    """Test getting tool result when no request available."""
    set_request_context(None)

    result = get_tool_result("test_tool")
    assert result is None


def test_get_tool_result_no_tool_results_attr():
    """Test getting tool result when request has no tool_results attribute."""
    from types import SimpleNamespace

    request = MagicMock()
    request.state = SimpleNamespace()  # No tool_results attribute

    result = get_tool_result("test_tool", request)
    assert result is None
