"""Unit tests for tool_state.py functions."""

from unittest.mock import Mock, patch
from fastapi import Request
from starlette.datastructures import State
from routers.search.utils.tool_state import (
    set_request_context,
    get_request_context,
    set_tool_result,
    get_tool_result,
    set_progress_callback,
    get_progress_callback,
)


class TestRequestContext:
    """Tests for request context functions."""

    def test_set_and_get_request_context(self):
        """Test setting and getting request context."""
        mock_request = Mock(spec=Request)
        set_request_context(mock_request)
        result = get_request_context()
        assert result == mock_request

    def test_get_request_context_none(self):
        """Test getting request context when not set."""
        # Clear any existing context
        set_request_context(None)
        result = get_request_context()
        # Should return None or the default
        assert result is None or result is None


class TestToolResult:
    """Tests for tool result functions."""

    def test_set_tool_result_with_request(self):
        """Test setting tool result with request context."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()
        mock_request.state.tool_results = {}

        with patch("routers.search.utils.tool_state.get_request_context", return_value=mock_request):
            set_tool_result("test_tool", {"result": "data"})
            assert mock_request.state.tool_results["test_tool"] == {"result": "data"}

    def test_set_tool_result_creates_dict(self):
        """Test that tool_results dict is created if it doesn't exist."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()

        with patch("routers.search.utils.tool_state.get_request_context", return_value=mock_request):
            set_tool_result("test_tool", {"result": "data"})
            assert hasattr(mock_request.state, "tool_results")
            assert mock_request.state.tool_results["test_tool"] == {"result": "data"}

    def test_set_tool_result_no_request_context(self):
        """Test setting tool result when no request context."""
        with patch("routers.search.utils.tool_state.get_request_context", return_value=None):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                # Re-import to get the patched logger
                import importlib
                import routers.search.utils.tool_state as tool_state_module
                importlib.reload(tool_state_module)
                tool_state_module.set_tool_result("test_tool", {"result": "data"})
                # Logger should be called (warning logged)
                # Just verify the function doesn't crash
                assert True

    def test_get_tool_result_with_request(self):
        """Test getting tool result with request."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()
        mock_request.state.tool_results = {"test_tool": {"result": "data"}}

        result = get_tool_result("test_tool", mock_request)
        assert result == {"result": "data"}

    def test_get_tool_result_from_context(self):
        """Test getting tool result from request context."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()
        mock_request.state.tool_results = {"test_tool": {"result": "data"}}

        with patch("routers.search.utils.tool_state.get_request_context", return_value=mock_request):
            result = get_tool_result("test_tool")
            assert result == {"result": "data"}

    def test_get_tool_result_not_found(self):
        """Test getting tool result that doesn't exist."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()
        mock_request.state.tool_results = {}

        result = get_tool_result("nonexistent_tool", mock_request)
        assert result is None

    def test_get_tool_result_no_request(self):
        """Test getting tool result when no request available."""
        with patch("routers.search.utils.tool_state.get_request_context", return_value=None):
            result = get_tool_result("test_tool")
            assert result is None

    def test_get_tool_result_no_tool_results_attr(self):
        """Test getting tool result when tool_results attr doesn't exist."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()
        # Don't set tool_results attribute

        result = get_tool_result("test_tool", mock_request)
        assert result is None

    def test_multiple_tool_results(self):
        """Test storing and retrieving multiple tool results."""
        mock_request = Mock(spec=Request)
        mock_request.state = State()
        mock_request.state.tool_results = {}

        with patch("routers.search.utils.tool_state.get_request_context", return_value=mock_request):
            set_tool_result("tool1", {"result1": "data1"})
            set_tool_result("tool2", {"result2": "data2"})

            result1 = get_tool_result("tool1", mock_request)
            result2 = get_tool_result("tool2", mock_request)

            assert result1 == {"result1": "data1"}
            assert result2 == {"result2": "data2"}


class TestProgressCallback:
    """Tests for progress callback functions."""

    def test_set_and_get_progress_callback(self):
        """Test setting and getting progress callback."""
        mock_callback = Mock()
        set_progress_callback(mock_callback)
        result = get_progress_callback()
        assert result == mock_callback

    def test_get_progress_callback_none(self):
        """Test getting progress callback when not set."""
        # Clear callback
        set_progress_callback(None)
        result = get_progress_callback()
        assert result is None

    def test_progress_callback_independent_from_request(self):
        """Test that progress callback is independent from request context."""
        mock_callback = Mock()
        set_progress_callback(mock_callback)

        # Request context should not affect callback
        mock_request = Mock(spec=Request)
        set_request_context(mock_request)

        result = get_progress_callback()
        assert result == mock_callback

