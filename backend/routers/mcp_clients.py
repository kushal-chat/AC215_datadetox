from fastmcp import Client
from typing import Dict, Any, Optional

class HF_MCP_Client:
    """FastMCP client that works with SSE transport"""
    
    def __init__(self, mcp_url: str):
        self.mcp_url = mcp_url
    
    async def hf_search(self, query: str, kind: str = "models", limit: int = 5, filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Search using MCP tool"""
        async with Client(self.mcp_url) as client:
            result = await client.call_tool("hf_search", {
                "query": query,
                "kind": kind,
                "limit": limit,
                "filter": filter or {}
            })
            return result.content[0].text if result.content else {}
    
    async def hf_card(self, repo_id: str, kind: str = "model") -> Dict[str, Any]:
        """Get card using MCP tool"""
        async with Client(self.mcp_url) as client:
            result = await client.call_tool("hf_card", {
                "repo_id": repo_id,
                "kind": kind
            })
            return result.content[0].text if result.content else {}