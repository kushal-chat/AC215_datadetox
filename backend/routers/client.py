import os
import json
import sys
import asyncio
import logging
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from routers.mcp_clients import HF_MCP_Client
from routers.api_models import QueryRequest, QueryResponse
from agents.prompts import Prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
HF_MCP_URL = os.getenv("HF_MCP_URL", "http://0.0.0.0:8081/sse")  # FastMCP uses /sse endpoint

router = APIRouter(prefix="/datadetox", tags=["client"])

# Initialize MCP client
mcp_client = HF_MCP_Client(HF_MCP_URL)

@router.post("/query_hf", response_model=QueryResponse)
async def query_hf(request: QueryRequest):
    """Query Hugging Face Hub using MCP and synthesize results with LLM"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not configured")
    
    try:
        # # Step 1: Initialize LLM and generate optimized search query
        # openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # query_generation_prompt = Prompt.get_hf_search_prompt(request)

        # logger.info("Generating optimized search query with LLM")
        # query_response = openai_client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "You are an expert at converting user queries into effective Hugging Face Hub search terms."},
        #         {"role": "user", "content": query_generation_prompt}
        #     ],
        #     max_tokens=50,
        #     temperature=0.1
        # )
        
        # optimized_query = query_response.choices[0].message.content.strip()
        # logger.info(f"Original query: '{request.query}' → Optimized query: '{optimized_query}'")
        
        # Step 2: Search Hugging Face using the optimized query
        logger.info(f"Searching HF for: {request.query} (kind: {request.kind})")
        search_response = await mcp_client.hf_search(
            query=request.query,
            kind=request.kind,
            limit=request.limit
        )

        # Parse the response (it comes as JSON string from MCP)
        if isinstance(search_response, str):
            search_response = json.loads(search_response)

        search_results = search_response.get("results", [])
        if not search_results:
            return QueryResponse(
                answer="No results found for your query.",
                search_results=[],
                sources=[]
            )
        
        logger.info(f"Found {len(search_results)} results")
        
        # Step 3: Optionally fetch detailed cards for top results
        cards = []
        if request.include_cards and search_results:
            logger.info("Fetching detailed cards for top results")
            for result in search_results:  
                try:
                    card_response = await mcp_client.hf_card(
                        repo_id=result["repo_id"],
                        kind=result["kind"]
                    )
                    if isinstance(card_response, str):
                        card_response = json.loads(card_response)
                    cards.append(card_response)
                except Exception as e:
                    logger.warning(f"Failed to fetch card for {result['repo_id']}: {e}")
        
        # Step 4: Use LLM to synthesize the results
        
        # # Prepare context for LLM
        # results_summary = []
        # for i, result in enumerate(search_results, 1):
        #     summary = f"{i}. **{result['repo_id']}** by {result.get('author', 'Unknown')}\n"
        #     summary += f"   - Likes: {result.get('likes', 0)}, Downloads: {result.get('downloads', 0)}\n"
        #     summary += f"   - Tags: {', '.join(result.get('tags', [])[:5])}\n"
        #     summary += f"   - URL: {result['url']}\n"
        #     if result.get('library_name'):
        #         summary += f"   - Library: {result['library_name']}\n"
        #     results_summary.append(summary)
        
        # # Add card details if available
        # card_context = ""
        # if cards:
        #     card_context = "\n\nDetailed information for top results:\n"
        #     for card in cards:
        #         card_context += f"\n**{card['repo_id']}**:\n"
        #         if card.get('readme'):
        #             # Truncate readme to avoid token limits
        #             readme_preview = card['readme'][:1000] + "..." if len(card['readme']) > 1000 else card['readme']
        #             card_context += f"README: {readme_preview}\n"
        
        # # Create LLM prompt
        # prompt = Prompt.get_hf_summarise_results_prompt(request, optimized_query, results_summary, card_context)

        # # Get LLM response
        # response = openai_client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that analyzes Hugging Face Hub search results and provides clear, actionable recommendations."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=800,
        #     temperature=0.3
        # )
        
        # answer = response.choices[0].message.content
        
        # Extract sources (URLs from results)
        sources = [result["url"] for result in search_results]
            
        return QueryResponse(
            answer="test", # answer,
            search_results=search_results,
            cards=cards if request.include_cards else None,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error in query_hf: {e}")
        raise HTTPException(500, f"Query failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mcp_url": HF_MCP_URL}

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    async def main():
        try:
            result = await query_hf(QueryRequest(
                query="deepseek r1", 
                kind="models", 
                limit=5, 
                include_cards=True
            ))
            
            print("=" * 50)
            print("QUERY RESULT:")
            print("=" * 50)
            print(f"Answer: {result.answer}")
            print("\n" + "=" * 50)
            print("SEARCH RESULTS:")
            print("=" * 50)
            
            for i, search_result in enumerate(result.search_results, 1):
                print(f"{i}. {search_result['repo_id']}")
                print(f"   Author: {search_result.get('author', 'Unknown')}")
                print(f"   Likes: {search_result.get('likes', 0)}")
                print(f"   Downloads: {search_result.get('downloads', 0)}")
                print(f"   Tags: {', '.join(search_result.get('tags', [])[:5])}")
                print(f"   URL: {search_result['url']}")
                print()
            
            if result.cards:
                print("=" * 50)
                print("DETAILED CARDS:")
                print("=" * 50)
                for card in result.cards:
                    print(f"Card for {card.get('repo_id', 'Unknown')}:")
                    if card.get('readme'):
                        readme_preview = card['readme'][:500] + "..." if len(card['readme']) > 500 else card['readme']
                        print(f"README: {readme_preview}")
                    print()
            
            print("=" * 50)
            print("SOURCES:")
            print("=" * 50)
            for source in result.sources:
                print(f"- {source}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())