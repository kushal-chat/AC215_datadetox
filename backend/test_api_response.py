"""Test script to verify API response includes dataset information."""

import asyncio
from routers.search.agent import search_agent
from routers.search.utils.tool_state import set_request_context
from agents import Runner
from unittest.mock import Mock


async def test_api_workflow():
    """Simulate the API workflow and check if dataset info is included."""

    # Create a mock request object
    mock_request = Mock()
    mock_request.state = Mock()
    mock_request.state.tool_results = {}
    mock_request.state.original_query = "BERT models"

    # Set the mock request in context
    set_request_context(mock_request)

    print("Running agent workflow for query: 'BERT models'")
    print("=" * 80)

    # Run the agent
    result = await Runner.run(search_agent, input="Query: BERT models")

    print("\n📝 Agent Response:")
    print("-" * 80)
    print(result.final_output_as(str))

    # Check if dataset extraction results were stored
    if hasattr(mock_request.state, "tool_results"):
        print("\n\n🔍 Tool Results Stored:")
        print("-" * 80)

        # Check for neo4j results
        if "search_neo4j" in mock_request.state.tool_results:
            print("✅ Neo4j results: Found")
            neo4j_data = mock_request.state.tool_results["search_neo4j"]
            if hasattr(neo4j_data, "nodes"):
                print(f"   Models found: {len(neo4j_data.nodes.nodes)}")

        # Check for dataset extraction results
        if "extract_training_datasets" in mock_request.state.tool_results:
            print("✅ Dataset extraction results: Found")
            dataset_data = mock_request.state.tool_results["extract_training_datasets"]
            print(f"   Type: {type(dataset_data)}")

            if isinstance(dataset_data, dict):
                print(f"   Models processed: {len(dataset_data)}")
                for model_id, info in list(dataset_data.items())[:2]:
                    print(f"\n   📦 {model_id}:")
                    print(f"      Arxiv: {info.get('arxiv_url', 'Not found')}")
                    print(f"      Datasets: {len(info.get('datasets', []))}")
                    for dataset in info.get("datasets", [])[:3]:
                        print(f"        - {dataset.get('name')}")
        else:
            print("❌ Dataset extraction results: Not found")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Testing API workflow with dataset extraction...\n")
    asyncio.run(test_api_workflow())
    print("\nTest completed!")
