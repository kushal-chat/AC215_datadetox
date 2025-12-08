#!/usr/bin/env python3
"""Test script to query the backend and save responses for frontend testing."""

import requests
import json

# Backend endpoint
BACKEND_URL = "http://localhost:8000/flow/search"

# Test queries
queries = [
    ("qwen3 4b", "test_response_qwen3_fixed.json"),
    ("bert", "test_response_bert_fixed.json"),
]


def test_query(query_text: str, output_file: str):
    """Send a query to the backend and save the response."""
    print(f"\n{'=' * 60}")
    print(f"Testing query: '{query_text}'")
    print(f"{'=' * 60}")

    payload = {"query_val": query_text}

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()

        # Save to file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Response saved to {output_file}")

        # Print summary
        neo4j_data = data.get("neo4j_data", {})
        nodes = neo4j_data.get("nodes", {}).get("nodes", [])

        print("\nSummary:")
        print(f"  Total nodes: {len(nodes)}")

        # Check training_datasets field
        nodes_with_datasets = 0
        for node in nodes:
            if "training_datasets" in node:
                datasets = node["training_datasets"].get("datasets", [])
                if datasets:
                    nodes_with_datasets += 1
                    print(f"  ✅ {node['model_id']}: {len(datasets)} datasets")

        print(f"\n  Models with training_datasets: {nodes_with_datasets}/{len(nodes)}")

        return data

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("Testing backend with asyncio fix...")

    for query, output_file in queries:
        test_query(query, output_file)

    print(f"\n{'=' * 60}")
    print("Testing complete!")
    print(f"{'=' * 60}")
