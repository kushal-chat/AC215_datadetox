"""Test script for arxiv dataset extraction."""

import asyncio
import logging
from routers.search.utils.arxiv_extractor import ArxivDatasetExtractor

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_extraction():
    """Test the arxiv dataset extraction for a few well-known models."""

    # Test with models that likely have arxiv papers
    test_models = [
        "bert-base-uncased",  # BERT paper
        "gpt2",  # GPT-2 paper
        "facebook/opt-125m",  # OPT paper
    ]

    logger.info(f"Testing arxiv extraction for {len(test_models)} models")
    logger.info(f"Models: {test_models}")

    extractor = ArxivDatasetExtractor()

    # Run extraction
    results = await extractor.extract_for_models(test_models, max_concurrent=3)

    # Display results
    print("\n" + "=" * 80)
    print("ARXIV DATASET EXTRACTION TEST RESULTS")
    print("=" * 80)

    for model_id, info in results.items():
        print(f"\n📦 Model: {model_id}")
        print(f"   Arxiv: {info.arxiv_url or 'Not found'}")

        if info.datasets:
            print(f"   Datasets found: {len(info.datasets)}")
            for dataset in info.datasets:
                print(f"     - {dataset.name}")
                if dataset.url:
                    print(f"       URL: {dataset.url}")
                if dataset.description:
                    desc = dataset.description[:100].replace("\n", " ")
                    print(f"       Context: {desc}...")
        else:
            print("   Datasets found: 0")

    print("\n" + "=" * 80)

    # Summary statistics
    total_models = len(results)
    models_with_arxiv = sum(1 for info in results.values() if info.arxiv_url)
    total_datasets = sum(len(info.datasets) for info in results.values())

    print("\nSummary:")
    print(f"  Total models tested: {total_models}")
    print(f"  Models with arxiv links: {models_with_arxiv}")
    print(f"  Total datasets found: {total_datasets}")
    print(f"  Average datasets per model: {total_datasets / total_models:.1f}")


def test_extraction_sync():
    """Synchronous wrapper for the test."""
    asyncio.run(test_extraction())


if __name__ == "__main__":
    print("Starting arxiv dataset extraction test...\n")
    test_extraction_sync()
    print("\nTest completed!")
