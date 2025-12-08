# Arxiv Dataset Extraction Feature

This feature enhances the agentic workflow by automatically extracting training dataset information from arxiv papers for models discovered in the Neo4j tree search.

## Overview

When a user queries for a model, the agent will now:
1. Search HuggingFace for the model
2. Query Neo4j for related models (tree search)
3. Search HuggingFace for details on related models
4. **NEW:** Extract arxiv paper links from model cards and parse papers to find training datasets
5. Summarize findings including training datasets

## Architecture

### Components

#### 1. ArxivLinkExtractor
- **Purpose**: Extracts arxiv paper links from HuggingFace model cards
- **How it works**:
  - Fetches the model card page from HuggingFace
  - Searches for arxiv.org URLs in links and text content
  - Returns the arxiv paper URL if found

#### 2. ArxivPaperParser
- **Purpose**: Parses arxiv papers to extract dataset information
- **How it works**:
  - Downloads the PDF from arxiv
  - Extracts text from the first 8 pages (configurable)
  - Searches for:
    - Known datasets (ImageNet, COCO, SQuAD, etc.)
    - Dataset URLs (HuggingFace datasets, GitHub repos)
    - Context around dataset mentions
  - Returns structured dataset information

#### 3. ArxivDatasetExtractor
- **Purpose**: Orchestrates the extraction process
- **How it works**:
  - Processes multiple models in parallel (default: 5 concurrent)
  - Uses async/await for efficient I/O
  - Returns dataset information for all models

#### 4. extract_training_datasets (Agent Tool)
- **Purpose**: Provides the agent with access to the extraction functionality
- **How it works**:
  - Takes a list of model IDs as input
  - Calls ArxivDatasetExtractor
  - Returns structured JSON with arxiv links and datasets

## Usage

### Agent Workflow

The agent automatically uses this tool when processing a query. The workflow is:

```
User Query: "Show me BERT models"
    ↓
Step 1: search_huggingface("BERT models")
    ↓
Step 2: search_neo4j("bert-base-uncased")
    → Returns: [bert-base-uncased, bert-base-german, distilbert-base-uncased, ...]
    ↓
Step 3: search_huggingface() for connected models
    ↓
Step 4: extract_training_datasets([
    "bert-base-uncased",
    "bert-base-german",
    "distilbert-base-uncased"
])
    → For each model:
        1. Extract arxiv link from HF model card
        2. Download and parse PDF (first 8 pages)
        3. Find dataset mentions
    → Returns: {
        "bert-base-uncased": {
            "arxiv_url": "https://arxiv.org/abs/1810.04805",
            "datasets": [
                {"name": "bookcorpus", "url": null, "description": "..."},
                {"name": "wikipedia", "url": null, "description": "..."}
            ]
        },
        ...
    }
    ↓
Step 5: Agent summarizes findings
```

### Direct Usage (Python)

You can also use the extractor directly in Python code:

```python
from backend.routers.search.utils.arxiv_extractor import ArxivDatasetExtractor

# Create extractor instance
extractor = ArxivDatasetExtractor()

# Extract for multiple models (async)
import asyncio

async def main():
    results = await extractor.extract_for_models(
        model_ids=["bert-base-uncased", "gpt2"],
        max_concurrent=5
    )

    for model_id, info in results.items():
        print(f"\nModel: {model_id}")
        print(f"Arxiv: {info.arxiv_url}")
        print(f"Datasets: {len(info.datasets)}")
        for dataset in info.datasets:
            print(f"  - {dataset.name}: {dataset.url}")

asyncio.run(main())

# Or use synchronous wrapper
results = extractor.extract_sync(
    model_ids=["bert-base-uncased", "gpt2"],
    max_concurrent=5
)
```

## Performance Considerations

### Parallel Processing
- **Default concurrency**: 5 models at a time
- **Why**: Balance between speed and rate limiting
- **Adjustable**: Modify `max_concurrent` parameter

### Latency Optimization
1. **Concurrent requests**: Uses asyncio for parallel processing
2. **Connection pooling**: aiohttp session reuses connections
3. **Limited page reading**: Only reads first 8 pages of PDFs
4. **Efficient PDF parsing**: Uses pymupdf (faster than PyPDF2)

### Example Timings
For a query returning 10 models:
- **Sequential**: ~60-90 seconds (6-9s per model)
- **Parallel (5 concurrent)**: ~20-30 seconds
- **Speedup**: ~3x faster

## Dataset Detection

### Known Datasets Detected

The parser recognizes these well-known datasets:
- **Vision**: ImageNet, COCO, OpenImages, ADE20K, Cityscapes, Kinetics, ActivityNet
- **NLP**: SQuAD, GLUE, SuperGLUE, WMT, BookCorpus, Wikipedia, C4, Pile, RedPajama
- **Audio**: CommonVoice, LibriSpeech, VoxCeleb
- **Multimodal**: LAION

### URL Pattern Detection

The parser also extracts dataset URLs:
- HuggingFace dataset links: `huggingface.co/datasets/...`
- GitHub repositories
- Generic dataset URLs

### Context Extraction

For each dataset found, the parser extracts:
- **Name**: Dataset identifier
- **URL**: Link to dataset (if available)
- **Description**: Context around the dataset mention (up to 200 chars)

## Error Handling

The system gracefully handles errors:
- **No arxiv link found**: Returns empty datasets list
- **PDF download fails**: Logs warning, continues with other models
- **PDF parsing fails**: Logs error, returns empty datasets list
- **Network errors**: Logged, doesn't block other models

## Configuration

### Environment Variables
No additional environment variables required. Uses existing:
- `HF_TOKEN`: HuggingFace API token (optional, for private models)

### Code Configuration

#### Change max pages to read
```python
# In arxiv_extractor.py, modify ArxivPaperParser.parse_paper()
datasets = await self.paper_parser.parse_paper(arxiv_url, session, max_pages=10)
```

#### Change concurrency
```python
# In extract_datasets.py, modify max_concurrent parameter
results = extractor.extract_sync(model_ids, max_concurrent=10)
```

#### Add more known datasets
```python
# In arxiv_extractor.py, modify ArxivPaperParser.KNOWN_DATASETS
KNOWN_DATASETS = {
    'imagenet', 'coco', ...,
    'your-dataset-name',  # Add here
}
```

## Output Format

The tool returns a dictionary with this structure:

```json
{
  "model_id": {
    "arxiv_url": "https://arxiv.org/abs/1234.5678",
    "datasets": [
      {
        "name": "dataset-name",
        "url": "https://huggingface.co/datasets/dataset-name",
        "description": "Context around the dataset mention..."
      }
    ]
  }
}
```

## Limitations

1. **Only direct arxiv links**: Currently only extracts arxiv links directly referenced in the HF model card (not from external sources)
2. **First 8 pages only**: Datasets mentioned later in the paper won't be detected
3. **Pattern-based extraction**: May miss datasets with non-standard naming
4. **No citation parsing**: Doesn't parse bibliography to extract dataset papers

## Future Enhancements

Potential improvements:
1. Store arxiv links in Neo4j during scraping
2. Parse full paper (not just 8 pages)
3. Extract dataset version/configuration information
4. Parse dataset papers recursively
5. Use LLM-based extraction for better accuracy
6. Cache extracted dataset information
7. Add dataset size/statistics extraction

## Testing

### Manual Test

Create a test script:

```python
# test_arxiv_extraction.py
from backend.routers.search.utils.extract_datasets import extract_training_datasets

# Test with a known model
result = extract_training_datasets(["bert-base-uncased"])
print(result)
```

Run:
```bash
cd backend
python -c "from routers.search.utils.extract_datasets import extract_training_datasets; print(extract_training_datasets(['bert-base-uncased']))"
```

### Integration Test

Test the full agent workflow:
```bash
curl -X POST http://localhost:8000/flow/search \
  -H "Content-Type: application/json" \
  -d '{"query_val": "BERT models"}'
```

The response should include training dataset information in the summary.

## Dependencies

New dependencies added:
- `aiohttp>=3.11.0` - Async HTTP client
- `beautifulsoup4>=4.12.0` - HTML parsing

Existing dependencies used:
- `pymupdf>=1.24.0` - PDF parsing (already installed)

## Files Modified/Created

### New Files
1. `backend/routers/search/utils/arxiv_extractor.py` - Core extraction logic
2. `backend/routers/search/utils/extract_datasets.py` - Agent tool wrapper
3. `backend/routers/search/utils/ARXIV_DATASET_EXTRACTION.md` - This documentation

### Modified Files
1. `backend/routers/search/agent.py` - Added new tool and updated instructions
2. `backend/pyproject.toml` - Added aiohttp and beautifulsoup4 dependencies

## Maintenance

### Logging
The module uses Python logging with these levels:
- `INFO`: High-level operations (starting extraction, completion)
- `DEBUG`: Detailed operations (no arxiv link found, etc.)
- `WARNING`: Recoverable errors (failed to fetch, etc.)
- `ERROR`: Serious errors (parsing failures, etc.)

### Monitoring
Key metrics to monitor:
- Extraction success rate (models with arxiv links found)
- Average extraction time per model
- Dataset detection rate
- Error rates by type

## Support

For issues or questions:
1. Check logs for error messages
2. Verify HuggingFace model card has arxiv link
3. Test arxiv paper link manually
4. Adjust concurrency if hitting rate limits
