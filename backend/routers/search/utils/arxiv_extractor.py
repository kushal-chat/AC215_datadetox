"""Arxiv paper link extraction and dataset information parsing."""

import asyncio
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import aiohttp
from bs4 import BeautifulSoup
import fitz  # pymupdf

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset found in a paper."""

    name: str
    url: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ModelPaperInfo:
    """Paper and dataset information for a model."""

    model_id: str
    arxiv_url: Optional[str] = None
    datasets: List[DatasetInfo] = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = []


class ArxivLinkExtractor:
    """Extracts arxiv paper links from HuggingFace model cards."""

    ARXIV_PATTERNS = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
        r"https?://arxiv\.org/(?:abs|pdf)/(\d+\.\d+)",
    ]

    async def extract_from_model_card(
        self, model_id: str, session: aiohttp.ClientSession
    ) -> Optional[str]:
        """
        Extract arxiv link from a HuggingFace model card.

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")
            session: aiohttp session for making requests

        Returns:
            Arxiv URL if found, None otherwise
        """
        try:
            # Fetch the model card page
            url = f"https://huggingface.co/{model_id}"
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    logger.debug(
                        f"Failed to fetch model card for {model_id}: {response.status}"
                    )
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Search for arxiv links in the page content
                # Check all links
                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    arxiv_id = self._extract_arxiv_id(href)
                    if arxiv_id:
                        return f"https://arxiv.org/abs/{arxiv_id}"

                # Also check plain text content for arxiv references
                text_content = soup.get_text()
                arxiv_id = self._extract_arxiv_id(text_content)
                if arxiv_id:
                    return f"https://arxiv.org/abs/{arxiv_id}"

                logger.debug(f"No arxiv link found for {model_id}")
                return None

        except Exception as e:
            logger.warning(f"Error extracting arxiv link for {model_id}: {e}")
            return None

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arxiv ID from text using regex patterns."""
        for pattern in self.ARXIV_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None


class ArxivPaperParser:
    """Parses arxiv papers to extract dataset information."""

    # Common dataset name patterns
    DATASET_INDICATORS = [
        "dataset",
        "corpus",
        "benchmark",
        "collection",
        "trained on",
        "fine-tuned on",
        "pretrained on",
    ]

    # Well-known datasets to look for
    KNOWN_DATASETS = {
        "imagenet",
        "coco",
        "openimages",
        "ade20k",
        "cityscapes",
        "squad",
        "glue",
        "superglue",
        "wmt",
        "commonvoice",
        "librispeech",
        "voxceleb",
        "kinetics",
        "activitynet",
        "wikihow",
        "c4",
        "pile",
        "redpajama",
        "laion",
        "bookcorpus",
        "wikipedia",
        "openwebtext",
        "cc-news",
    }

    # Dataset URL patterns
    DATASET_URL_PATTERNS = [
        r"huggingface\.co/datasets/([^\s\)]+)",
        r"github\.com/[^\s/]+/[^\s/]+",
        r"https?://[^\s]+dataset[^\s]*",
    ]

    def __init__(self, use_llm: bool = True):
        """
        Initialize parser with optional LLM extraction.

        Args:
            use_llm: Whether to use LLM-based extraction (default: True)
        """
        self.use_llm = use_llm
        if use_llm:
            try:
                from .arxiv_llm_extractor import LLMDatasetExtractor

                self.llm_extractor = LLMDatasetExtractor()
                if not self.llm_extractor.is_available():
                    logger.warning(
                        "LLM extraction not available, will use pattern matching"
                    )
                    self.llm_extractor = None
            except Exception as e:
                logger.warning(f"Failed to initialize LLM extractor: {e}")
                self.llm_extractor = None
        else:
            self.llm_extractor = None

    async def parse_paper(
        self,
        arxiv_url: str,
        session: aiohttp.ClientSession,
        max_pages: int = 8,
        model_id: str = "",
    ) -> List[DatasetInfo]:
        """
        Parse an arxiv paper to extract dataset information.

        Args:
            arxiv_url: URL to arxiv paper (abs or pdf)
            session: aiohttp session for making requests
            max_pages: Maximum number of pages to read (default: 8)
            model_id: HuggingFace model ID (for LLM context)

        Returns:
            List of DatasetInfo objects found in the paper
        """
        try:
            # Convert abs URL to pdf URL
            pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf"

            logger.info(f"Fetching arxiv paper: {pdf_url}")

            # Download the PDF
            async with session.get(
                pdf_url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch arxiv PDF: {response.status}")
                    return []

                pdf_content = await response.read()

            # Extract text from PDF first (needed for both methods)
            paper_text = self._extract_text_from_pdf(pdf_content, max_pages)

            # Try LLM extraction first if available
            if self.llm_extractor:
                logger.info(f"Using LLM extraction for {arxiv_url}")
                try:
                    llm_datasets = self.llm_extractor.extract_datasets(
                        paper_text, model_id, arxiv_url
                    )
                    if llm_datasets:
                        # Convert to DatasetInfo objects
                        datasets = []
                        for ds in llm_datasets:
                            datasets.append(
                                DatasetInfo(
                                    name=ds.name, url=ds.hf_url, description=ds.context
                                )
                            )
                        logger.info(
                            f"LLM extracted {len(datasets)} datasets from {arxiv_url}"
                        )
                        return datasets
                    else:
                        logger.warning(
                            "LLM returned no datasets, falling back to pattern matching"
                        )
                except Exception as e:
                    logger.error(
                        f"LLM extraction failed: {e}, falling back to pattern matching"
                    )

            # Fallback to pattern matching
            logger.info(f"Using pattern matching for {arxiv_url}")
            datasets = self._extract_datasets_from_text(paper_text)
            logger.info(f"Pattern matching found {len(datasets)} datasets in paper")
            return datasets

        except Exception as e:
            logger.error(f"Error parsing arxiv paper {arxiv_url}: {e}")
            return []

    def _extract_text_from_pdf(self, pdf_content: bytes, max_pages: int) -> str:
        """Extract text from PDF content."""
        try:
            # Open PDF from bytes using pymupdf
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

            # Read up to max_pages
            num_pages = min(pdf_document.page_count, max_pages)

            text_parts = []
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                text_parts.append(page.get_text())

            pdf_document.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def _extract_datasets_from_text(self, text: str) -> List[DatasetInfo]:
        """Extract dataset information from text using pattern matching."""
        datasets = {}  # Use dict to deduplicate by name

        # Extract datasets from text
        page_datasets = self._find_datasets_in_text(text)
        for dataset in page_datasets:
            # Deduplicate by name (case-insensitive)
            key = dataset.name.lower()
            if key not in datasets:
                datasets[key] = dataset

        return list(datasets.values())

    def _find_datasets_in_text(self, text: str) -> List[DatasetInfo]:
        """Find dataset mentions in text."""
        datasets = []
        text_lower = text.lower()

        # Look for known datasets
        for dataset_name in self.KNOWN_DATASETS:
            if dataset_name in text_lower:
                # Try to find context around the dataset name
                context = self._extract_context(text, dataset_name)
                url = self._extract_url_from_context(context)

                datasets.append(
                    DatasetInfo(
                        name=dataset_name,
                        url=url,
                        description=context[:200] if context else None,
                    )
                )

        # Look for explicit dataset mentions with URLs
        dataset_urls = self._extract_dataset_urls(text)
        for url in dataset_urls:
            # Extract dataset name from HuggingFace URL if possible
            if "huggingface.co/datasets/" in url:
                dataset_name = url.split("datasets/")[-1].split("/")[0]
                datasets.append(DatasetInfo(name=dataset_name, url=url))
            else:
                datasets.append(
                    DatasetInfo(
                        name=url.split("/")[-1],  # Use last part of URL as name
                        url=url,
                    )
                )

        return datasets

    def _extract_context(
        self, text: str, dataset_name: str, window: int = 200
    ) -> Optional[str]:
        """Extract context around a dataset mention."""
        text_lower = text.lower()
        index = text_lower.find(dataset_name.lower())
        if index == -1:
            return None

        # Get surrounding context
        start = max(0, index - window)
        end = min(len(text), index + len(dataset_name) + window)
        return text[start:end]

    def _extract_url_from_context(self, context: str) -> Optional[str]:
        """Extract URL from context text."""
        if not context:
            return None

        for pattern in self.DATASET_URL_PATTERNS:
            match = re.search(pattern, context)
            if match:
                return match.group(0)
        return None

    def _extract_dataset_urls(self, text: str) -> List[str]:
        """Extract all dataset-related URLs from text."""
        urls = []
        for pattern in self.DATASET_URL_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                urls.append(match.group(0))
        return urls


class ArxivDatasetExtractor:
    """Main orchestrator for extracting dataset information from arxiv papers."""

    def __init__(self, progress_callback=None):
        self.link_extractor = ArxivLinkExtractor()
        self.paper_parser = ArxivPaperParser()
        self.progress_callback = progress_callback

    async def extract_for_models(
        self, model_ids: List[str], max_concurrent: int = 8
    ) -> Dict[str, ModelPaperInfo]:
        """
        Extract dataset information from arxiv papers for multiple models.

        Args:
            model_ids: List of HuggingFace model IDs
            max_concurrent: Maximum number of concurrent requests (default: 5)

        Returns:
            Dictionary mapping model_id to ModelPaperInfo
        """
        results = {}

        # Create aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Process models with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_model(model_id: str) -> tuple[str, ModelPaperInfo]:
                async with semaphore:
                    return model_id, await self._extract_for_single_model(
                        model_id, session
                    )

            # Run all tasks concurrently
            tasks = [process_model(model_id) for model_id in model_ids]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in completed:
                if isinstance(result, Exception):
                    logger.error(f"Error processing model: {result}")
                    continue
                model_id, info = result
                results[model_id] = info

        return results

    async def _extract_for_single_model(
        self, model_id: str, session: aiohttp.ClientSession
    ) -> ModelPaperInfo:
        """Extract dataset information for a single model."""
        info = ModelPaperInfo(model_id=model_id)

        try:
            # Step 1: Extract arxiv link from model card
            logger.info(f"Extracting arxiv link for {model_id}")
            if self.progress_callback:
                await self.progress_callback(
                    f"Stage 3.1: Searching for paper link in {model_id}"
                )

            arxiv_url = await self.link_extractor.extract_from_model_card(
                model_id, session
            )
            info.arxiv_url = arxiv_url

            if not arxiv_url:
                logger.info(f"No arxiv link found for {model_id}")
                if self.progress_callback:
                    await self.progress_callback(
                        f"Stage 3.1: No paper found for {model_id}"
                    )
                return info

            # Step 2: Parse arxiv paper for dataset information
            logger.info(f"Parsing arxiv paper for {model_id}: {arxiv_url}")
            arxiv_id = arxiv_url.split("/")[-1]
            if self.progress_callback:
                await self.progress_callback(
                    f"Stage 3.2: Reading paper {arxiv_id} for {model_id}"
                )

            datasets = await self.paper_parser.parse_paper(
                arxiv_url, session, model_id=model_id
            )
            info.datasets = datasets

            logger.info(
                f"Completed extraction for {model_id}: found {len(datasets)} datasets"
            )

            if self.progress_callback:
                await self.progress_callback(
                    f"Stage 3.3: Found {len(datasets)} datasets for {model_id}"
                )

        except Exception as e:
            logger.error(f"Error extracting for {model_id}: {e}")
            if self.progress_callback:
                await self.progress_callback(f"Stage 3: Error processing {model_id}")

        return info

    def extract_sync(
        self, model_ids: List[str], max_concurrent: int = 5
    ) -> Dict[str, ModelPaperInfo]:
        """
        Synchronous wrapper for extract_for_models.

        Args:
            model_ids: List of HuggingFace model IDs
            max_concurrent: Maximum number of concurrent requests

        Returns:
            Dictionary mapping model_id to ModelPaperInfo
        """
        # Try to get existing event loop (agent framework already has one)
        try:
            asyncio.get_running_loop()
            # We're in an async context (FastAPI with uvloop) - run in separate thread
            import threading

            result_container = {}
            exception_container = {}

            def run_in_thread():
                """Run async code in a new thread with its own event loop."""
                try:
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)
                    try:
                        result = thread_loop.run_until_complete(
                            self.extract_for_models(model_ids, max_concurrent)
                        )
                        result_container["result"] = result
                    finally:
                        thread_loop.close()
                except Exception as e:
                    exception_container["error"] = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            # Check if there was an exception
            if "error" in exception_container:
                raise exception_container["error"]

            return result_container.get("result", {})

        except RuntimeError:
            # No event loop running, create new one
            return asyncio.run(self.extract_for_models(model_ids, max_concurrent))
