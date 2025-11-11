"""
Helper functions to fetch and parse arXiv papers directly without MCP.
"""
import requests
import arxiv
import fitz  # PyMuPDF
from typing import Optional, Dict
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def download_pdf(pdf_url: str, save_path: str) -> bool:
    """
    Download PDF from URL.

    Args:
        pdf_url: URL to the PDF file
        save_path: Path to save the downloaded PDF

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"PDF downloaded successfully to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        return False


def extract_text_from_pdf(pdf_path: str, max_pages: int = 30) -> Optional[str]:
    """
    Extract text from PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (to avoid token limits)

    Returns:
        Extracted text or None if error
    """
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        text = ""
        pages_to_read = min(len(doc), max_pages)

        for page_num in range(pages_to_read):
            page = doc[page_num]
            text += f"\n\n--- Page {page_num + 1} ---\n\n"
            text += page.get_text()

        doc.close()

        logger.info(f"Extracted {len(text)} characters from {pages_to_read} pages")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None


def analyze_paper_with_llm(paper_text: str, arxiv_id: str, title: str) -> str:
    """
    Use GPT-4o to analyze paper text and extract training data information.

    Args:
        paper_text: Full text extracted from the paper
        arxiv_id: arXiv paper ID
        title: Paper title

    Returns:
        Formatted training data information
    """
    try:
        logger.info(f"Analyzing paper with GPT-4o: {title}")

        # Truncate text if too long (GPT-4o has 128k token context, but we'll be conservative)
        max_chars = 80000  # Roughly 20k tokens
        if len(paper_text) > max_chars:
            logger.warning(f"Paper text too long ({len(paper_text)} chars), truncating to {max_chars}")
            paper_text = paper_text[:max_chars] + "\n\n[... rest of paper truncated ...]"

        prompt = f"""You are analyzing a research paper to extract detailed training data information.

Paper Title: {title}
arXiv ID: {arxiv_id}

Please carefully read through the paper text below and extract ALL training data information.

Focus on finding:
- **Exact dataset names** used for training/pretraining/fine-tuning
- **Dataset sources** (URLs, papers, where they came from)
- **Dataset sizes** (number of samples, images, text pairs, tokens, etc.)
- **Data collection methods** (web scraping, manual curation, synthetic generation, etc.)
- **Known issues** with the datasets (biases, legal problems, content policy violations)
- **Preprocessing steps** applied to the data
- **Train/val/test splits** if mentioned
- **Data filtering** or cleaning procedures
- **Ethical considerations** mentioned about the data

Format your response as:

### 📄 Training Data from Research Paper

**Paper Link:** https://arxiv.org/abs/{arxiv_id}

---

### Training Datasets

[List each dataset with details: name, size, source, purpose]

### Data Collection & Preprocessing

[Describe how data was collected and preprocessed]

### Known Issues & Considerations

[Any biases, ethical concerns, or legal issues mentioned]

### Additional Notes

[Any other relevant training data information]

If you cannot find specific training data information, be honest and state what sections you checked.

Paper text:

{paper_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a research paper analyst specializing in extracting training data information from ML/AI papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        analysis = response.choices[0].message.content
        logger.info(f"Successfully analyzed paper")
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing paper with LLM: {str(e)}")
        return f"⚠️ Error analyzing paper: {str(e)}"


def fetch_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, str]]:
    """
    Fetch arXiv paper details and full text using the arxiv Python library.

    Args:
        arxiv_id: arXiv paper ID (e.g., "1810.04805")

    Returns:
        Dict with paper title, abstract, pdf_url, and full_text (if available)
    """
    try:
        logger.info(f"Fetching arXiv paper {arxiv_id}")

        # Search for the paper by ID
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        paper_data = {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "abstract": paper.summary,
            "pdf_url": paper.pdf_url,
            "authors": [author.name for author in paper.authors],
            "published": str(paper.published),
        }

        logger.info(f"Successfully fetched paper: {paper.title}")
        return paper_data

    except Exception as e:
        logger.error(f"Error fetching arXiv paper {arxiv_id}: {str(e)}")
        return None


def extract_training_data_info(paper_data: Dict[str, str]) -> str:
    """
    Download PDF, extract text, and analyze with LLM to get training data information.

    Args:
        paper_data: Dictionary with paper information

    Returns:
        Formatted string with training data information extracted from full paper
    """
    if not paper_data:
        return "Could not fetch paper information."

    arxiv_id = paper_data['arxiv_id']
    pdf_path = f"/tmp/papers/{arxiv_id}.pdf"

    # Download PDF
    if not download_pdf(paper_data['pdf_url'], pdf_path):
        return f"""### ⚠️ Could Not Download Paper

Unable to download PDF for paper: {paper_data['title']}

**Paper Link:** https://arxiv.org/abs/{arxiv_id}

Please download and read the paper manually from the link above.
"""

    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path, max_pages=30)

    if not paper_text:
        return f"""### ⚠️ Could Not Extract Text from PDF

Downloaded PDF but could not extract text for paper: {paper_data['title']}

**Paper Link:** https://arxiv.org/abs/{arxiv_id}

Please download and read the paper manually from the link above.
"""

    # Analyze with LLM
    analysis = analyze_paper_with_llm(paper_text, arxiv_id, paper_data['title'])

    # Clean up PDF file
    try:
        os.remove(pdf_path)
        logger.info(f"Cleaned up PDF file: {pdf_path}")
    except Exception as e:
        logger.warning(f"Could not clean up PDF file: {str(e)}")

    return analysis
