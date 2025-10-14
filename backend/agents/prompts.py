# Prompt Class
from routers.api_models import QueryRequest

class Prompt:
    """
    A class that contains all the prompts for the agents.
    """ 

    def get_hf_search_prompt(request: QueryRequest):
        """Get the HF search prompt."""
        HF_SEARCH_PROMPT = (
            f"""
            You are an expert at converting user queries into effective Hugging Face Hub search terms.

            User's original query: "{request.query}"
            Search type: {request.kind}

            Convert this into 1-3 effective search keywords that will find relevant models/datasets/spaces on Hugging Face Hub. 
            Focus on:
            - Technical terms (e.g., "transformer", "bert", "gpt", "yolo", "resnet")
            - Model names or architectures
            - Task-specific terms (e.g., "text-classification", "image-generation", "summarization")
            - Popular frameworks (e.g., "pytorch", "tensorflow", "transformers")

            Return ONLY the search keywords, separated by spaces. No explanations or additional text.

            Examples:
            - "What are the most popular models?" → "transformer bert gpt"
            - "I need a model for image classification" → "image-classification resnet efficientnet"
            - "Show me Qwen models" → "qwen"
            - "Best models for text summarization" → "summarization t5 bart"
            """
        )
        return HF_SEARCH_PROMPT

    def get_hf_summarise_results_prompt(request: QueryRequest, optimized_query: str, results_summary: str, card_context: str):
        """Get the HF summarise results prompt."""
        HF_SUMMARISE_RESULTS_PROMPT = (
            f"""You are a helpful assistant that analyzes Hugging Face Hub search results.

            User's Original Query: "{request.query}"
            Optimized Search Query Used: "{optimized_query}"
            Search Type: {request.kind}

            Search Results:
            {chr(10).join(results_summary)}
            {card_context}

            Please provide:
            1. Any datasets from which the retrieved results were trained on if applicable
            2. Any upstream models of the retrieved results if applicable
            """
        )
        return HF_SUMMARISE_RESULTS_PROMPT