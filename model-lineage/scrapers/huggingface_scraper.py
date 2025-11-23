"""HuggingFace Hub scraper for model information and relationships."""
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import HfApi, ModelInfo, DatasetInfo
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)

class HuggingFaceScraper:
    """Scraper for HuggingFace Hub models."""
    
    def __init__(self):
        settings.validate()
        self.api = HfApi(token=settings.HF_TOKEN)
        self.rate_limit_delay = settings.RATE_LIMIT_DELAY
    
    def scrape_all_models(self, 
                         limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scrape all models from HuggingFace Hub.
        
        Args:
            limit: Maximum number of models to scrape (None for all)
            
        Returns:
            Tuple of (models list, datasets list, relationships list)
        """
        logger.info("Starting HuggingFace model scraping...")
        
        models = []
        datasets = []
        relationships = []
        dataset_ids_seen = set()
        
        # Get all models
        try:
            model_list = self.api.list_models(
                limit=limit,
                sort="downloads",
                direction=-1,  # Most downloaded first
            )
            
            model_list = list(model_list)
            logger.info(f"Found {len(model_list)} models to process")
            
            # Process models with progress bar
            for model_info in tqdm(model_list, desc="Scraping models"):
                try:
                    model_data = self._extract_model_info(model_info)
                    models.append(model_data)
                    
                    # Extract model-to-model relationships
                    try:
                        model_rels = self._extract_relationships(model_info, model_data)
                        relationships.extend(model_rels)
                    except Exception as rel_error:
                        logger.debug(f"Could not extract relationships for {model_info.id}: {rel_error}")
                    
                    # Extract dataset relationships from model tags
                    try:
                        dataset_rels, new_datasets = self._extract_dataset_relationships_from_model(model_info, model_data)
                        relationships.extend(dataset_rels)
                        for dataset in new_datasets:
                            if dataset["dataset_id"] not in dataset_ids_seen:
                                datasets.append(dataset)
                                dataset_ids_seen.add(dataset["dataset_id"])
                    except Exception as dataset_error:
                        logger.debug(f"Could not extract dataset relationships for {model_info.id}: {dataset_error}")
                    
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Error processing model {model_info.id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
        
        logger.info(f"Scraped {len(models)} models, {len(datasets)} datasets, and {len(relationships)} relationships")
        return models, datasets, relationships
    
    def _extract_model_info(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Extract relevant information from a model."""
        created_at = getattr(model_info, "created_at", None)
        updated_at = getattr(model_info, "updated_at", None)
        
        return {
            "model_id": model_info.id,
            "author": model_info.author,
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "tags": model_info.tags or [],
            "library_name": getattr(model_info, "library_name", None),
            "pipeline_tag": getattr(model_info, "pipeline_tag", None),
            "private": model_info.private,
            "sha": model_info.sha,
            "created_at": created_at.isoformat() if created_at else None,
            "updated_at": updated_at.isoformat() if updated_at else None,
            "url": f"https://huggingface.co/{model_info.id}",
        }
    
    def _extract_relationships(self, model_info: ModelInfo, model_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relationships from HuggingFace model tree.
        Only extracts: finetuned, adapters, merges, quantizations
        Uses the model page HTML to parse the model tree section.
        """
        relationships = []
        model_id = model_data["model_id"]
        
        try:
            # Get base model from model card first
            base_model = self._get_base_model_from_card(model_info)
            
            if base_model:
                # Determine relationship type by checking the base model's tree
                relationship_type = self._get_relationship_type_from_tree(model_id, base_model)
                
                if relationship_type:
                    relationships.append({
                        "source": model_id,
                        "target": base_model,
                        "relationship_type": relationship_type,
                        "source_type": "model",
                        "target_type": "model",
                    })
            # If no base model found, this is likely a base model itself
                
        except Exception as e:
            logger.debug(f"Could not extract relationships for {model_info.id}: {e}")
        
        return relationships
    
    def _extract_dataset_relationships_from_model(self, model_info: ModelInfo, model_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract dataset relationships from model tags with 'dataset:' prefix.
        
        Returns:
            Tuple of (relationships list, datasets list)
        """
        relationships = []
        datasets = []
        model_id = model_data["model_id"]
        
        # Extract datasets from tags
        dataset_tags = [tag for tag in model_data.get("tags", []) if tag.startswith("dataset:")]
        
        for tag in dataset_tags:
            # Remove 'dataset:' prefix
            dataset_id = tag.replace("dataset:", "").strip()
            if dataset_id:
                # Normalize dataset ID format
                # Tags might be like "dataset:dataset-name" or "dataset:author/dataset-name"
                if "/" not in dataset_id:
                    # If no author, try to find it from common dataset patterns
                    # Many datasets are under their own namespace or common orgs
                    # For now, keep as-is and let the dataset scraping handle it
                    pass
                
                relationships.append({
                    "source": model_id,
                    "target": dataset_id,
                    "relationship_type": "trained_on",
                    "source_type": "model",
                    "target_type": "dataset",
                })
                
                # Add dataset to list (minimal info, will be enriched during scraping)
                author = dataset_id.split("/")[0] if "/" in dataset_id else None
                datasets.append({
                    "dataset_id": dataset_id,
                    "author": author,
                    "downloads": None,
                    "tags": [],
                })
        
        return relationships, datasets
    
    def scrape_datasets(self, dataset_ids: List[str], limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scrape datasets and extract relationships from dataset cards.
        
        Args:
            dataset_ids: List of dataset IDs to scrape
            limit: Maximum number of datasets to scrape (None for all)
            
        Returns:
            Tuple of (datasets list, relationships list)
        """
        logger.info(f"Starting dataset scraping for {len(dataset_ids)} datasets...")
        
        datasets = []
        relationships = []
        
        dataset_list = dataset_ids[:limit] if limit else dataset_ids
        
        for dataset_id in tqdm(dataset_list, desc="Scraping datasets"):
            try:
                # Normalize dataset ID if needed
                if "/" not in dataset_id:
                    # Try to find dataset by searching - for now, skip if no author
                    logger.debug(f"Skipping dataset {dataset_id} - no author specified")
                    continue
                
                # Get dataset info
                dataset_info = self.api.dataset_info(dataset_id)
                dataset_data = self._extract_dataset_info(dataset_info)
                datasets.append(dataset_data)
                
                # Extract relationships from dataset card
                try:
                    dataset_rels = self._extract_relationships_from_dataset_card(dataset_id)
                    relationships.extend(dataset_rels)
                except Exception as rel_error:
                    logger.debug(f"Could not extract relationships from dataset {dataset_id}: {rel_error}")
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error processing dataset {dataset_id}: {e}")
                continue
        
        logger.info(f"Scraped {len(datasets)} datasets and {len(relationships)} relationships")
        return datasets, relationships
    
    def _extract_dataset_info(self, dataset_info: DatasetInfo) -> Dict[str, Any]:
        """Extract relevant information from a dataset."""
        created_at = getattr(dataset_info, "created_at", None)
        updated_at = getattr(dataset_info, "updated_at", None)
        
        return {
            "dataset_id": dataset_info.id,
            "author": dataset_info.author,
            "downloads": getattr(dataset_info, "downloads", None),
            "tags": dataset_info.tags or [],
            "created_at": created_at.isoformat() if created_at else None,
            "updated_at": updated_at.isoformat() if updated_at else None,
        }
    
    def _extract_relationships_from_dataset_card(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Extract relationships from dataset card showing models trained on this dataset.
        Looks for the "Models trained or fine-tuned on [dataset]" section.
        """
        relationships = []
        
        try:
            dataset_url = f"https://huggingface.co/datasets/{dataset_id}"
            headers = {}
            if settings.HF_TOKEN:
                headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"
            
            response = requests.get(dataset_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for section with models trained on this dataset
            # The section might be titled "Models trained or fine-tuned on [dataset]"
            # or similar variations
            headings = soup.find_all(['h2', 'h3', 'h4'], string=lambda text: text and (
                'model' in text.lower() and ('train' in text.lower() or 'fine-tun' in text.lower() or 'finetun' in text.lower())
            ))
            
            for heading in headings:
                # Find the parent container
                container = heading.find_next_sibling(['div', 'section', 'ul', 'ol'])
                if not container:
                    container = heading.parent
                
                # Look for model links
                model_links = container.find_all('a', href=re.compile(r'/models/'))
                
                for link in model_links:
                    href = link.get('href', '')
                    # Extract model ID from href (e.g., /models/author/model-name)
                    match = re.search(r'/models/([^/]+/[^/]+)', href)
                    if match:
                        model_id = match.group(1)
                        relationships.append({
                            "source": model_id,
                            "target": dataset_id,
                            "relationship_type": "trained_on",
                            "source_type": "model",
                            "target_type": "dataset",
                        })
            
            # Also check for JSON data in script tags (common pattern on HF)
            script_tags = soup.find_all('script', type='application/json')
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    # Look for model references in the data structure
                    # This is a heuristic - HF's structure may vary
                    if isinstance(data, dict):
                        models = data.get('models', [])
                        if isinstance(models, list):
                            for model in models:
                                if isinstance(model, dict) and 'id' in model:
                                    model_id = model['id']
                                    relationships.append({
                                        "source": model_id,
                                        "target": dataset_id,
                                        "relationship_type": "trained_on",
                                        "source_type": "model",
                                        "target_type": "dataset",
                                    })
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        except Exception as e:
            logger.debug(f"Error extracting relationships from dataset card {dataset_id}: {e}")
        
        return relationships
    
    def _get_base_model_from_card(self, model_info: ModelInfo) -> Optional[str]:
        """Get base model from model card YAML data."""
        try:
            from huggingface_hub import ModelCard
            
            card = ModelCard.load(model_info.id)
            card_data = card.data if hasattr(card, 'data') else {}
            
            base_model = (
                card_data.get("base_model") or 
                card_data.get("base_model_name") or
                card_data.get("base_model_config")
            )
            
            if base_model:
                # Normalize base model ID
                if "/" not in base_model and model_info.author:
                    base_model = f"{model_info.author}/{base_model}"
                return base_model
        except Exception as e:
            logger.debug(f"Could not get base model from card: {e}")
        
        return None
    
    def _get_relationship_type_from_tree(self, model_id: str, base_model: str) -> Optional[str]:
        """
        Determine relationship type by checking the base model's siblings API.
        The HuggingFace API provides sibling models categorized by type.
        """
        try:
            # Use HuggingFace API to get siblings/related models
            # The siblings endpoint shows finetuned, adapters, merges, quantizations
            siblings_url = f"https://huggingface.co/api/models/{base_model}/siblings"
            
            headers = {}
            if settings.HF_TOKEN:
                headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"
            
            response = requests.get(siblings_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                siblings_data = response.json()
                
                # Check each category for the current model
                # The API might return different structures, so we check multiple possible formats
                
                # Format: Direct lists
                for category in ["finetuned", "adapters", "merges", "quantizations"]:
                    category_models = siblings_data.get(category, [])
                    if isinstance(category_models, list):
                        for sibling in category_models:
                            sibling_id = sibling.get("id") if isinstance(sibling, dict) else sibling
                            if sibling_id == model_id:
                                return category
                
                # If not found in siblings, infer from name
                return self._infer_relationship_type_from_name(model_id, base_model)
            else:
                # Fallback: try to parse from the model page HTML
                return self._get_relationship_type_from_html(model_id, base_model)
                
        except Exception as e:
            logger.debug(f"Error getting relationship type from tree: {e}")
            return self._infer_relationship_type_from_name(model_id, base_model)
    
    def _get_relationship_type_from_html(self, model_id: str, base_model: str) -> Optional[str]:
        """
        Fallback: Parse HTML to find relationship type in model tree section.
        """
        try:
            # Fetch the base model's page
            base_model_url = f"https://huggingface.co/{base_model}"
            headers = {}
            if settings.HF_TOKEN:
                headers["Authorization"] = f"Bearer {settings.HF_TOKEN}"
            
            response = requests.get(base_model_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Look for model tree JSON data embedded in the page
                # HuggingFace often embeds JSON data in script tags
                import re
                import json
                
                # Try to find embedded JSON with model tree data
                json_match = re.search(r'<script[^>]*id="model-tree-data"[^>]*>(.*?)</script>', response.text, re.DOTALL)
                if json_match:
                    try:
                        tree_data = json.loads(json_match.group(1))
                        for category in ["finetuned", "adapters", "merges", "quantizations"]:
                            models = tree_data.get(category, [])
                            if any(m.get("id") == model_id or m == model_id for m in models):
                                return category
                    except json.JSONDecodeError:
                        pass
                
                model_id_short = model_id.split("/")[-1]
                for category in ["Finetuned", "Adapters", "Merges", "Quantizations"]:
                    pattern = rf'{category}.*?{re.escape(model_id_short)}'
                    if re.search(pattern, response.text, re.IGNORECASE | re.DOTALL):
                        return category.lower()
                
                # If not found, infer from name
                return self._infer_relationship_type_from_name(model_id, base_model)
            else:
                return self._infer_relationship_type_from_name(model_id, base_model)
                
        except Exception as e:
            logger.debug(f"Error parsing HTML: {e}")
            return self._infer_relationship_type_from_name(model_id, base_model)
    
    def _infer_relationship_type_from_name(self, model_id: str, base_model: str) -> Optional[str]:
        """
        Infer relationship type from model naming patterns.
        This is a fallback when model tree API doesn't provide clear categorization.
        """
        model_name_lower = model_id.lower()
        base_name_lower = base_model.lower()
        
        # Quantization patterns (most specific)
        quant_patterns = ["-8bit", "-4bit", "-gguf", "-gptq", "-awq", "-fp8", "-fp4", "-quantized"]
        if any(pattern in model_name_lower for pattern in quant_patterns):
            return "quantizations"
        
        # Adapter patterns
        adapter_patterns = ["-adapter", "-lora", "-peft", "-adapterhub"]
        if any(pattern in model_name_lower for pattern in adapter_patterns):
            return "adapters"
        
        # Merge patterns
        merge_patterns = ["-merge", "-merged", "-soup"]
        if any(pattern in model_name_lower for pattern in merge_patterns):
            return "merges"
        
        # Finetuned (default if base model exists and no other pattern matches)
        if base_model and base_model != model_id:
            return "finetuned"
        
        return None
    
    def _extract_from_model_card(self, model_info: ModelInfo, model_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback method: Extract base model from model card YAML.
        Used when model tree API is not available.
        """
        relationships = []
        model_id = model_data["model_id"]
        
        try:
            from huggingface_hub import ModelCard
            
            card = ModelCard.load(model_info.id)
            card_data = card.data if hasattr(card, 'data') else {}
            
            base_model = (
                card_data.get("base_model") or 
                card_data.get("base_model_name") or
                card_data.get("base_model_config")
            )
            
            if base_model:
                # Normalize base model ID
                if "/" not in base_model and model_info.author:
                    base_model = f"{model_info.author}/{base_model}"
                
                # Infer relationship type from name
                relationship_type = self._infer_relationship_type_from_name(model_id, base_model)
                
                if relationship_type:
                    relationships.append({
                        "source": model_id,
                        "target": base_model,
                        "relationship_type": relationship_type,
                        "source_type": "model",
                        "target_type": "model",
                    })
        except Exception as e:
            logger.debug(f"Could not extract from model card for {model_info.id}: {e}")
        
        return relationships
    
    def scrape_model_by_id(self, model_id: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Scrape a specific model by ID.
        
        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")
            
        Returns:
            Tuple of (model dict, relationships list)
        """
        try:
            model_info = self.api.model_info(model_id)
            model_data = self._extract_model_info(model_info)
            relationships = self._extract_relationships(model_info, model_data)
            return model_data, relationships
        except Exception as e:
            logger.error(f"Error scraping model {model_id}: {e}")
            raise

