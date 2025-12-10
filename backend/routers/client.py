import asyncio
import json
import logging
import re
from typing import Any, Dict, List, AsyncGenerator

from agents import RunResultStreaming, Runner
from fastapi import APIRouter, Request
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel
from rich.logging import RichHandler
from starlette.responses import StreamingResponse

from .search import (
    compiler_agent,
    dataset_extractor_agent,
    dataset_risk_agent,
    hf_search_agent,
    neo4j_search_agent,
)
from .search.utils.dataset_risk import build_dataset_risk_context
from .search.utils.tool_state import (
    get_tool_result,
    set_request_context,
    set_progress_callback,
)

router = APIRouter(prefix="/backend/flow")

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


class Query(BaseModel):
    query_val: str


async def _collect_response_text(result: RunResultStreaming) -> str:
    """Consume a streaming run and return the concatenated response text."""
    chunks: List[str] = []
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            delta = event.data.delta
            if delta:
                chunks.append(delta)
    return "".join(chunks).strip()


def _extract_model_ids_from_text(text: str) -> List[str]:
    """Extract HuggingFace model IDs or dataset IDs from text using regex patterns."""
    # Pattern to match model/dataset IDs like "author/model-name" or "Qwen/Qwen3-4B" or "allenai/c4"
    # Look for patterns like [author/model], "author/model", bullet lists, or lines with model/dataset IDs
    patterns = [
        r"\*\*?\d+\.\s*\[([^/\]]+/[^\]]+)\]",  # **1. [author/model]
        r"-\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",  # - author/model-name (bullet list)
        r"\[([^/\]]+/[^\]]+)\]",  # [author/model]
        r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b",  # author/model-name (word boundary)
    ]

    entity_ids = []
    seen = set()

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Clean up the match
            entity_id = match.strip()
            # Basic validation: should have a slash and look like a model/dataset ID
            if "/" in entity_id and len(entity_id.split("/")) == 2:
                # Skip if it's clearly a URL or other non-entity patterns
                if (
                    "huggingface.co" not in entity_id.lower()
                    and "http" not in entity_id.lower()
                    and len(entity_id.split("/")[0]) > 0  # author exists
                    and len(entity_id.split("/")[1]) > 0
                ):  # entity name exists
                    if entity_id not in seen:
                        seen.add(entity_id)
                        entity_ids.append(entity_id)

    return entity_ids[:5]  # Return up to 5 entity IDs (models or datasets)


def _extract_model_ids_from_graph(graph: Any, limit: int = 10) -> List[str]:
    """Collect unique model IDs and dataset IDs from the Neo4j graph result."""
    if not graph:
        return []

    nodes = getattr(graph, "nodes", None)
    if not nodes:
        return []

    raw_nodes = getattr(nodes, "nodes", None) or []
    entity_ids: List[str] = []
    for node in raw_nodes:
        if isinstance(node, dict):
            entity_id = node.get("model_id") or node.get("dataset_id")
        else:
            entity_id = getattr(node, "model_id", None) or getattr(
                node, "dataset_id", None
            )

        if entity_id:
            entity_ids.append(entity_id)

    queried_entity_id = getattr(graph, "queried_model_id", None)
    if queried_entity_id:
        entity_ids.insert(0, queried_entity_id)

    seen = set()
    deduped: List[str] = []
    for eid in entity_ids:
        if eid not in seen:
            seen.add(eid)
            deduped.append(eid)
        if len(deduped) >= limit:
            break

    return deduped


def _serialize_graph_with_datasets(
    graph: Any, training_datasets: Dict[str, Any] | None
) -> Dict[str, Any] | None:
    """Serialize the Neo4j graph and attach training dataset info to matching nodes."""
    if not graph:
        return None

    graph_dict = graph.model_dump()
    dataset_map = training_datasets or {}

    if not isinstance(dataset_map, dict):
        return graph_dict

    nodes = graph_dict.get("nodes", {}).get("nodes", [])
    for node in nodes:
        model_id = node.get("model_id")
        if model_id and model_id in dataset_map:
            node["training_datasets"] = dataset_map[model_id]

    return graph_dict


@router.post("/search")
async def run_search(query: Query, request: Request):
    search_logger = logger.getChild("search")
    request.state.tool_results = {}
    request.state.original_query = query.query_val
    set_request_context(request)

    search_logger.info(f"Running multi-agent workflow for query: {query.query_val}")

    async def stream_response() -> AsyncGenerator[str, None]:
        """Stream stage status updates, compiler output, then metadata."""
        stage_summaries: Dict[str, str] = {}
        status_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def emit_status(message: str) -> None:
            """Push a status line to the queue."""
            await status_queue.put(message + "\n")

        async def run_workflow():
            try:
                # Stage 1: HuggingFace search agent
                await emit_status("Stage 1: Running HuggingFace search...")
                hf_result = Runner.run_streamed(
                    starting_agent=hf_search_agent,
                    input=query.query_val,
                )
                hf_summary = await _collect_response_text(hf_result)
                stage_summaries["huggingface_initial"] = hf_summary
                await emit_status("Stage 1 complete.")

                # Stage 2: Neo4j lineage agent
                await emit_status("Stage 2: Extracting Neo4j lineage...")
                hf_summary_text = hf_result.final_output_as(str)
                neo4j_result = Runner.run_streamed(
                    starting_agent=neo4j_search_agent,
                    input=hf_summary_text,
                )
                neo4j_summary = await _collect_response_text(neo4j_result)
                stage_summaries["neo4j_lineage"] = neo4j_summary
                await emit_status("Stage 2 complete.")
                neo4j_graph = get_tool_result("search_neo4j", request)

                # Fallback: If Neo4j agent didn't call the tool, try extracting model/dataset IDs and calling directly
                if not neo4j_graph:
                    search_logger.warning(
                        "Neo4j agent did not call search_neo4j tool, attempting fallback"
                    )
                    from .search.utils.search_neo4j import (
                        search_query_impl as search_neo4j_tool,
                    )

                    # Extract model/dataset IDs from HuggingFace summary
                    extracted_entity_ids = _extract_model_ids_from_text(hf_summary_text)
                    search_logger.info(
                        f"Extracted entity IDs from HF summary: {extracted_entity_ids}"
                    )
                    if extracted_entity_ids:
                        try:
                            # Try searching with the first entity ID (model or dataset)
                            first_entity_id = extracted_entity_ids[0]
                            search_logger.info(
                                f"Calling search_neo4j directly with entity_id: {first_entity_id}"
                            )
                            neo4j_graph = search_neo4j_tool(first_entity_id)
                            search_logger.info(
                                f"Fallback Neo4j search successful, found {len(neo4j_graph.nodes.nodes) if neo4j_graph else 0} nodes"
                            )
                        except Exception as fallback_error:
                            search_logger.error(
                                f"Fallback Neo4j search failed: {fallback_error}"
                            )

                # Check if model or dataset was found in Neo4j - if not, end early
                entity_ids = _extract_model_ids_from_graph(neo4j_graph)
                if not entity_ids or (
                    neo4j_graph and len(neo4j_graph.nodes.nodes) == 0
                ):
                    search_logger.warning(
                        "Model or dataset not found in Neo4j database, ending workflow at Stage 2"
                    )

                    # Prepare a user-friendly response
                    not_found_message = (
                        "I couldn't find this model or dataset in our Neo4j lineage database. "
                        "This could mean:\n"
                        "- The model/dataset hasn't been indexed yet\n"
                        "- The model/dataset ID may be incorrect\n"
                        "- The model/dataset exists on HuggingFace but doesn't have lineage relationships in our database\n\n"
                        f"Based on the HuggingFace search:\n{hf_summary}"
                    )

                    await status_queue.put(not_found_message)

                    # Send empty metadata to signal completion
                    metadata_payload: Dict[str, Any] = {
                        "neo4j_data": None,
                        "training_datasets": {},
                        "dataset_risk": {},
                        "models_analyzed": [],
                        "stage_summaries": {
                            "huggingface_initial": hf_summary,
                            "neo4j_lineage": "Model not found in Neo4j database",
                        },
                    }

                    metadata_json = json.dumps(metadata_payload)
                    await status_queue.put("\n\n<METADATA_START>")
                    await status_queue.put(metadata_json)
                    await status_queue.put("<METADATA_END>")
                    return

                dataset_task = None
                hf_followup_task = None
                # Stage 3: Arxiv dataset extraction agent (uses new tooling)
                # Only extract datasets for models, not for datasets themselves
                # Filter entity_ids to only include models (those with model_id, not dataset_id)
                model_ids: List[str] = []
                if neo4j_graph and neo4j_graph.nodes:
                    for node in neo4j_graph.nodes.nodes:
                        if isinstance(node, dict):
                            model_id = node.get("model_id")
                        else:
                            model_id = getattr(node, "model_id", None)
                        if model_id and model_id in entity_ids:
                            model_ids.append(model_id)

                if model_ids:
                    await emit_status("Stage 3: Extracting training datasets...")

                    # Set up progress callback for fine-grained updates
                    async def dataset_progress(message: str):
                        await emit_status(message)

                    set_progress_callback(dataset_progress)

                    dataset_payload = json.dumps(
                        {
                            "instruction": "Extract training datasets from arxiv papers for these HuggingFace models.",
                            "model_ids": model_ids,
                            "notes": "Always call extract_training_datasets with the provided list.",
                        }
                    )
                    dataset_result = Runner.run_streamed(
                        starting_agent=dataset_extractor_agent,
                        input=dataset_payload,
                    )
                    dataset_task = asyncio.create_task(
                        _collect_response_text(dataset_result)
                    )
                else:
                    stage_summaries["dataset_extraction"] = (
                        "No model IDs available to run dataset extraction."
                    )

                # Stage 4: Follow-up HuggingFace pass using Neo4j insights (run concurrently with Stage 3)
                # Note: We start this in parallel with Stage 3 for performance, but don't emit status yet
                try:
                    followup_input = neo4j_result.final_output_as(str)
                    hf_followup = Runner.run_streamed(
                        starting_agent=hf_search_agent,
                        input=followup_input,
                    )
                    hf_followup_task = asyncio.create_task(
                        _collect_response_text(hf_followup)
                    )
                except Exception as e:
                    search_logger.warning(
                        f"Follow-up HuggingFace agent failed to start: {e}"
                    )
                    stage_summaries["huggingface_followup"] = (
                        "Follow-up HuggingFace lookup failed."
                    )

                parallel_tasks: list[tuple[str, asyncio.Task[str]]] = []
                if dataset_task:
                    parallel_tasks.append(("dataset_extraction", dataset_task))
                if hf_followup_task:
                    parallel_tasks.append(("huggingface_followup", hf_followup_task))

                if parallel_tasks:
                    results = await asyncio.gather(
                        *(task for _, task in parallel_tasks), return_exceptions=True
                    )
                    for (label, _), result in zip(parallel_tasks, results):
                        if isinstance(result, Exception):
                            if label == "dataset_extraction":
                                search_logger.error(
                                    f"Dataset extractor agent failed: {result}"
                                )
                                stage_summaries[label] = "Dataset extraction failed."
                            else:
                                search_logger.warning(
                                    f"HuggingFace follow-up failed: {result}"
                                )
                                stage_summaries[label] = (
                                    "Follow-up HuggingFace lookup failed."
                                )
                        else:
                            stage_summaries[label] = result or (
                                "No additional data returned."
                                if label == "huggingface_followup"
                                else "Dataset extraction returned no content."
                            )

                if dataset_task:
                    await emit_status("Stage 3 complete.")
                if hf_followup_task:
                    await emit_status("Stage 4: Follow-up HuggingFace lookup complete.")

                training_dataset_map = get_tool_result(
                    "extract_training_datasets", request
                )

                dataset_risk_context = build_dataset_risk_context(
                    training_dataset_map
                    if isinstance(training_dataset_map, dict)
                    else {}
                )

                dataset_risk_summary = None
                if dataset_risk_context.get("models"):
                    await emit_status("Stage 5: Assessing dataset risk...")
                    risk_payload = json.dumps(
                        {
                            "query": query.query_val,
                            "risk_context": dataset_risk_context,
                            "guidance": "Emphasize synthetic/English-only/unknown datasets with clear risk levels.",
                        }
                    )
                    risk_result = Runner.run_streamed(
                        starting_agent=dataset_risk_agent,
                        input=risk_payload,
                    )
                    dataset_risk_summary = await _collect_response_text(risk_result)
                    stage_summaries["dataset_risk"] = dataset_risk_summary
                    await emit_status("Stage 5 complete.")
                else:
                    stage_summaries["dataset_risk"] = (
                        "No training datasets available to assess risk."
                    )

                neo4j_payload = _serialize_graph_with_datasets(
                    neo4j_graph,
                    (
                        training_dataset_map
                        if isinstance(training_dataset_map, dict)
                        else {}
                    ),
                )

                # Prepare metadata to send after streaming
                metadata_payload: Dict[str, Any] = {
                    "neo4j_data": neo4j_payload,
                    "training_datasets": training_dataset_map,
                    "dataset_risk": dataset_risk_context,
                    "dataset_risk_summary": dataset_risk_summary,
                    "models_analyzed": entity_ids,  # Include both models and datasets
                    "stage_summaries": stage_summaries,
                }

                search_logger.info(
                    f"Prepared metadata with neo4j_data: {neo4j_payload is not None}, nodes: {neo4j_payload.get('nodes', {}).get('nodes', [])[:2] if neo4j_payload else 'None'}"
                )

                # Stage 6: Compiler agent synthesizes findings - stream the response
                compiler_payload = {
                    "query": query.query_val,
                    "model_brief": stage_summaries.get("huggingface_initial"),
                    "neo4j_brief": stage_summaries.get("neo4j_lineage"),
                    "dataset_highlights": stage_summaries.get("dataset_extraction"),
                    "risk_notes": stage_summaries.get("dataset_risk"),
                    "instruction": "Respond with <=6 sentences across two sections: Model Findings and Dependency Watchlist.",
                }

                compiler_input = json.dumps(compiler_payload)
                await emit_status("Stage 6: Compiling response...")

                compiled_response = Runner.run_streamed(
                    starting_agent=compiler_agent,
                    input=compiler_input,
                )

                async for event in compiled_response.stream_events():
                    if event.type == "raw_response_event" and isinstance(
                        event.data, ResponseTextDeltaEvent
                    ):
                        delta = event.data.delta
                        if delta:
                            await status_queue.put(delta)

                # After text streaming completes, send metadata as JSON
                metadata_json = json.dumps(metadata_payload)
                search_logger.info(
                    f"Sending metadata JSON (length: {len(metadata_json)}), neo4j_data present: {metadata_payload.get('neo4j_data') is not None}"
                )
                await status_queue.put("\n\n<METADATA_START>")
                await status_queue.put(metadata_json)
                await status_queue.put("<METADATA_END>")

            except Exception as e:
                search_logger.error(f"Workflow failed: {e}")
                await status_queue.put(f"\n[error] {e}\n")
            finally:
                await status_queue.put(None)

        worker = asyncio.create_task(run_workflow())
        try:
            while True:
                chunk = await status_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not worker.done():
                worker.cancel()

    return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")
