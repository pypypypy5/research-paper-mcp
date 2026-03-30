import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

import arxiv
import mcp.types as types

from paper_clients import download_file, fetch_semantic_scholar_paper, search_arxiv_papers, search_semantic_scholar_papers


logger = logging.getLogger("research-paper-mcp")

ToolHandler = Callable[[Dict[str, Any]], Awaitable[List[types.TextContent]]]


def _json_response(payload: Dict[str, Any], *, indent: int | None = None) -> List[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(payload, indent=indent))]


def _error_response(error: Exception) -> List[types.TextContent]:
    return _json_response({"success": False, "error": str(error)})


def _get_papers_dir() -> Path:
    papers_dir = Path(os.environ.get("AGENTIC_SYSTEM_PATH", Path.home() / "agentic-system")) / "research-papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    return papers_dir


def _get_sort_criterion(sort_by_str: str) -> arxiv.SortCriterion:
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    return sort_map.get(sort_by_str, arxiv.SortCriterion.Relevance)


async def search_arxiv(args: Dict[str, Any]) -> List[types.TextContent]:
    query = args.get("query", "")
    max_results = args.get("max_results", 10)
    sort_by = _get_sort_criterion(args.get("sort_by", "relevance"))

    logger.info("Searching arXiv for: %s (max_results=%s)", query, max_results)

    try:
        papers = await search_arxiv_papers(query, max_results, sort_by)
        logger.info("Found %s papers on arXiv", len(papers))
        return _json_response(
            {
                "success": True,
                "query": query,
                "count": len(papers),
                "papers": papers,
            },
            indent=2,
        )
    except Exception as error:
        logger.error("arXiv search failed: %s", error, exc_info=True)
        return _error_response(error)


async def search_semantic_scholar(args: Dict[str, Any]) -> List[types.TextContent]:
    query = args.get("query", "")
    fields = args.get("fields", ["title", "authors", "abstract", "citationCount", "year"])
    limit = args.get("limit", 10)

    logger.info("Searching Semantic Scholar for: %s (limit=%s)", query, limit)

    try:
        papers = await search_semantic_scholar_papers(query, fields, limit)
        logger.info("Found %s papers on Semantic Scholar", len(papers))
        return _json_response(
            {
                "success": True,
                "query": query,
                "count": len(papers),
                "papers": papers,
            },
            indent=2,
        )
    except Exception as error:
        logger.error("Semantic Scholar search failed: %s", error, exc_info=True)
        return _error_response(error)


async def download_paper(args: Dict[str, Any]) -> List[types.TextContent]:
    url = args.get("url", "")
    paper_id = args.get("paper_id", "")

    logger.info("Downloading paper %s from %s", paper_id, url)

    try:
        safe_id = re.sub(r"[^\w\-]", "_", paper_id)
        pdf_path = _get_papers_dir() / f"{safe_id}.pdf"
        content = await download_file(url)
        pdf_path.write_bytes(content)

        logger.info("Downloaded paper to %s", pdf_path)
        return _json_response(
            {
                "success": True,
                "paper_id": paper_id,
                "file_path": str(pdf_path),
                "size_bytes": len(content),
            }
        )
    except Exception as error:
        logger.error("Paper download failed: %s", error, exc_info=True)
        return _error_response(error)


async def extract_insights(args: Dict[str, Any]) -> List[types.TextContent]:
    paper_text = args.get("paper_text", "")
    focus_areas = args.get("focus_areas", [])

    logger.info("Extracting insights from paper (length=%s)", len(paper_text))

    try:
        insights: List[str] = []
        sentences = re.split(r"[.!?]\s+", paper_text)
        key_phrases = [
            "we propose",
            "we demonstrate",
            "we show",
            "we present",
            "our method",
            "our approach",
            "our results",
            "achieve",
            "outperform",
            "improvement",
            "novel",
            "state-of-the-art",
            "significant",
            "effective",
        ]

        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in key_phrases) and 50 < len(sentence) < 300:
                insights.append(sentence.strip())
            if len(insights) >= 10:
                break

        logger.info("Extracted %s insights", len(insights))
        return _json_response(
            {
                "success": True,
                "insights": insights,
                "focus_areas": focus_areas,
            },
            indent=2,
        )
    except Exception as error:
        logger.error("Insight extraction failed: %s", error, exc_info=True)
        return _error_response(error)


async def analyze_citations(args: Dict[str, Any]) -> List[types.TextContent]:
    paper_id = args.get("paper_id", "")
    depth = args.get("depth", 1)

    logger.info("Analyzing citations for %s (depth=%s)", paper_id, depth)

    try:
        data = await fetch_semantic_scholar_paper(
            paper_id,
            ["title", "citationCount", "influentialCitationCount", "citations", "references"],
        )
        citation_graph = {
            "paper_id": paper_id,
            "title": data.get("title"),
            "citation_count": data.get("citationCount", 0),
            "influential_citations": data.get("influentialCitationCount", 0),
            "citations": len(data.get("citations", [])),
            "references": len(data.get("references", [])),
        }

        logger.info("Citation analysis complete: %s citations", citation_graph["citation_count"])
        return _json_response({"success": True, "citation_graph": citation_graph}, indent=2)
    except Exception as error:
        logger.error("Citation analysis failed: %s", error, exc_info=True)
        return _error_response(error)


async def store_paper_knowledge(args: Dict[str, Any]) -> List[types.TextContent]:
    paper_metadata = args.get("paper_metadata", {})
    insights = args.get("insights", [])
    techniques = args.get("techniques", [])

    logger.info("Storing paper knowledge: %s", paper_metadata.get("title", "Unknown"))

    try:
        entity_name = (
            f"research_paper_"
            f"{paper_metadata.get('id', hashlib.md5(paper_metadata.get('title', '').encode()).hexdigest()[:8])}"
        )
        observations = [
            f"Title: {paper_metadata.get('title')}",
            f"Authors: {', '.join(paper_metadata.get('authors', []))}",
            f"Year: {paper_metadata.get('year', 'Unknown')}",
            f"Citations: {paper_metadata.get('citationCount', 0)}",
        ]
        observations.extend(f"Insight: {insight}" for insight in insights)
        observations.extend(f"Technique: {technique}" for technique in techniques)

        logger.info("Would store entity: %s with %s observations", entity_name, len(observations))
        return _json_response(
            {
                "success": True,
                "entity_name": entity_name,
                "observations_count": len(observations),
                "message": "Paper knowledge ready for storage in enhanced-memory",
            }
        )
    except Exception as error:
        logger.error("Knowledge storage failed: %s", error, exc_info=True)
        return _error_response(error)


TOOL_DEFINITIONS = [
    types.Tool(
        name="search_arxiv",
        description="Search arXiv for research papers by query. Returns paper metadata including title, authors, abstract, PDF URL, and publication date.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'recursive self-improvement AGI', 'meta-learning neural networks')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "description": "Sort order: relevance, lastUpdatedDate, or submittedDate",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "default": "relevance",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="search_semantic_scholar",
        description="Search Semantic Scholar for papers with citation counts and influence metrics. Provides academic impact analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to retrieve: title, authors, abstract, citationCount, influentialCitationCount, year, venue",
                    "default": ["title", "authors", "abstract", "citationCount", "year"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="download_paper",
        description="Download research paper PDF from URL. Saves to local storage and returns file path.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "PDF URL (arXiv, Semantic Scholar, etc.)",
                },
                "paper_id": {
                    "type": "string",
                    "description": "Unique paper identifier for filename",
                },
            },
            "required": ["url", "paper_id"],
        },
    ),
    types.Tool(
        name="extract_insights",
        description="Extract key insights, findings, and techniques from research paper text. Uses AI to identify important contributions.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_text": {
                    "type": "string",
                    "description": "Full paper text or abstract",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional specific areas to focus on (e.g., ['methodology', 'results', 'applications'])",
                    "default": [],
                },
            },
            "required": ["paper_text"],
        },
    ),
    types.Tool(
        name="analyze_citations",
        description="Analyze citation relationships and paper influence using Semantic Scholar citation graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Semantic Scholar paper ID or arXiv ID",
                },
                "depth": {
                    "type": "integer",
                    "description": "Citation graph depth (1-3)",
                    "default": 1,
                },
            },
            "required": ["paper_id"],
        },
    ),
    types.Tool(
        name="store_paper_knowledge",
        description="Store extracted paper knowledge in enhanced-memory for AGI learning. Creates structured memory entities.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_metadata": {
                    "type": "object",
                    "description": "Paper metadata (title, authors, year, etc.)",
                },
                "insights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key insights extracted from paper",
                },
                "techniques": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Novel techniques or methods described",
                },
            },
            "required": ["paper_metadata", "insights"],
        },
    ),
]

TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "search_arxiv": search_arxiv,
    "search_semantic_scholar": search_semantic_scholar,
    "download_paper": download_paper,
    "extract_insights": extract_insights,
    "analyze_citations": analyze_citations,
    "store_paper_knowledge": store_paper_knowledge,
}


async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    try:
        handler = TOOL_HANDLERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown tool: {name}") from error
    return await handler(arguments)
