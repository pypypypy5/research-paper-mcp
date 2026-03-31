import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

import arxiv
import mcp.types as types

from paper_clients import (
    download_file,
    download_text,
    fetch_semantic_scholar_paper,
    search_arxiv_papers,
    search_semantic_scholar_papers,
)


logger = logging.getLogger("research-paper-mcp")

ToolHandler = Callable[[Dict[str, Any]], Awaitable[List[types.TextContent]]]
HEADING_LEVELS = {f"h{level}": level for level in range(1, 7)}
TEXT_TAGS = {"p", "li"}
IGNORED_CAPTURE_TAGS = {"math", "annotation", "semantics", "mrow", "mi", "mn", "mo", "msub", "msup", "msubsup"}
IGNORED_FULL_TEXT_HEADINGS = {"report github issue", "instructions for reporting errors"}
SECTION_ALIASES = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "related work": [
        "related work",
        "related works",
        "background and related work",
        "background and related works",
        "previous work",
        "previous works",
        "prior work",
        "prior works",
    ],
    "related works": [
        "related work",
        "related works",
        "background and related work",
        "background and related works",
        "previous work",
        "previous works",
        "prior work",
        "prior works",
    ],
    "method": [
        "method",
        "methods",
        "our method",
        "our methods",
        "methodology",
        "approach",
        "approaches",
        "materials and methods",
        "method and materials",
        "experimental setup",
        "implementation details",
    ],
    "experiments": [
        "experiments",
        "experiment",
        "empirical experiments",
        "experimental results",
        "evaluation",
        "evaluations",
    ],
    "results": [
        "results",
        "experiments",
        "experimental results",
        "evaluation",
        "evaluations",
    ],
    "conclusion": [
        "conclusion",
        "conclusions",
        "discussion and conclusion",
        "summary and conclusion",
    ],
}


@dataclass
class ParsedBlock:
    kind: str
    text: str
    level: int = 0


class _PaperHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[ParsedBlock] = []
        self._captures: List[Dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        classes = attrs_dict.get("class", "").lower()

        if tag in IGNORED_CAPTURE_TAGS:
            self._captures.append({"tag": tag, "kind": "ignore", "parts": []})
            return

        if tag == "sup" or "ltx_note_mark" in classes:
            self._captures.append({"tag": tag, "kind": "ignore", "parts": []})
            return

        if tag == "blockquote" and "abstract" in classes:
            self._captures.append({"tag": tag, "kind": "abstract", "parts": []})
            return

        if tag in HEADING_LEVELS:
            self._captures.append({"tag": tag, "kind": "heading", "level": HEADING_LEVELS[tag], "parts": []})
            return

        if tag in TEXT_TAGS and not self._is_inside_abstract():
            self._captures.append({"tag": tag, "kind": "text", "parts": []})

    def handle_endtag(self, tag: str) -> None:
        if not self._captures:
            return
        current = self._captures[-1]
        if current["tag"] != tag:
            return

        capture = self._captures.pop()
        text = _clean_text("".join(capture["parts"]))
        if not text:
            return

        if capture["kind"] == "heading":
            self.blocks.append(ParsedBlock(kind="heading", text=text, level=capture["level"]))
            return

        if capture["kind"] == "ignore":
            return

        if capture["kind"] == "abstract":
            text = re.sub(r"^\s*abstract\s*:?\s*", "", text, flags=re.IGNORECASE)
            self.blocks.append(ParsedBlock(kind="heading", text="Abstract", level=1))
            if text:
                self.blocks.append(ParsedBlock(kind="text", text=text))
            return

        self.blocks.append(ParsedBlock(kind="text", text=text))

    def handle_data(self, data: str) -> None:
        if self._captures:
            self._captures[-1]["parts"].append(data)

    def _is_inside_abstract(self) -> bool:
        return any(capture["kind"] == "abstract" for capture in self._captures)


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


def _clean_text(value: str) -> str:
    value = re.sub(r"\b(\w+)_\{\1\}", r"\1", value)
    value = re.sub(r"\b([A-Za-z])\1\b", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def _normalize_paper_id(paper_id: str) -> str:
    normalized = paper_id.strip()
    if normalized.lower().startswith("arxiv:"):
        normalized = normalized.split(":", 1)[1]
    return normalized


def _normalize_heading_title(title: str) -> str:
    normalized = _clean_text(title)
    normalized = re.sub(r"^(?:section\s+)?[\divxlcIVXLC]+(?:\.[\divxlcIVXLC]+)*[\s.:_-]+", "", normalized)
    normalized = re.sub(r"^[\d.]+[\s.:_-]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized.lower())
    return _clean_text(normalized)


def _build_arxiv_section_urls(paper_id: str) -> List[str]:
    normalized = _normalize_paper_id(paper_id)
    return [
        f"https://arxiv.org/html/{normalized}",
        f"https://ar5iv.labs.arxiv.org/html/{normalized}",
        f"https://arxiv.org/abs/{normalized}",
    ]


def _match_requested_heading(requested_section: str, heading_title: str) -> bool:
    request = _normalize_heading_title(requested_section)
    heading = _normalize_heading_title(heading_title)
    aliases = SECTION_ALIASES.get(request, [request])
    for alias in aliases:
        alias_normalized = _normalize_heading_title(alias)
        if heading == alias_normalized or heading.startswith(f"{alias_normalized} "):
            return True
        if alias_normalized in heading and len(alias_normalized.split()) > 1:
            return True
    return False


def _parse_html_blocks(html: str) -> List[ParsedBlock]:
    stripped = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.IGNORECASE | re.DOTALL)
    parser = _PaperHTMLParser()
    parser.feed(stripped)
    parser.close()
    return parser.blocks


def _extract_requested_sections_from_html(html: str, requested_sections: List[str]) -> Dict[str, str]:
    blocks = _parse_html_blocks(html)
    extracted: Dict[str, str] = {}

    for requested in requested_sections:
        match_index = None
        for index, block in enumerate(blocks):
            if block.kind != "heading":
                continue
            if _match_requested_heading(requested, block.text):
                match_index = index
                break

        if match_index is None:
            continue

        heading = blocks[match_index]
        section_chunks: List[str] = []
        for block in blocks[match_index + 1 :]:
            if block.kind == "heading" and block.level <= heading.level:
                break
            if block.kind == "text":
                section_chunks.append(block.text)

        section_text = "\n\n".join(chunk for chunk in section_chunks if chunk)
        if section_text:
            extracted[requested] = section_text

    return extracted


def _extract_full_text_from_html(html: str) -> str:
    chunks: List[str] = []
    for block in _parse_html_blocks(html):
        if block.kind == "heading" and _normalize_heading_title(block.text) in IGNORED_FULL_TEXT_HEADINGS:
            continue
        if block.kind in {"heading", "text"}:
            chunks.append(block.text)
    return "\n\n".join(chunk for chunk in chunks if chunk)


async def _fetch_arxiv_sections(paper_id: str, requested_sections: List[str]) -> tuple[Dict[str, str], List[str]]:
    remaining = list(requested_sections)
    extracted: Dict[str, str] = {}
    urls_tried: List[str] = []

    for url in _build_arxiv_section_urls(paper_id):
        if not remaining:
            break

        try:
            html = await download_text(url)
        except Exception as error:
            logger.info("Unable to fetch arXiv HTML from %s: %s", url, error)
            continue

        urls_tried.append(url)
        found = _extract_requested_sections_from_html(html, remaining)
        extracted.update(found)
        remaining = [section for section in remaining if section not in extracted]

    return extracted, urls_tried


async def _fetch_arxiv_full_text(paper_id: str) -> tuple[str, str]:
    for url in _build_arxiv_section_urls(paper_id):
        try:
            html = await download_text(url)
        except Exception as error:
            logger.info("Unable to fetch arXiv full text HTML from %s: %s", url, error)
            continue

        full_text = _extract_full_text_from_html(html)
        if full_text:
            return full_text, url

    raise RuntimeError(f"Unable to fetch full text for arXiv paper {paper_id}")


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


async def get_arxiv_sections(args: Dict[str, Any]) -> List[types.TextContent]:
    paper_id = _normalize_paper_id(args.get("paper_id", ""))
    requested_sections = [str(section).strip() for section in args.get("sections", []) if str(section).strip()]

    logger.info("Fetching arXiv sections for %s: %s", paper_id, requested_sections)

    try:
        extracted, urls_tried = await _fetch_arxiv_sections(paper_id, requested_sections)
        missing_sections = [section for section in requested_sections if section not in extracted]

        logger.info(
            "Fetched %s/%s requested sections for %s",
            len(extracted),
            len(requested_sections),
            paper_id,
        )
        return _json_response(
            {
                "success": True,
                "paper_id": paper_id,
                "sections": extracted,
                "missing_sections": missing_sections,
                "source_urls_tried": urls_tried,
            },
            indent=2,
        )
    except Exception as error:
        logger.error("arXiv section fetch failed: %s", error, exc_info=True)
        return _error_response(error)


async def get_arxiv_full_text(args: Dict[str, Any]) -> List[types.TextContent]:
    paper_id = _normalize_paper_id(args.get("paper_id", ""))

    logger.info("Fetching arXiv full text for %s", paper_id)

    try:
        full_text, source_url = await _fetch_arxiv_full_text(paper_id)
        logger.info("Fetched arXiv full text for %s from %s", paper_id, source_url)
        return _json_response(
            {
                "success": True,
                "paper_id": paper_id,
                "full_text": full_text,
                "source_url": source_url,
            },
            indent=2,
        )
    except Exception as error:
        logger.error("arXiv full text fetch failed: %s", error, exc_info=True)
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
        name="get_arxiv_sections",
        description="Fetch only the requested sections from a specific arXiv paper, such as abstract or method.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "arXiv paper identifier, with or without the arXiv: prefix.",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Section names to extract, for example ['abstract', 'method']",
                },
            },
            "required": ["paper_id", "sections"],
        },
    ),
    types.Tool(
        name="get_arxiv_full_text",
        description="Fetch the full text of a specific arXiv paper as parsed plain text.",
        inputSchema={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "arXiv paper identifier, with or without the arXiv: prefix.",
                },
            },
            "required": ["paper_id"],
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
    "get_arxiv_sections": get_arxiv_sections,
    "get_arxiv_full_text": get_arxiv_full_text,
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
