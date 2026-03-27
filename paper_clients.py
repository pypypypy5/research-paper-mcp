import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import aiohttp
import arxiv

from env_loader import load_env_file

load_env_file()


ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=5, sock_connect=5, sock_read=15)
DEFAULT_MAX_RETRIES = int(os.environ.get("RESEARCH_PAPER_HTTP_MAX_RETRIES", "3"))
DEFAULT_RETRY_DELAY_SECONDS = float(os.environ.get("RESEARCH_PAPER_RETRY_DELAY_SECONDS", "2"))
ARXIV_MIN_INTERVAL_SECONDS = float(os.environ.get("ARXIV_MIN_INTERVAL_SECONDS", "3"))
DEFAULT_HEADERS = {
    "User-Agent": "research-paper-mcp/1.0 (+https://modelcontextprotocol.io)",
    "Accept": "application/json, application/atom+xml, text/xml",
}
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_arxiv_rate_limit_lock = asyncio.Lock()
_last_arxiv_request_started_at = 0.0


def _build_headers() -> Dict[str, str]:
    headers = dict(DEFAULT_HEADERS)
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _collapse_whitespace(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_arxiv_sort(sort_by: Any) -> str:
    sort_map = {
        arxiv.SortCriterion.Relevance: "relevance",
        arxiv.SortCriterion.LastUpdatedDate: "lastUpdatedDate",
        arxiv.SortCriterion.SubmittedDate: "submittedDate",
        "relevance": "relevance",
        "lastUpdatedDate": "lastUpdatedDate",
        "submittedDate": "submittedDate",
    }
    return sort_map.get(sort_by, "relevance")


def _parse_arxiv_feed(feed_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(feed_text)
    papers = []
    for entry in root.findall("atom:entry", ARXIV_NAMESPACES):
        entry_id = _collapse_whitespace(entry.findtext("atom:id", default="", namespaces=ARXIV_NAMESPACES))
        pdf_url = None
        for link in entry.findall("atom:link", ARXIV_NAMESPACES):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break

        papers.append(
            {
                "id": entry_id.split("/")[-1],
                "title": _collapse_whitespace(entry.findtext("atom:title", default="", namespaces=ARXIV_NAMESPACES)),
                "authors": [
                    _collapse_whitespace(author.text)
                    for author in entry.findall("atom:author/atom:name", ARXIV_NAMESPACES)
                    if _collapse_whitespace(author.text)
                ],
                "abstract": _collapse_whitespace(entry.findtext("atom:summary", default="", namespaces=ARXIV_NAMESPACES)),
                "pdf_url": pdf_url,
                "published": _collapse_whitespace(entry.findtext("atom:published", default="", namespaces=ARXIV_NAMESPACES)) or None,
                "updated": _collapse_whitespace(entry.findtext("atom:updated", default="", namespaces=ARXIV_NAMESPACES)) or None,
                "categories": [category.attrib.get("term") for category in entry.findall("atom:category", ARXIV_NAMESPACES)],
                "primary_category": (
                    entry.find("arxiv:primary_category", ARXIV_NAMESPACES).attrib.get("term")
                    if entry.find("arxiv:primary_category", ARXIV_NAMESPACES) is not None
                    else None
                ),
            }
        )
    return papers


async def _throttle_arxiv() -> None:
    global _last_arxiv_request_started_at

    async with _arxiv_rate_limit_lock:
        delay = ARXIV_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_arxiv_request_started_at)
        if delay > 0:
            await asyncio.sleep(delay)
        _last_arxiv_request_started_at = time.monotonic()


def _retry_delay_seconds(response: Optional[Any], attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), DEFAULT_RETRY_DELAY_SECONDS)
            except ValueError:
                pass
    return DEFAULT_RETRY_DELAY_SECONDS * (2 ** attempt)


async def _perform_request(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> Any:
    if url.startswith(ARXIV_API_URL):
        await _throttle_arxiv()
    assert session is not None
    return session.request(method, url, params=params)


async def _request_text(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> str:
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, headers=_build_headers())

    try:
        assert session is not None
        for attempt in range(DEFAULT_MAX_RETRIES + 1):
            try:
                async with await _perform_request(method, url, params=params, session=session) as response:
                    body = await response.text()
                    if response.status == 429 and attempt < DEFAULT_MAX_RETRIES:
                        await asyncio.sleep(_retry_delay_seconds(response, attempt))
                        continue
                    if response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status}: {body}")
                    return body
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= DEFAULT_MAX_RETRIES:
                    raise
                await asyncio.sleep(_retry_delay_seconds(None, attempt))
    finally:
        if owns_session and session is not None:
            await session.close()


async def search_arxiv_papers(
    query: str,
    max_results: int,
    sort_by: Any,
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[Dict[str, Any]]:
    feed = await _request_text(
        "GET",
        ARXIV_API_URL,
        params={
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": _normalize_arxiv_sort(sort_by),
            "sortOrder": "descending",
        },
        session=session,
    )
    return _parse_arxiv_feed(feed)


async def _request_json(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Any]:
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, headers=_build_headers())

    try:
        assert session is not None
        for attempt in range(DEFAULT_MAX_RETRIES + 1):
            try:
                async with await _perform_request(method, url, params=params, session=session) as response:
                    body = await response.text()
                    if response.status == 429 and attempt < DEFAULT_MAX_RETRIES:
                        await asyncio.sleep(_retry_delay_seconds(response, attempt))
                        continue
                    if response.status >= 400:
                        raise RuntimeError(f"HTTP {response.status}: {body}")
                    return json.loads(body) if body else {}
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= DEFAULT_MAX_RETRIES:
                    raise
                await asyncio.sleep(_retry_delay_seconds(None, attempt))
    finally:
        if owns_session and session is not None:
            await session.close()


async def _request_bytes(
    method: str,
    url: str,
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> bytes:
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, headers=_build_headers())

    try:
        assert session is not None
        for attempt in range(DEFAULT_MAX_RETRIES + 1):
            try:
                async with await _perform_request(method, url, session=session) as response:
                    content = await response.read()
                    if response.status == 429 and attempt < DEFAULT_MAX_RETRIES:
                        await asyncio.sleep(_retry_delay_seconds(response, attempt))
                        continue
                    if response.status >= 400:
                        error_text = content.decode("utf-8", errors="replace")
                        raise RuntimeError(f"HTTP {response.status}: {error_text}")
                    return content
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt >= DEFAULT_MAX_RETRIES:
                    raise
                await asyncio.sleep(_retry_delay_seconds(None, attempt))
    finally:
        if owns_session and session is not None:
            await session.close()


async def search_semantic_scholar_papers(
    query: str,
    fields: Iterable[str],
    limit: int,
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[Dict[str, Any]]:
    data = await _request_json(
        "GET",
        f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
        params={
            "query": query,
            "fields": ",".join(fields),
            "limit": limit,
        },
        session=session,
    )
    return data.get("data", [])


async def fetch_semantic_scholar_paper(
    paper_id: str,
    fields: Iterable[str],
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Any]:
    encoded_paper_id = quote(paper_id, safe="")
    return await _request_json(
        "GET",
        f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/{encoded_paper_id}",
        params={"fields": ",".join(fields)},
        session=session,
    )


async def download_file(url: str, *, session: Optional[aiohttp.ClientSession] = None) -> bytes:
    return await _request_bytes("GET", url, session=session)
