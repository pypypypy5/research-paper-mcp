import asyncio
import json
import os
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import aiohttp
import arxiv

from env_loader import load_env_file

load_env_file()


SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=5, sock_connect=5, sock_read=15)
DEFAULT_MAX_RETRIES = int(os.environ.get("RESEARCH_PAPER_HTTP_MAX_RETRIES", "3"))
DEFAULT_RETRY_DELAY_SECONDS = float(os.environ.get("RESEARCH_PAPER_RETRY_DELAY_SECONDS", "2"))
ARXIV_MIN_INTERVAL_SECONDS = float(os.environ.get("ARXIV_MIN_INTERVAL_SECONDS", "3"))
DEFAULT_HEADERS = {
    "User-Agent": "research-paper-mcp/1.0 (+https://modelcontextprotocol.io)",
    "Accept": "application/json, application/atom+xml, text/xml",
}
_arxiv_client_lock = asyncio.Lock()
_arxiv_client: Optional[arxiv.Client] = None


def _build_headers() -> Dict[str, str]:
    headers = dict(DEFAULT_HEADERS)
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _serialize_arxiv_result(result: Any) -> Dict[str, Any]:
    return {
        "id": result.entry_id.split("/")[-1],
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "abstract": result.summary,
        "pdf_url": result.pdf_url,
        "published": result.published.isoformat() if result.published else None,
        "updated": result.updated.isoformat() if result.updated else None,
        "categories": result.categories,
        "primary_category": result.primary_category,
    }


def _get_arxiv_client() -> arxiv.Client:
    global _arxiv_client
    if _arxiv_client is None:
        _arxiv_client = arxiv.Client(
            page_size=100,
            delay_seconds=ARXIV_MIN_INTERVAL_SECONDS,
            num_retries=DEFAULT_MAX_RETRIES,
        )
    return _arxiv_client


def _search_arxiv_sync(query: str, max_results: int, sort_by: Any) -> List[Dict[str, Any]]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
    )
    return [_serialize_arxiv_result(result) for result in _get_arxiv_client().results(search)]


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
    assert session is not None
    return session.request(method, url, params=params)


async def search_arxiv_papers(
    query: str,
    max_results: int,
    sort_by: Any,
) -> List[Dict[str, Any]]:
    async with _arxiv_client_lock:
        return await asyncio.to_thread(_search_arxiv_sync, query, max_results, sort_by)


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


async def _request_text(
    method: str,
    url: str,
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> str:
    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, headers=_build_headers())

    try:
        assert session is not None
        for attempt in range(DEFAULT_MAX_RETRIES + 1):
            try:
                async with await _perform_request(method, url, session=session) as response:
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


async def download_text(url: str, *, session: Optional[aiohttp.ClientSession] = None) -> str:
    return await _request_text("GET", url, session=session)
