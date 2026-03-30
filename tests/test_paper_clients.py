import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import arxiv

import paper_clients


class FakeResponse:
    def __init__(self, status=200, text_data="", bytes_data=b"", headers=None):
        self.status = status
        self._text_data = text_data
        self._bytes_data = bytes_data
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text_data

    async def read(self):
        return self._bytes_data


class FakeSession:
    def __init__(self, response):
        self.responses = response if isinstance(response, list) else [response]
        self.calls = []

    def request(self, method, url, params=None):
        self.calls.append({"method": method, "url": url, "params": params})
        return self.responses.pop(0)


class PaperClientsAsyncTests(unittest.IsolatedAsyncioTestCase):
    @patch("paper_clients.asyncio.to_thread", new_callable=AsyncMock)
    async def test_search_arxiv_papers_uses_threaded_library_call(self, to_thread_mock):
        to_thread_mock.return_value = [{"id": "1234.5678v1"}]

        papers = await paper_clients.search_arxiv_papers("agentic systems", 3, arxiv.SortCriterion.Relevance)

        self.assertEqual(papers, [{"id": "1234.5678v1"}])
        to_thread_mock.assert_awaited_once_with(
            paper_clients._search_arxiv_sync,
            "agentic systems",
            3,
            arxiv.SortCriterion.Relevance,
        )

    @patch("paper_clients.arxiv.Search")
    @patch("paper_clients._get_arxiv_client")
    def test_search_arxiv_sync_serializes_results(self, get_client_mock, search_cls):
        get_client_mock.return_value.results.return_value = [
            SimpleNamespace(
                entry_id="http://arxiv.org/abs/1234.5678v1",
                title="Test Paper",
                authors=[SimpleNamespace(name="Alice"), SimpleNamespace(name="Bob")],
                summary="Short abstract",
                pdf_url="https://arxiv.org/pdf/1234.5678v1.pdf",
                published=datetime(2024, 1, 2, 3, 4, 5),
                updated=datetime(2024, 2, 3, 4, 5, 6),
                categories=["cs.AI", "cs.LG"],
                primary_category="cs.AI",
            )
        ]

        papers = paper_clients._search_arxiv_sync("agentic systems", 3, arxiv.SortCriterion.Relevance)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["id"], "1234.5678v1")
        self.assertEqual(papers[0]["authors"], ["Alice", "Bob"])
        self.assertEqual(papers[0]["primary_category"], "cs.AI")
        search_cls.assert_called_once_with(
            query="agentic systems",
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )

    async def test_search_semantic_scholar_papers_returns_data(self):
        response = FakeResponse(
            status=200,
            text_data=json.dumps({"data": [{"title": "Attention Is All You Need"}]}),
        )
        session = FakeSession(response)

        papers = await paper_clients.search_semantic_scholar_papers(
            "attention",
            ["title"],
            1,
            session=session,
        )

        self.assertEqual(papers, [{"title": "Attention Is All You Need"}])
        self.assertEqual(session.calls[0]["url"], f"{paper_clients.SEMANTIC_SCHOLAR_BASE_URL}/paper/search")
        self.assertEqual(session.calls[0]["params"], {"query": "attention", "fields": "title", "limit": 1})

    async def test_fetch_semantic_scholar_paper_url_encodes_identifier(self):
        session = FakeSession(FakeResponse(status=200, text_data=json.dumps({"title": "Encoded"})))

        await paper_clients.fetch_semantic_scholar_paper(
            "arXiv:1706.03762/extra",
            ["title"],
            session=session,
        )

        self.assertEqual(
            session.calls[0]["url"],
            f"{paper_clients.SEMANTIC_SCHOLAR_BASE_URL}/paper/arXiv%3A1706.03762%2Fextra",
        )

    async def test_request_json_raises_with_response_body_on_http_error(self):
        session = FakeSession(FakeResponse(status=429, text_data='{"error":"rate limit"}'))

        with patch("paper_clients.DEFAULT_MAX_RETRIES", 0):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 429: {"error":"rate limit"}'):
                await paper_clients._request_json(
                    "GET",
                    "https://example.com",
                    session=session,
                )

    @patch("paper_clients.asyncio.sleep")
    async def test_request_json_retries_after_rate_limit(self, sleep_mock):
        session = FakeSession(
            [
                FakeResponse(status=429, text_data='{"error":"slow down"}', headers={"Retry-After": "1"}),
                FakeResponse(status=200, text_data=json.dumps({"data": [{"title": "Recovered"}]})),
            ]
        )

        papers = await paper_clients.search_semantic_scholar_papers(
            "attention",
            ["title"],
            1,
            session=session,
        )

        self.assertEqual(papers, [{"title": "Recovered"}])
        sleep_mock.assert_awaited_once()

    async def test_download_file_returns_bytes(self):
        session = FakeSession(FakeResponse(status=200, bytes_data=b"%PDF-1.7"))

        content = await paper_clients.download_file("https://example.com/paper.pdf", session=session)

        self.assertEqual(content, b"%PDF-1.7")

    async def test_download_text_returns_text(self):
        session = FakeSession(FakeResponse(status=200, text_data="<html><body>ok</body></html>"))

        content = await paper_clients.download_text("https://example.com/paper.html", session=session)

        self.assertEqual(content, "<html><body>ok</body></html>")

    @patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "secret-key"}, clear=False)
    def test_build_headers_includes_semantic_scholar_api_key(self):
        headers = paper_clients._build_headers()
        self.assertEqual(headers["x-api-key"], "secret-key")


if __name__ == "__main__":
    unittest.main()
