import json
import unittest
from unittest.mock import AsyncMock, patch

import paper_tools
import server


class ServerToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_arxiv_wraps_client_response(self):
        with patch("paper_tools.search_arxiv_papers", new=AsyncMock(return_value=[{"id": "1234.5678", "title": "Paper"}])):
            response = await server.search_arxiv({"query": "agentic", "max_results": 1})

        payload = json.loads(response[0].text)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["papers"][0]["id"], "1234.5678")

    async def test_analyze_citations_builds_graph_from_client_response(self):
        data = {
            "title": "Attention Is All You Need",
            "citationCount": 100,
            "influentialCitationCount": 25,
            "citations": [{"paperId": "c1"}, {"paperId": "c2"}],
            "references": [{"paperId": "r1"}],
        }
        with patch("paper_tools.fetch_semantic_scholar_paper", new=AsyncMock(return_value=data)):
            response = await server.analyze_citations({"paper_id": "arXiv:1706.03762"})

        payload = json.loads(response[0].text)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["citation_graph"]["citation_count"], 100)
        self.assertEqual(payload["citation_graph"]["influential_citations"], 25)
        self.assertEqual(payload["citation_graph"]["citations"], 2)
        self.assertEqual(payload["citation_graph"]["references"], 1)

    async def test_handle_call_tool_dispatches_through_tool_registry(self):
        response = [paper_tools.types.TextContent(type="text", text='{"success": true}')]

        with patch("server.call_tool", new=AsyncMock(return_value=response)) as call_tool_mock:
            actual = await server.handle_call_tool("search_arxiv", {"query": "agentic"})

        self.assertEqual(actual, response)
        call_tool_mock.assert_awaited_once_with("search_arxiv", {"query": "agentic"})


if __name__ == "__main__":
    unittest.main()
