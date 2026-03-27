import json
import unittest
from unittest.mock import AsyncMock, patch

import server


class ServerToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_arxiv_wraps_client_response(self):
        with patch("server.search_arxiv_papers", new=AsyncMock(return_value=[{"id": "1234.5678", "title": "Paper"}])):
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
        with patch("server.fetch_semantic_scholar_paper", new=AsyncMock(return_value=data)):
            response = await server.analyze_citations({"paper_id": "arXiv:1706.03762"})

        payload = json.loads(response[0].text)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["citation_graph"]["citation_count"], 100)
        self.assertEqual(payload["citation_graph"]["influential_citations"], 25)
        self.assertEqual(payload["citation_graph"]["citations"], 2)
        self.assertEqual(payload["citation_graph"]["references"], 1)


if __name__ == "__main__":
    unittest.main()
