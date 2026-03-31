import json
import unittest
from unittest.mock import AsyncMock, patch

import paper_tools


SAMPLE_ARXIV_HTML = """
<html>
  <body>
    <section>
      <h2>Abstract</h2>
      <p>We propose a compact transformer variant for long-context reasoning.</p>
    </section>
    <section>
      <h2>2 Methods</h2>
      <p>Our method combines grouped-query attention with sparse routing.</p>
      <h3>2.1 Training Details</h3>
      <p>We train with curriculum sampling and label smoothing.</p>
    </section>
    <section>
      <h2>3 Results</h2>
      <p>We outperform the baseline by 4.2 points.</p>
    </section>
  </body>
</html>
"""

ABS_PAGE_HTML = """
<html>
  <body>
    <blockquote class="abstract mathjax">
      <span class="descriptor">Abstract:</span>
      We show a direct abstract fallback from the arXiv abs page.
    </blockquote>
  </body>
</html>
"""


class PaperToolSectionParsingTests(unittest.TestCase):
    def test_build_arxiv_section_urls_prefers_official_html_before_fallbacks(self):
        urls = paper_tools._build_arxiv_section_urls("arXiv:1234.5678")

        self.assertEqual(
            urls,
            [
                "https://arxiv.org/html/1234.5678",
                "https://ar5iv.labs.arxiv.org/html/1234.5678",
                "https://arxiv.org/abs/1234.5678",
            ],
        )

    def test_extract_requested_sections_from_html_returns_only_requested_text(self):
        sections = paper_tools._extract_requested_sections_from_html(SAMPLE_ARXIV_HTML, ["abstract", "method"])

        self.assertEqual(set(sections.keys()), {"abstract", "method"})
        self.assertIn("compact transformer variant", sections["abstract"])
        self.assertIn("grouped-query attention", sections["method"])
        self.assertIn("curriculum sampling", sections["method"])
        self.assertNotIn("outperform the baseline", sections["method"])

    def test_extract_requested_sections_matches_alias_heading(self):
        html = """
        <html><body>
          <section>
            <h2>4 Methodology</h2>
            <p>We optimize the planner with iterative rollouts.</p>
          </section>
        </body></html>
        """

        sections = paper_tools._extract_requested_sections_from_html(html, ["method"])

        self.assertEqual(sections["method"], "We optimize the planner with iterative rollouts.")

    def test_extract_requested_sections_matches_our_method_and_empirical_experiments(self):
        html = """
        <html><body>
          <section>
            <h2>4 Our Method</h2>
            <p>We describe the simple design of LoRA and its practical benefits.</p>
          </section>
          <section>
            <h2>5 Empirical Experiments</h2>
            <p>We evaluate the downstream task performance of LoRA.</p>
          </section>
        </body></html>
        """

        sections = paper_tools._extract_requested_sections_from_html(html, ["method", "experiments"])

        self.assertEqual(sections["method"], "We describe the simple design of LoRA and its practical benefits.")
        self.assertEqual(sections["experiments"], "We evaluate the downstream task performance of LoRA.")

    def test_extract_requested_sections_matches_related_works_plural_and_singular(self):
        html = """
        <html><body>
          <section>
            <h2>6 Related Work</h2>
            <p>We compare against prior prompting and planning frameworks.</p>
          </section>
          <section>
            <h2>7 Conclusion</h2>
            <p>Closing remarks.</p>
          </section>
        </body></html>
        """

        sections = paper_tools._extract_requested_sections_from_html(html, ["related work", "related works"])

        expected = "We compare against prior prompting and planning frameworks."
        self.assertEqual(sections["related work"], expected)
        self.assertEqual(sections["related works"], expected)

    def test_extract_full_text_from_html_keeps_headings_and_body_text(self):
        html = """
        <html><body>
          <h1>Sample Paper</h1>
          <section>
            <h2>Abstract</h2>
            <p>We propose a compact transformer variant for long-context reasoning.</p>
          </section>
          <section>
            <h2>2 Our Method</h2>
            <p>Our method combines grouped-query attention with sparse routing.</p>
          </section>
        </body></html>
        """

        full_text = paper_tools._extract_full_text_from_html(html)

        self.assertIn("Sample Paper", full_text)
        self.assertIn("Abstract", full_text)
        self.assertIn("We propose a compact transformer variant", full_text)
        self.assertIn("2 Our Method", full_text)
        self.assertIn("Our method combines grouped-query attention", full_text)


class PaperToolAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_arxiv_sections_returns_only_requested_sections(self):
        with patch("paper_tools.download_text", new=AsyncMock(return_value=SAMPLE_ARXIV_HTML)):
            response = await paper_tools.get_arxiv_sections(
                {"paper_id": "arXiv:1234.5678", "sections": ["abstract", "method"]}
            )

        payload = json.loads(response[0].text)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["paper_id"], "1234.5678")
        self.assertEqual(set(payload["sections"].keys()), {"abstract", "method"})
        self.assertEqual(payload["missing_sections"], [])
        self.assertIn("compact transformer variant", payload["sections"]["abstract"])
        self.assertNotIn("outperform the baseline", payload["sections"]["method"])

    async def test_get_arxiv_sections_falls_back_to_abs_page_for_abstract(self):
        side_effect = [
            RuntimeError("ar5iv unavailable"),
            RuntimeError("arxiv html unavailable"),
            ABS_PAGE_HTML,
        ]

        with patch("paper_tools.download_text", new=AsyncMock(side_effect=side_effect)):
            response = await paper_tools.get_arxiv_sections({"paper_id": "1234.5678", "sections": ["abstract"]})

        payload = json.loads(response[0].text)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["sections"], {"abstract": "We show a direct abstract fallback from the arXiv abs page."})
        self.assertEqual(payload["missing_sections"], [])
        self.assertEqual(payload["source_urls_tried"], ["https://arxiv.org/abs/1234.5678"])

    async def test_get_arxiv_sections_supports_related_works_request_for_related_work_heading(self):
        html = """
        <html><body>
          <section>
            <h2>5 Related Work</h2>
            <p>We compare against prior prompting and planning frameworks.</p>
          </section>
          <section>
            <h2>6 Conclusion</h2>
            <p>Closing remarks.</p>
          </section>
        </body></html>
        """

        with patch("paper_tools.download_text", new=AsyncMock(return_value=html)):
            response = await paper_tools.get_arxiv_sections({"paper_id": "1234.5678", "sections": ["related works"]})

        payload = json.loads(response[0].text)
        self.assertEqual(
            payload["sections"]["related works"],
            "We compare against prior prompting and planning frameworks.",
        )
        self.assertEqual(payload["missing_sections"], [])

    async def test_get_arxiv_full_text_returns_parsed_text_and_source_url(self):
        html = """
        <html><body>
          <h1>Sample Paper</h1>
          <section>
            <h2>Abstract</h2>
            <p>We propose a compact transformer variant for long-context reasoning.</p>
          </section>
          <section>
            <h2>2 Our Method</h2>
            <p>Our method combines grouped-query attention with sparse routing.</p>
          </section>
        </body></html>
        """

        with patch("paper_tools.download_text", new=AsyncMock(return_value=html)):
            response = await paper_tools.get_arxiv_full_text({"paper_id": "arXiv:1234.5678"})

        payload = json.loads(response[0].text)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["paper_id"], "1234.5678")
        self.assertEqual(payload["source_url"], "https://arxiv.org/html/1234.5678")
        self.assertIn("Sample Paper", payload["full_text"])
        self.assertIn("Abstract", payload["full_text"])
        self.assertIn("Our method combines grouped-query attention", payload["full_text"])


if __name__ == "__main__":
    unittest.main()
