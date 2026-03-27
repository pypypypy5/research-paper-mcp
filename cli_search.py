#!/usr/bin/env python3
"""
CLI wrapper for research-paper-mcp server
Allows direct command-line searches for use by AGI loop
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from env_loader import load_env_file

load_env_file()

import arxiv

from paper_clients import search_arxiv_papers


async def search_arxiv(query: str, max_results: int = 5):
    """Search arXiv for papers"""
    try:
        return await search_arxiv_papers(query, max_results, arxiv.SortCriterion.Relevance)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Search arXiv for research papers")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results")

    args = parser.parse_args()

    # Run async search
    results = asyncio.run(search_arxiv(args.query, args.max_results))

    # Output JSON to stdout
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
