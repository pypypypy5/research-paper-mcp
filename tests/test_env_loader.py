import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from env_loader import load_env_file


class EnvLoaderTests(unittest.TestCase):
    def test_load_env_file_sets_missing_values_and_preserves_existing_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "SEMANTIC_SCHOLAR_API_KEY=from-file",
                        "ARXIV_MIN_INTERVAL_SECONDS=\"7\"",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "from-env"}, clear=False):
                load_env_file(env_path)
                self.assertEqual(os.environ["SEMANTIC_SCHOLAR_API_KEY"], "from-env")
                self.assertEqual(os.environ["ARXIV_MIN_INTERVAL_SECONDS"], "7")


if __name__ == "__main__":
    unittest.main()
