"""Tests for free-tier LLMBackend resolution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scheduler.llm_backend import get_backend


class TestLLMBackend(unittest.TestCase):
    def test_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("paid-only")

    def test_missing_keys_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                get_backend()
            self.assertIn("free-tier", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
