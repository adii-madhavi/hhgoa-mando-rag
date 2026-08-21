"""
Config defaults must match what actually ships, not a disproved hypothesis
or stale documentation. Small, targeted regression tests only.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config                               # noqa: E402


class TestCrossLingualDefault:
    def test_defaults_to_false(self):
        """
        Cross-lingual retrieval routing was tested and rejected
        (EXPERIMENTS.md). The flag is unused at runtime; its default must not
        silently re-enable a rejected hypothesis for anyone reading the
        config as documentation. (Config's dataclass fields bake in
        os.environ at import time, same as every other setting here, so
        this is checked directly rather than via monkeypatch.)
        """
        assert Config().cross_lingual is False


class TestChunkStrategyDefault:
    def test_defaults_to_fixed(self):
        """
        `fixed` is the winning chunking strategy (EXPERIMENTS.md). README's
        index-build command must agree with this, not the earlier
        `hierarchical` pick that a same-language-only comparison favoured.
        """
        assert Config().chunk_strategy == "fixed"
