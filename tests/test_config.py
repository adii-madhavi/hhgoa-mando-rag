"""
Config defaults must match what actually ships, not a disproved hypothesis
or stale documentation. Small, targeted regression tests only.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config                               # noqa: E402
from app.generation.llm_client import ChatClient            # noqa: E402


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


class TestProviderCredentialIsolation:
    def test_primary_never_reuses_sarvam_key(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("SARVAM_API_KEY", "stt-only-placeholder")
        cfg = Config()
        client = ChatClient(api_key=None)
        try:
            assert cfg.llm_api_key is None
            assert cfg.has_llm is False
            assert client.available is False
        finally:
            client.close()

    def test_primary_and_external_credentials_remain_independent(self,
                                                                 monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "primary-placeholder")
        monkeypatch.setenv("EXTERNAL_LLM_API_KEY", "external-placeholder")
        monkeypatch.setenv("EXTERNAL_LLM_BASE_URL",
                           "https://external.invalid/v1")
        monkeypatch.setenv("EXTERNAL_LLM_MODEL", "external-model")
        cfg = Config()
        assert cfg.llm_api_key == "primary-placeholder"
        assert cfg.external_llm_api_key == "external-placeholder"
        assert cfg.external_llm_base_url == "https://external.invalid/v1"
        assert cfg.external_llm_model == "external-model"
        assert cfg.has_llm is True
        assert cfg.has_external_llm is True
