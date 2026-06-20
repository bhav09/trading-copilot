"""
Unit and integration tests for the candidate-owned components (Client and Translator).
"""

from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytest
import httpx
from src.client import TradeRepositoryClient
from src.translator import PowerTradeTranslator, ParsedTrade, DirectionEnum


class TestTradeRepositoryClient(unittest.TestCase):
    """Tests for the robust repository HTTP client, focusing on retry logic."""

    @patch("httpx.Client.request")
    def test_client_retry_on_503_then_success(self, mock_request) -> None:
        # Mock responses: first 503 (transient error), then 201 (success)
        mock_response_503 = MagicMock(spec=httpx.Response)
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0"}  # Sleep 0 for fast test

        mock_response_201 = MagicMock(spec=httpx.Response)
        mock_response_201.status_code = 201
        mock_response_201.json.return_value = {"trade_id": 1, "status": "DRAFT"}

        mock_request.side_effect = [mock_response_503, mock_response_201]

        client = TradeRepositoryClient(base_url="http://127.0.0.1:8000")
        
        # Test create_trade
        payload = {"direction": "BUY", "quantity_mw": 100}
        result = client.create_trade(payload)

        self.assertEqual(result["trade_id"], 1)
        self.assertEqual(mock_request.call_count, 2)

    @patch("httpx.Client.request")
    def test_client_retry_limit_reached(self, mock_request) -> None:
        # Mock responses: always 503
        mock_response_503 = MagicMock(spec=httpx.Response)
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0"}
        mock_response_503.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Service Unavailable",
            request=MagicMock(),
            response=mock_response_503
        )

        mock_request.return_value = mock_response_503

        client = TradeRepositoryClient(base_url="http://127.0.0.1:8000")

        # Expect an exception when retries are exhausted
        with self.assertRaises(httpx.HTTPStatusError):
            client.create_trade({"direction": "BUY", "quantity_mw": 100})

        # By default max_retries = 3 in Client
        self.assertEqual(mock_request.call_count, 3)

    @patch("httpx.Client.request")
    def test_client_bypass_chaos_header(self, mock_request) -> None:
        # Mock successful response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_request.return_value = mock_response

        # Client with bypass_chaos enabled
        client = TradeRepositoryClient(base_url="http://127.0.0.1:8000", bypass_chaos=True)
        client.health()

        # Verify request was sent with bypass header
        args, kwargs = mock_request.call_args
        headers = kwargs.get("headers", {})
        self.assertEqual(headers.get("x-eval-bypass-chaos"), "true")

    @patch("httpx.Client.request")
    def test_health_returns_chaos_on_503(self, mock_request) -> None:
        """health() must return {'status': 'chaos'} on a 503/504 instead of raising."""
        mock_response_503 = MagicMock(spec=httpx.Response)
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0"}
        mock_request.return_value = mock_response_503

        client = TradeRepositoryClient(base_url="http://127.0.0.1:8000")
        result = client.health()

        # Must NOT raise; must return the chaos status dict
        self.assertEqual(result["status"], "chaos")
        self.assertEqual(result["http_code"], "503")

    @patch("httpx.Client.request")
    def test_health_timeout_3s_default(self, mock_request) -> None:
        """health() default timeout must be 3.0 s (to survive chaos 1.25 s DELAY fault)."""
        import inspect
        sig = inspect.signature(TradeRepositoryClient.health)
        default_timeout = sig.parameters["timeout"].default
        self.assertEqual(default_timeout, 3.0, "health() default timeout must be 3.0 s")


class TestPowerTradeTranslator(unittest.TestCase):
    """Tests for the translation engine parsing and validation."""

    @patch("google.genai.Client")
    def test_translation_happy_path(self, mock_genai_client_class) -> None:
        # Set up a mock response from the Gemini API
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_parsed = ParsedTrade(
            direction=DirectionEnum.BUY,
            quantity_mw=100.0,
            price_per_mwh=47.0,
            counterparty="Shell",
            delivery_start="2026-06-20T00:00:00Z",
            delivery_end="2026-06-21T00:00:00Z",
            hub="MISO",
            notes="Parsed cleanly",
            confidence_score=0.98,
            ambiguous_or_missing_fields=[],
            reasoning="Found BUY direction, 100 MW, $47 price, Shell cp, PJM hub. Tomorrow resolved using Ref Time."
        )
        mock_response.parsed = mock_parsed
        mock_client.models.generate_content.return_value = mock_response

        translator = PowerTradeTranslator()
        ref_time = datetime(2026, 6, 19, 18, 0, 0)
        
        text = "Buy 100 MW tomorrow at $47 from Shell"
        result = translator.translate(text, reference_time=ref_time)

        self.assertEqual(result.direction, DirectionEnum.BUY)
        self.assertEqual(result.quantity_mw, 100.0)
        self.assertEqual(result.price_per_mwh, 47.0)
        self.assertEqual(result.counterparty, "Shell")
        self.assertEqual(result.confidence_score, 0.98)
        self.assertEqual(len(result.ambiguous_or_missing_fields), 0)

    @patch("google.genai.Client")
    def test_translation_missing_fields(self, mock_genai_client_class) -> None:
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_parsed = ParsedTrade(
            direction=DirectionEnum.SELL,
            quantity_mw=80.0,
            price_per_mwh=51.0,
            counterparty="BP",
            delivery_start="2026-06-20T00:00:00Z",
            delivery_end="2026-06-21T00:00:00Z",
            hub=None,
            notes="Missing hub",
            confidence_score=0.65,
            ambiguous_or_missing_fields=["hub"],
            reasoning="Missing grid hub name."
        )
        mock_response.parsed = mock_parsed
        mock_client.models.generate_content.return_value = mock_response

        translator = PowerTradeTranslator()
        ref_time = datetime(2026, 6, 19, 18, 0, 0)
        
        text = "Please sell 80 MW tomorrow at $51 to BP"
        result = translator.translate(text, reference_time=ref_time)

        self.assertEqual(result.direction, DirectionEnum.SELL)
        self.assertNil = self.assertIsNone(result.hub)
        self.assertEqual(result.confidence_score, 0.65)
        self.assertIn("hub", result.ambiguous_or_missing_fields)
