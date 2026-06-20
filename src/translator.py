"""
Power Trade Natural Language Translator.
Translates unstructured chat, email, or free-form queries into structured trade models using Google AI Gemini.
Also classifies query intent (trade_booking, market_research, off_topic) to support guardrails and routing.
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger("trade_translator")


class DirectionEnum(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class QueryTypeEnum(str, Enum):
    """
    Classifies the intent of the incoming natural language query.
    - conversational: Greeting, small talk, or general chat with no trade/market intent.
    - trade_booking: User wants to place, confirm, or clarify a power trade (buy/sell electricity).
    - market_research: User is asking for market data, prices, quotes for any asset.
    - off_topic: Not related to energy trading, financial markets, or general conversation.
    """
    CONVERSATIONAL = "conversational"
    TRADE_BOOKING = "trade_booking"
    MARKET_RESEARCH = "market_research"
    OFF_TOPIC = "off_topic"


class ParsedTrade(BaseModel):
    """
    Structured result of NL-to-trade parsing, enriched with query classification
    for routing and guardrail decisions.
    """
    # ── Query classification ─────────────────────────────────────────────────
    query_type: QueryTypeEnum = Field(
        default=QueryTypeEnum.TRADE_BOOKING,
        description=(
            "Classify the query intent: "
            "'trade_booking' if the user wants to place/confirm a power trade; "
            "'market_research' if the user is asking for current prices, quotes, or market data on any asset; "
            "'off_topic' if the query is completely unrelated to energy trading or financial markets."
        )
    )
    research_query: Optional[str] = Field(
        None,
        description=(
            "Only set when query_type is 'market_research'. "
            "Extract a clean, concise search query suitable for a market data lookup "
            "(e.g. 'AAPL stock price', 'PJM electricity spot price', 'crude oil WTI price')."
        )
    )
    off_topic_response: Optional[str] = Field(
        None,
        description=(
            "Only set when query_type is 'off_topic'. "
            "A short, polite, professional response explaining that this assistant "
            "focuses on power trading and market research, and cannot help with the given topic."
        )
    )

    # ── Trade fields (relevant only for query_type = trade_booking) ──────────
    direction: Optional[DirectionEnum] = Field(
        None,
        description="BUY or SELL if mentioned or inferred, else null."
    )
    quantity_mw: Optional[float] = Field(
        None,
        description="The volume capacity in megawatts (MW), must be > 0. E.g. 100. Set null if missing or ambiguous."
    )
    price_per_mwh: Optional[float] = Field(
        None,
        description="The price per megawatt-hour (MWh) in currency, must be > 0. Set null if missing or if user says 'market price'."
    )
    counterparty: Optional[str] = Field(
        None,
        description="The counterparty name (e.g. Shell, BP, Conoco, Vitol)."
    )
    delivery_start: Optional[str] = Field(
        None,
        description="The delivery start date-time in ISO-8601 UTC format (YYYY-MM-DDTHH:MM:SSZ). Resolve relative expressions relative to the Reference Time."
    )
    delivery_end: Optional[str] = Field(
        None,
        description="The delivery end date-time in ISO-8601 UTC format (YYYY-MM-DDTHH:MM:SSZ). Must be chronologically after delivery_start."
    )
    hub: Optional[str] = Field(
        None,
        description="The power grid hub (e.g. MISO, PJM, ERCOT, SPP)."
    )
    notes: Optional[str] = Field(
        None,
        description="Optional additional details, assumptions made, or parsing anomalies."
    )
    confidence_score: float = Field(
        0.0,
        description=(
            "Confidence score from 0.0 to 1.0. For trade_booking, lower score (<0.70) "
            "if any critical field is missing. For market_research or off_topic, set to 1.0."
        )
    )
    ambiguous_or_missing_fields: List[str] = Field(
        default_factory=list,
        description="List of fields that are missing, ambiguous, or need clarification."
    )
    reasoning: str = Field(
        "",
        description="Brief step-by-step reasoning explaining query classification, date calculations, and confidence assessment."
    )


class PowerTradeTranslator:
    """
    Translator component that calls Google AI Studio Gemini with Structured Outputs.
    Classifies incoming queries and parses trade details when applicable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.model_name = model_name
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is required. "
                "Get a key from https://aistudio.google.com/"
            )
        self.client = genai.Client(api_key=resolved_key)

    def translate(self, text: str, reference_time: Optional[datetime] = None) -> ParsedTrade:
        """
        Translates unstructured trade query text into a structured `ParsedTrade`.
        Also classifies query intent for guardrails and routing.

        Args:
            text: Raw natural language input from the user.
            reference_time: Current server time used for resolving relative date expressions.

        Returns:
            ParsedTrade with query_type and trade fields populated appropriately.
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        ref_time_str = reference_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        ref_day_name = reference_time.strftime("%A")

        system_instruction = (
            "You are an expert power trading operations assistant. "
            "Your FIRST task is to classify the incoming query, then parse trade details if applicable.\n\n"

            # ── STEP 1: Query Classification ───────────────────────────────
            "STEP 1 — CLASSIFY the query intent:\n"
            "  • 'conversational'  → Greeting, small talk, or general chat (e.g. 'hello', 'thanks', 'what can you do').\n"
            "  • 'trade_booking'   → User wants to BUY or SELL electricity/power (e.g. 'Buy 100 MW at $47 from Shell in PJM').\n"
            "  • 'market_research' → User asks for market data, prices, or quotes for any financial asset "
            "(stocks, ETFs, energy, oil, gas, indices, mutual funds, etc.).\n"
            "  • 'off_topic'       → The query has NOTHING to do with trading, markets, or general chat "
            "(e.g. questions about weather, sports, recipes, personal advice unrelated to finance).\n\n"

            "For 'market_research': extract a concise search query in field `research_query` "
            "(e.g. 'AAPL stock price', 'crude oil WTI spot price', 'PJM electricity day-ahead price').\n"
            "For 'off_topic': write a polite, short refusal in `off_topic_response` explaining "
            "you only assist with power trading and financial market research. Set confidence_score to 1.0.\n\n"

            # ── STEP 2: Trade Parsing (only if query_type = trade_booking) ─
            "STEP 2 — Only if query_type is 'trade_booking', parse the following fields:\n\n"
            f"REFERENCE TIME context:\n"
            f"  - Current Date/Time: {ref_time_str}\n"
            f"  - Current Day of Week: {ref_day_name}\n\n"

            "DATE RESOLUTION GUIDELINES:\n"
            "1. 'tomorrow' means the day starting 00:00:00 UTC on the next calendar day, running 24 hours.\n"
            "2. 'next week' starts on the upcoming Monday at 00:00:00 UTC, ends the following Monday at 00:00:00 UTC.\n"
            "3. If only a date is given (e.g. 'June 20'), delivery_start is that date at 00:00:00 UTC, "
            "and delivery_end is the next day at 00:00:00 UTC.\n"
            "4. Convert all timezone terms to UTC. If no timezone is specified, assume UTC.\n\n"

            "VALIDATION & CONFIDENCE RULES (for trade_booking):\n"
            "- Direction must be BUY or SELL. ('purchased' or 'getting' = BUY; 'sold' or 'supplying' = SELL).\n"
            "- If key fields (direction, quantity_mw, price_per_mwh, counterparty, delivery_start, "
            "delivery_end, hub) are missing or ambiguous, set them to null, list in "
            "`ambiguous_or_missing_fields`, and set confidence_score below 0.70.\n"
            "- If all fields are clearly parsed, set confidence_score between 0.90 and 1.00.\n"
            "- For market_research or off_topic queries, trade fields should all be null and "
            "confidence_score should be 1.0."
        )

        prompt = f"Please classify and translate the following natural language request:\n\n{text}"

        logger.info(f"Sending translation request for text: '{text[:100]}...'")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.0,
                )
            )

            raw = response.text or "{}"
            parsed: ParsedTrade = ParsedTrade.model_validate(json.loads(raw))
            logger.info(
                f"Translated text. query_type={parsed.query_type}, "
                f"confidence={parsed.confidence_score:.2f}"
            )
            return parsed

        except Exception as e:
            logger.error(f"Failed to generate translation content: {e}", exc_info=True)
            # Safe fallback: return minimal error state
            return ParsedTrade(
                query_type=QueryTypeEnum.TRADE_BOOKING,
                direction=None,
                quantity_mw=None,
                price_per_mwh=None,
                counterparty=None,
                delivery_start=None,
                delivery_end=None,
                hub=None,
                notes=f"Parsing error: {str(e)}",
                confidence_score=0.0,
                ambiguous_or_missing_fields=["all"],
                reasoning=f"LLM API call failed: {e}"
            )
