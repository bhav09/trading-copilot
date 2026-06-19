"""
Candidate-owned FastAPI server and Web Dashboard.
Exposes endpoints for natural language parsing, trade booking, and hosts the visual dashboard.
"""

import logging
import collections
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import httpx

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.client import TradeRepositoryClient
from src.translator import PowerTradeTranslator, ParsedTrade, QueryTypeEnum

# Configure Logging
logger = logging.getLogger("trade_copilot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

# In-memory log buffer for frontend terminal observability
class MemoryLogHandler(logging.Handler):
    def __init__(self, maxlen: int = 50):
        super().__init__()
        self.log_buffer = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        self.log_buffer.append({
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "msg": record.getMessage()
        })

mem_handler = MemoryLogHandler()
mem_handler.setFormatter(formatter)
logger.addHandler(mem_handler)

# Add handler to our sub-loggers to aggregate details
logging.getLogger("trade_client").addHandler(mem_handler)
logging.getLogger("trade_client").setLevel(logging.INFO)
logging.getLogger("trade_translator").addHandler(mem_handler)
logging.getLogger("trade_translator").setLevel(logging.INFO)

# Initialize Rate Limiter: 20 requests per minute per IP for parsing/booking
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Power Trade AI Copilot Dashboard",
    description="Natural language to structured power trade orchestration portal.",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize translation engine
translator = PowerTradeTranslator()

# Requests schemas
class ParseRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=1000)
    bypass_chaos: bool = Field(default=False)

class BookRequest(BaseModel):
    trade_data: Dict[str, Any] = Field(...)
    bypass_chaos: bool = Field(default=False)


@app.get("/api/logs", response_model=List[Dict[str, str]])
def get_system_logs() -> List[Dict[str, str]]:
    """Returns the latest in-memory logs for terminal rendering."""
    return list(mem_handler.log_buffer)


@app.get("/api/health")
def get_health(bypass_chaos: bool = False) -> Dict[str, Any]:
    """Check the health status of our app and connection to the backend repository."""
    client = TradeRepositoryClient(bypass_chaos=bypass_chaos)
    try:
        repo_health = client.health(timeout=1.5, max_retries=1)
        return {"copilot_status": "ok", "repository_status": repo_health.get("status", "unknown")}
    except Exception as e:
        logger.error(f"Failed health check for repository: {e}")
        return {"copilot_status": "ok", "repository_status": f"unreachable: {str(e)}"}


@app.post("/api/parse", response_model=Dict[str, Any])
@limiter.limit("20/minute")
def parse_natural_language(request: Request, payload: ParseRequest) -> Dict[str, Any]:
    """
    Parses unstructured text into a trade draft.
    Two-tier routing:
      1. Frontend fast-classifier handles clear conversational/greeting queries with zero latency.
      2. This endpoint handles trade_booking, market_research, off_topic, and any edge-case
         conversational queries that bypassed the frontend classifier.
    """
    logger.info(f"Parsing query: '{payload.text}'")
    ref_time = datetime.now()

    parsed_trade: ParsedTrade = translator.translate(payload.text, reference_time=ref_time)

    # ── Conversational fallback (edge cases that bypass the JS fast classifier) ─
    if parsed_trade.query_type == QueryTypeEnum.CONVERSATIONAL:
        logger.info("Query classified as conversational — returning lightweight reply.")
        return {
            "query_type": "off_topic",  # Reuse off_topic UI on frontend for clean display
            "message": (
                "Hello! I'm your Power Trade Copilot. I can help you book power trades, "
                "look up stock or commodity prices, or research energy market data. "
                "What would you like to do?"
            ),
            "reference_time": ref_time.isoformat()
        }

    # ── Off-topic guardrail: return gracefully without trade card ──────────
    if parsed_trade.query_type == QueryTypeEnum.OFF_TOPIC:
        logger.info("Query classified as off_topic — returning graceful guardrail message.")
        return {
            "query_type": "off_topic",
            "message": parsed_trade.off_topic_response or (
                "I'm your Power Trade Copilot — I specialise in energy trading and financial market research. "
                "I can't help with that topic, but I'm happy to assist you book or research power trades, "
                "check stock prices, or look up commodity data. What can I help you with?"
            ),
            "reference_time": ref_time.isoformat()
        }

    # ── Market research: return query for frontend to call /api/market-data ─
    if parsed_trade.query_type == QueryTypeEnum.MARKET_RESEARCH:
        logger.info(f"Query classified as market_research — topic: {parsed_trade.research_query}")
        return {
            "query_type": "market_research",
            "research_query": parsed_trade.research_query or payload.text,
            "message": f"Fetching market data for: {parsed_trade.research_query or payload.text}",
            "reference_time": ref_time.isoformat()
        }

    # ── Trade booking: run full field validation ────────────────────────────
    required_fields = ["direction", "quantity_mw", "price_per_mwh", "counterparty", "delivery_start", "delivery_end"]
    missing = list(parsed_trade.ambiguous_or_missing_fields)

    # Run local validation constraints matching repository rules
    invalid_reasons = []
    if parsed_trade.price_per_mwh is not None:
        if parsed_trade.price_per_mwh <= 0 or parsed_trade.price_per_mwh > 1000:
            invalid_reasons.append("Price must be greater than 0 and up to $1000/MWh")
            if "price_per_mwh" not in missing:
                missing.append("price_per_mwh")
    if parsed_trade.quantity_mw is not None:
        if parsed_trade.quantity_mw <= 0 or parsed_trade.quantity_mw > 10000:
            invalid_reasons.append("Quantity must be greater than 0 and up to 10000 MW")
            if "quantity_mw" not in missing:
                missing.append("quantity_mw")
    if parsed_trade.delivery_start and parsed_trade.delivery_end:
        try:
            start_dt = datetime.fromisoformat(parsed_trade.delivery_start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(parsed_trade.delivery_end.replace("Z", "+00:00"))
            if end_dt <= start_dt:
                invalid_reasons.append("Delivery end must be after delivery start")
                if "delivery_end" not in missing:
                    missing.append("delivery_end")
        except Exception:
            invalid_reasons.append("Invalid delivery date format")
            if "delivery_start" not in missing:
                missing.append("delivery_start")
            if "delivery_end" not in missing:
                missing.append("delivery_end")

    for field in required_fields:
        val = getattr(parsed_trade, field, None)
        if val is None and field not in missing:
            missing.append(field)

    if parsed_trade.hub is None and "hub" not in missing:
        missing.append("hub")

    # Determine booking readiness
    ready_for_booking = parsed_trade.confidence_score >= 0.70 and len(missing) == 0

    reasoning_text = parsed_trade.reasoning
    if invalid_reasons:
        reasoning_text = "[Validation Check: " + "; ".join(invalid_reasons) + "] " + reasoning_text

    return {
        "query_type": "trade_booking",
        "parsed_trade": parsed_trade.model_dump(),
        "confidence_score": parsed_trade.confidence_score,
        "missing_fields": missing,
        "ready_for_booking": ready_for_booking,
        "reasoning": reasoning_text,
        "reference_time": ref_time.isoformat()
    }


@app.post("/api/confirm-book", status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
def confirm_and_book_trade(request: Request, payload: BookRequest) -> Dict[str, Any]:
    """
    Takes the confirmed parsed payload, validates it, and posts it to the repository.
    Includes simple retries via our repository client.
    """
    trade_data = payload.trade_data
    logger.info(f"Attempting to book trade with counterparty {trade_data.get('counterparty')}")
    
    # Validation check: Pydantic parsing error handling
    # Let's perform simple sanity checks before calling backend
    try:
        # Check delivery end is after start
        start_str = trade_data.get("delivery_start")
        end_str = trade_data.get("delivery_end")
        if start_str and end_str:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            if end_dt <= start_dt:
                raise ValueError("delivery_end must be after delivery_start")
                
        qty = float(trade_data.get("quantity_mw", 0))
        price = float(trade_data.get("price_per_mwh", 0))
        if qty <= 0 or qty > 10000:
            raise ValueError("quantity_mw must be greater than 0 and up to 10000")
        if price <= 0 or price > 1000:
            raise ValueError("price_per_mwh must be greater than 0 and up to 1000")
            
        if not trade_data.get("direction"):
            raise ValueError("direction is required")
        if not trade_data.get("counterparty"):
            raise ValueError("counterparty is required")
        if not trade_data.get("hub"):
            raise ValueError("hub is required")
            
    except Exception as ve:
        logger.error(f"Pre-validation check failed: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation Error: {str(ve)}"
        )

    # Initialize repository client with chaos bypass configuration
    client = TradeRepositoryClient(bypass_chaos=payload.bypass_chaos)
    
    try:
        created_trade = client.create_trade(trade_data)
        logger.info(f"Successfully booked trade ID {created_trade.get('trade_id')}")
        return created_trade
        
    except httpx.HTTPStatusError as he:
        status_code = he.response.status_code
        detail = "Server error booking trade"
        try:
            detail = he.response.json().get("detail", detail)
        except Exception:
            detail = he.response.text or detail
            
        logger.error(f"Repository API returned error {status_code}: {detail}")
        raise HTTPException(status_code=status_code, detail=detail)
        
    except Exception as exc:
        logger.error(f"Uncaught exception booking trade: {exc}")
        raise HTTPException(status_code=500, detail=f"Booking failed: {str(exc)}")


@app.get("/api/trades", response_model=List[Dict[str, Any]])
def list_booked_trades(bypass_chaos: bool = False) -> List[Dict[str, Any]]:
    """List all persisted trades in the backend repository."""
    client = TradeRepositoryClient(bypass_chaos=bypass_chaos)
    try:
        return client.list_trades()
    except Exception as e:
        logger.error(f"Failed to list trades from repository: {e}")
        return []


@app.get("/api/market-data", response_model=Dict[str, Any])
async def get_market_data(query: str) -> Dict[str, Any]:
    """
    Fetches real-time or recent market data for a given asset query.
    - For stock/ETF tickers (e.g. 'AAPL', 'SPY'): uses Yahoo Finance unofficial API.
    - For other queries: uses DuckDuckGo Instant Answer API as a fallback.
    Returns structured price/description data for the frontend to display.
    """
    logger.info(f"Market data request for query: '{query}'")

    # ── Try Yahoo Finance for stock tickers ──────────────────────────────────
    # Heuristic: if the query looks like a ticker (1-5 uppercase letters) try YF
    import re
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
    yf_data = None

    if ticker_match:
        ticker = ticker_match.group(1)
        try:
            yf_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(yf_url, headers=headers)
                if resp.status_code == 200:
                    yf_json = resp.json()
                    result = yf_json.get("chart", {}).get("result", [])
                    if result:
                        meta = result[0].get("meta", {})
                        price = meta.get("regularMarketPrice")
                        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
                        currency = meta.get("currency", "USD")
                        name = meta.get("shortName") or meta.get("longName") or ticker
                        exchange = meta.get("exchangeName", "")
                        change = round(price - prev_close, 4) if price and prev_close else None
                        change_pct = round((change / prev_close) * 100, 2) if change and prev_close else None

                        yf_data = {
                            "source": "Yahoo Finance",
                            "type": "stock",
                            "ticker": ticker,
                            "name": name,
                            "price": price,
                            "currency": currency,
                            "exchange": exchange,
                            "change": change,
                            "change_pct": change_pct,
                            "prev_close": prev_close,
                            "summary": (
                                f"{name} ({ticker}) is trading at {currency} {price:,.2f} "
                                f"({'up' if change and change > 0 else 'down'} "
                                f"{abs(change_pct):.2f}% from prev close {currency} {prev_close:,.2f})"
                            ) if price and prev_close else f"{name} ({ticker}) price: {currency} {price}"
                        }
                        logger.info(f"Yahoo Finance data fetched for {ticker}: {price} {currency}")
        except Exception as yf_err:
            logger.warning(f"Yahoo Finance lookup failed for '{ticker}': {yf_err}")

    if yf_data:
        return yf_data

    # ── DuckDuckGo Instant Answer fallback ───────────────────────────────────
    # Used for general market queries (commodity prices, indices, descriptions)
    try:
        ddg_url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        headers = {"User-Agent": "Mozilla/5.0"}
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(ddg_url, params=params, headers=headers)
            if resp.status_code == 200:
                ddg_json = resp.json()
                abstract = ddg_json.get("AbstractText", "")
                abstract_source = ddg_json.get("AbstractSource", "")
                answer = ddg_json.get("Answer", "")
                answer_type = ddg_json.get("AnswerType", "")
                image = ddg_json.get("Image", "")
                related = [
                    r.get("Text", "") for r in ddg_json.get("RelatedTopics", [])[:3]
                    if r.get("Text")
                ]

                result_text = answer or abstract or (related[0] if related else None)

                if result_text:
                    logger.info(f"DuckDuckGo instant answer found for query: '{query}'")
                    return {
                        "source": abstract_source or "DuckDuckGo",
                        "type": answer_type or "general",
                        "query": query,
                        "summary": result_text,
                        "related": related[:2],
                        "image": image
                    }
    except Exception as ddg_err:
        logger.warning(f"DuckDuckGo lookup failed for '{query}': {ddg_err}")

    # ── No data found ────────────────────────────────────────────────────────
    logger.warning(f"No market data found for query: '{query}'")
    return {
        "source": "No data",
        "type": "not_found",
        "query": query,
        "summary": (
            f"I wasn't able to retrieve live market data for '{query}' right now. "
            "This may be due to API rate limits or the query not matching a known ticker. "
            "Try a specific stock ticker (e.g. 'AAPL', 'SPY') or a different search term."
        )
    }

@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> str:
    """Renders the single page dashboard with Outfits Google Font, dark theme, and glassmorphism."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Conversational AI agent for power trade booking.">
    <title>Axso Power Trade Copilot</title>
    <!-- Google Fonts: Inter & JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-sidebar: #171717;
            --bg-chat: #212121;
            --bg-input: #2f2f2f;
            --border-color: #3e3e3e;
            --text-main: #ececec;
            --text-muted: #b4b4b4;
            --primary: #543fd7;
            --primary-hover: #6854e4;
            --accent-emerald: #10b981;
            --accent-rose: #ef4444;
            --accent-amber: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-chat);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Sidebar styling */
        .sidebar {
            width: 280px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            height: 100%;
        }

        .sidebar-header {
            padding: 1.5rem 1rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logo-dot {
            width: 10px;
            height: 10px;
            background-color: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--primary);
            animation: pulse 2s infinite;
        }

        .logo-text {
            font-size: 1.1rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .connection-badge {
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.25rem;
        }

        .badge-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: var(--text-muted);
        }

        .new-chat-btn {
            margin: 1rem;
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.6rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: background 0.2s;
        }

        .new-chat-btn:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }

        .history-section {
            flex: 1;
            overflow-y: auto;
            padding: 0 0.5rem;
        }

        .history-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            padding: 0.75rem 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .history-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .history-item {
            padding: 0.6rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            color: var(--text-muted);
            transition: background 0.2s, color 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
        }

        .history-item:hover {
            background-color: rgba(255, 255, 255, 0.03);
            color: var(--text-main);
        }

        .history-item.active-session {
            background-color: rgba(84, 63, 215, 0.12);
            border-left: 2px solid var(--primary);
            color: var(--text-main);
            font-weight: 500;
        }

        .sidebar-footer {
            padding: 1rem;
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .settings-toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            cursor: pointer;
        }

        .log-toggle-btn {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.4rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            text-align: center;
            width: 100%;
        }

        .log-toggle-btn:hover {
            background-color: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
        }

        /* Collapsible Logs Panel */
        .logs-panel {
            height: 180px;
            background: #0d0d0d;
            border-top: 1px solid var(--border-color);
            padding: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            overflow-y: auto;
            display: none; /* Toggled via JS */
        }

        .log-line {
            margin-bottom: 0.25rem;
            line-height: 1.3;
        }

        .log-time { color: #888; }
        .log-level-info { color: #3b82f6; }
        .log-level-warn { color: var(--accent-amber); }
        .log-level-err { color: var(--accent-rose); }

        /* Main Workspace & Tabs Styling */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
            position: relative;
        }

        .tabs-header {
            display: flex;
            background-color: var(--bg-sidebar);
            border-bottom: 1px solid var(--border-color);
            padding: 0.5rem 1.5rem;
            gap: 1rem;
            flex-shrink: 0;
            z-index: 10;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 500;
            padding: 0.5rem 0.75rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: color 0.2s, border-color 0.2s;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            outline: none;
        }

        .tab-btn:hover {
            color: var(--text-main);
        }

        .tab-btn.active {
            color: var(--text-main);
            border-bottom-color: var(--primary);
        }

        .ledger-count {
            font-size: 0.75rem;
            background-color: rgba(255, 255, 255, 0.08);
            padding: 0.1rem 0.45rem;
            border-radius: 10px;
            font-weight: 600;
            color: var(--text-muted);
        }

        .tab-btn.active .ledger-count {
            background-color: var(--primary);
            color: #fff;
            box-shadow: 0 0 6px rgba(84, 63, 215, 0.4);
        }

        /* View wrappers */
        .chat-view-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: calc(100% - 46px);
            position: relative;
        }

        .chat-feed {
            flex: 1;
            overflow-y: auto;
            padding: 2rem 1rem 6rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        /* Center column wrapper for ChatGPT look */
        .chat-width-wrapper {
            max-width: 720px;
            width: 100%;
            margin: 0 auto;
        }

        /* Message bubbles */
        .msg-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .msg-row.user {
            justify-content: flex-end;
        }

        .avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: bold;
            flex-shrink: 0;
        }

        .avatar.ai {
            background-color: var(--primary);
            color: #fff;
        }

        .msg-content {
            flex: 1;
            max-width: 85%;
            font-size: 0.95rem;
            line-height: 1.5;
            color: var(--text-main);
        }

        .msg-row.user .msg-content {
            background-color: var(--bg-input);
            padding: 0.75rem 1.25rem;
            border-radius: 18px;
            max-width: 70%;
            color: var(--text-main);
        }

        /* Welcome screen */
        .welcome-panel {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60%;
            text-align: center;
            color: var(--text-muted);
            padding: 2rem;
            margin: auto;
        }

        .welcome-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        .welcome-sub {
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        .chips-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            max-width: 600px;
            width: 100%;
        }

        .suggestion-chip {
            background-color: transparent;
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            font-size: 0.85rem;
            color: var(--text-main);
            cursor: pointer;
            text-align: left;
            transition: background 0.2s, border-color 0.2s;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .suggestion-chip:hover {
            background-color: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .suggestion-chip .chip-header {
            font-weight: 600;
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* Chat Input Block */
        .input-wrapper {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem 1rem 2rem 1rem;
            background: linear-gradient(180deg, transparent 0%, var(--bg-chat) 30%);
        }

        .input-box {
            background-color: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 0.5rem 0.5rem 0.5rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

        .input-box:focus-within {
            border-color: rgba(255, 255, 255, 0.2);
        }

        .chat-textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.95rem;
            resize: none;
            height: 28px;
            line-height: 24px;
        }

        .send-btn {
            width: 32px;
            height: 32px;
            background-color: var(--text-main);
            color: var(--bg-chat);
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: opacity 0.2s, background-color 0.2s;
        }

        .send-btn:hover {
            background-color: #ffffff;
        }

        .send-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        /* Structured Card in Chat */
        .trade-card {
            background-color: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-top: 0.75rem;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }

        .trade-card-header {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            background-color: rgba(255, 255, 255, 0.02);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .conf-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }

        .conf-high {
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--accent-emerald);
        }

        .conf-low {
            background-color: rgba(245, 158, 11, 0.1);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        .trade-form-table {
            width: 100%;
            border-collapse: collapse;
        }

        .trade-form-table td {
            padding: 0.6rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.85rem;
            vertical-align: top;
        }

        .trade-form-table tr:last-child td {
            border-bottom: none;
        }

        .field-label {
            color: var(--text-muted);
            width: 35%;
            padding-top: 0.4rem;
        }

        .form-input {
            width: 100%;
            background-color: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-main);
            padding: 0.35rem 0.5rem;
            font-family: inherit;
            font-size: 0.85rem;
            outline: none;
            transition: border-color 0.2s, background-color 0.2s;
        }

        /* Dark-themed select dropdown to match app theme */
        select.form-input {
            /* Remove browser default appearance */
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            /* Custom chevron arrow (white SVG inline) */
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%23888' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 0.6rem center;
            background-color: rgba(0, 0, 0, 0.35);
            padding-right: 2rem;
            cursor: pointer;
            color-scheme: dark;
        }

        select.form-input option {
            background-color: #1a1a1a;
            color: var(--text-main);
        }

        select.form-input:focus {
            border-color: rgba(255, 255, 255, 0.2);
            outline: none;
        }

        .form-input:focus {
            border-color: rgba(255, 255, 255, 0.2);
        }

        .form-input.missing-warning {
            border-color: var(--accent-amber);
            background-color: rgba(245, 158, 11, 0.03);
        }

        .form-input.validation-error-border {
            border-color: var(--accent-rose) !important;
            background-color: rgba(239, 68, 68, 0.04) !important;
        }

        .field-error-text {
            color: #f87171;
            font-size: 0.75rem;
            margin-top: 0.25rem;
            display: none;
        }

        .card-alert {
            background-color: rgba(245, 158, 11, 0.08);
            border-left: 3px solid var(--accent-amber);
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
            color: #ffb020;
            transition: all 0.2s;
        }

        .card-alert.error {
            background-color: rgba(239, 68, 68, 0.08);
            border-left-color: var(--accent-rose);
            color: #f87171;
        }

        .card-reasoning {
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            background-color: rgba(0, 0, 0, 0.1);
        }

        .card-actions {
            padding: 0.75rem 1rem;
            display: flex;
            gap: 0.5rem;
            background-color: rgba(255, 255, 255, 0.01);
            border-top: 1px solid var(--border-color);
        }

        .card-btn {
            flex: 1;
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: background 0.2s, opacity 0.2s;
            text-align: center;
        }

        .card-btn-success {
            background-color: var(--accent-emerald);
            color: #fff;
        }

        .card-btn-success:hover:not(:disabled) {
            background-color: #10a374;
        }

        .card-btn-secondary {
            background-color: rgba(255, 255, 255, 0.08);
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }

        .card-btn-secondary:hover {
            background-color: rgba(255, 255, 255, 0.12);
        }

        .card-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .status-badge-inline {
            width: 100%;
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            border-top: 1px solid var(--border-color);
        }

        .status-badge-inline.success {
            background-color: rgba(16, 185, 129, 0.08);
            color: var(--accent-emerald);
        }

        /* Typing indicator */
        .typing-indicator {
            display: flex;
            gap: 4px;
            align-items: center;
            height: 20px;
        }

        .typing-indicator span {
            width: 6px;
            height: 6px;
            background-color: var(--text-muted);
            border-radius: 50%;
            display: inline-block;
            animation: bounce 1.3s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: var(--text-main);
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }

        /* Trades Registry View CSS */
        .ledger-view-wrapper {
            flex: 1;
            min-height: 0; /* Critical: allows flex child to shrink below its content height for scrollability */
            display: flex;
            flex-direction: column;
            height: calc(100% - 46px);
            padding: 2rem;
            overflow: hidden; /* Prevent scrolling on the page wrapper itself */
            background-color: var(--bg-chat);
        }

        .ledger-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
            flex-shrink: 0;
        }

        .ledger-title {
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .ledger-search-box {
            background-color: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 18px;
            padding: 0.4rem 1rem;
            display: flex;
            align-items: center;
            width: 320px;
        }

        .ledger-search-input {
            width: 100%;
            background: transparent;
            border: none;
            color: var(--text-main);
            outline: none;
            font-family: inherit;
            font-size: 0.85rem;
        }

        .ledger-table-container {
            flex: 1;
            min-height: 0; /* Critical: allows this flex child to shrink and respect parent overflow:hidden */
            overflow-y: auto; /* Let the table container scroll vertically */
            border: 1px solid var(--border-color);
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.01);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .ledger-table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }

        /* Sticky table header */
        .ledger-table th {
            position: sticky;
            top: 0;
            background-color: #171717; /* Solid color background for sticky header overlay */
            border-bottom: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            font-weight: 600;
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            z-index: 10;
        }

        .ledger-table td {
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
            vertical-align: middle;
        }

        .ledger-table tr:last-child td {
            border-bottom: none;
        }

        .ledger-table tr {
            transition: background-color 0.2s;
        }

        .ledger-table tr:hover {
            background-color: rgba(255, 255, 255, 0.02);
        }

        .ledger-table tr.highlighted-row {
            background-color: rgba(84, 63, 215, 0.15);
            border-left: 3px solid var(--primary);
            animation: flashHighlight 2.5s ease-out;
        }

        @keyframes flashHighlight {
            0% { background-color: rgba(84, 63, 215, 0.4); }
            100% { background-color: rgba(84, 63, 215, 0.15); }
        }

        .ledger-tag {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            display: inline-block;
        }

        .ledger-tag-buy {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
        }

        .ledger-tag-sell {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--accent-rose);
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-4px); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── Research / Off-topic message cards ────────────────────────── */
        .research-card {
            background: rgba(84, 63, 215, 0.06);
            border: 1px solid rgba(84, 63, 215, 0.25);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-top: 0.5rem;
            animation: fadeIn 0.3s ease;
        }

        .research-card-header {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--primary);
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .research-price-block {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            margin-bottom: 0.5rem;
        }

        .research-price-value {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text-main);
        }

        .research-price-ticker {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .research-change-up   { color: var(--accent-emerald); font-size: 0.85rem; font-weight: 500; }
        .research-change-down { color: var(--accent-rose);    font-size: 0.85rem; font-weight: 500; }

        .research-summary {
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .research-source {
            font-size: 0.72rem;
            color: #555;
            margin-top: 0.4rem;
        }

        .offtopic-card {
            background: rgba(245, 158, 11, 0.06);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: 12px;
            padding: 0.9rem 1.25rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-muted);
            line-height: 1.55;
            animation: fadeIn 0.3s ease;
        }

        /* ── Confirmation Modal ─────────────────────────────────────────── */
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.2s ease;
        }

        .modal-box {
            background: #1e1e1e;
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            width: min(480px, 92vw);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
        }

        .modal-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }

        .modal-subtitle {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 1.25rem;
        }

        .modal-trade-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-size: 0.85rem;
        }

        .modal-trade-row:last-of-type { border-bottom: none; }

        .modal-trade-key   { color: var(--text-muted); }
        .modal-trade-value { font-weight: 500; }

        .modal-ref-price {
            margin: 1.1rem 0 1.25rem;
            padding: 0.85rem 1rem;
            background: rgba(84, 63, 215, 0.08);
            border: 1px solid rgba(84, 63, 215, 0.2);
            border-radius: 10px;
            font-size: 0.88rem;
            color: var(--text-muted);
        }

        .modal-ref-price strong { color: var(--text-main); }

        .modal-direction-buy  { color: var(--accent-emerald); font-weight: 700; }
        .modal-direction-sell { color: var(--accent-rose);    font-weight: 700; }

        .modal-actions {
            display: flex;
            gap: 0.75rem;
            margin-top: 1.25rem;
        }

        .modal-btn-confirm {
            flex: 1;
            padding: 0.65rem;
            border: none;
            border-radius: 8px;
            background: var(--accent-emerald);
            color: #fff;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }

        .modal-btn-confirm:hover { background: #0ea472; }
        .modal-btn-confirm:disabled { opacity: 0.5; cursor: not-allowed; }

        .modal-btn-cancel {
            flex: 1;
            padding: 0.65rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: transparent;
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background 0.2s;
        }

        .modal-btn-cancel:hover { background: rgba(255,255,255,0.05); }
    </style>
</head>
<body>
    <!-- Sidebar Left -->
    <div class="sidebar">
        <div class="sidebar-header">
            <div class="logo-section">
                <div class="logo-dot"></div>
                <div class="logo-text">Trade Copilot</div>
            </div>
            <div class="connection-badge" id="connectionBadge">
                <div class="badge-dot"></div>
                Checking API Status...
            </div>
        </div>

        <button class="new-chat-btn" onclick="createNewSession()">
            New Chat
        </button>

        <div class="history-section">
            <div class="history-title">Previous Chats</div>
            <ul class="history-list" id="recentBookingsList">
                <li style="padding: 0.75rem; font-size: 0.8rem; color: var(--text-muted); text-align: center;">No previous chats.</li>
            </ul>
        </div>

        <div class="sidebar-footer">
            <label class="settings-toggle">
                <input type="checkbox" id="bypassChaos" checked onchange="checkConnection()">
                <span>Bypass Chaos (resilience)</span>
            </label>
            <button class="log-toggle-btn" onclick="toggleLogsPanel()">
                Toggle Developer Logs
            </button>
        </div>

        <!-- Collapsible System Logs -->
        <div class="logs-panel" id="logsPanel"></div>
    </div>

    <!-- Main Chat Workspace -->
    <div class="chat-container">
        <!-- Dashboard Tabs -->
        <div class="tabs-header">
            <button class="tab-btn active" id="tabBtnChat" onclick="switchTab('chat')">
                Copilot Chat
            </button>
            <button class="tab-btn" id="tabBtnLedger" onclick="switchTab('ledger')">
                Trade Book <span class="ledger-count" id="ledgerCount">(0)</span>
            </button>
        </div>

        <!-- Tab 1: Chat Feed View -->
        <div class="chat-view-wrapper" id="chatView">
            <div class="chat-feed" id="chatFeed">
                <!-- Feed will contain either the Welcome Screen or a Chat stream of messages -->
                <div class="welcome-panel" id="welcomePanel">
                    <div class="welcome-title">How can I help you book a trade today?</div>
                    <div class="welcome-sub">Type your request below or select one of the examples:</div>
                    <div class="chips-grid">
                        <button class="suggestion-chip" onclick="submitChip('Buy 100 MW tomorrow at $47 from Shell')">
                            <span class="chip-header">Scenario 1</span>
                            <span>Buy 100 MW tomorrow at $47 from Shell</span>
                        </button>
                        <button class="suggestion-chip" onclick="submitChip('Please sell 80 MW tomorrow at $51 to BP')">
                            <span class="chip-header">Scenario 2 (Missing hub)</span>
                            <span>Please sell 80 MW tomorrow at $51 to BP</span>
                        </button>
                        <button class="suggestion-chip" onclick="submitChip('Maybe buy some power next week from Conoco')">
                            <span class="chip-header">Scenario 3 (Ambiguous)</span>
                            <span>Maybe buy some power next week from Conoco</span>
                        </button>
                        <button class="suggestion-chip" onclick="submitChip('Buy 100 MW tomorrow at -$5 from Shell in MISO')">
                            <span class="chip-header">Scenario 4 (Conflict Validation)</span>
                            <span>Buy 100 MW tomorrow at -$5 from Shell in MISO</span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Chat Input Field wrapper -->
            <div class="input-wrapper">
                <div class="chat-width-wrapper">
                    <div class="input-box">
                        <textarea 
                            class="chat-textarea" 
                            id="chatInput" 
                            placeholder="Message Trade Copilot..."
                            onkeydown="handleInputKeydown(event)"
                        ></textarea>
                        <button class="send-btn" id="sendBtn" onclick="submitUserMessage()" disabled>
                            ➔
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab 2: Trades Ledger Grid View -->
        <div class="ledger-view-wrapper" id="ledgerView" style="display: none;">
            <div class="ledger-header-row">
                <div class="ledger-title">Trade Registry</div>
                <div class="ledger-search-box">
                    <input type="text" class="ledger-search-input" id="ledgerSearch" placeholder="Search trades by counterparty, hub, notes..." oninput="renderLedgerTable(this.value)">
                </div>
            </div>
            <div class="ledger-table-container">
                <table class="ledger-table">
                    <thead>
                        <tr>
                            <th>Trade ID</th>
                            <th>Counterparty</th>
                            <th>Direction</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>Grid Hub</th>
                            <th>Delivery Window</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody id="ledgerTbody">
                        <tr>
                            <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">Loading trades...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- ─────────────────────────────────────────────────────────────────── -->
    <!-- Confirmation Modal Overlay (hidden by default)                      -->
    <!-- ─────────────────────────────────────────────────────────────────── -->
    <div class="modal-overlay" id="confirmModal" style="display:none;" role="dialog" aria-modal="true">
        <div class="modal-box">
            <div class="modal-title" id="modalTitle">Confirm Trade Booking</div>
            <div class="modal-subtitle">Please review the trade details before proceeding.</div>

            <!-- Trade detail rows injected by JS -->
            <div id="modalTradeDetails"></div>

            <!-- Reference market price (fetched async) -->
            <div class="modal-ref-price" id="modalRefPrice">
                Fetching current market reference price...
            </div>

            <div class="modal-actions">
                <button class="modal-btn-confirm" id="modalConfirmBtn" onclick="modalProceedBooking()">Yes, Book Trade</button>
                <button class="modal-btn-cancel" onclick="closeConfirmModal()">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;
        let cardCounter = 0;
        let activeTab = 'chat';
        let allTradesCache = [];
        let sessions = {};
        let currentSessionId = null;

        // Auto-initialize connection status and poll logs
        window.addEventListener('load', () => {
            checkConnection();
            loadSessions(); // Initialize local storage chat sessions
            fetchAllTrades(); // Fetch trades count and ledger cache
            startLogPolling();
            // Periodic health poll — ensures badge auto-recovers from transient chaos 503s
            setInterval(checkConnection, 15000);

            // Set up input event listener to toggle Send button state
            const textInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendBtn');
            textInput.addEventListener('input', () => {
                sendBtn.disabled = textInput.value.trim() === '' || isProcessing;
            });
        });

        function toggleLogsPanel() {
            const panel = document.getElementById('logsPanel');
            panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
        }

        // Sessions Storage Management
        function loadSessions() {
            const data = localStorage.getItem('copilot_chat_sessions');
            if (data) {
                try {
                    sessions = JSON.parse(data);
                } catch(e) {
                    sessions = {};
                }
            }
            
            const keys = Object.keys(sessions);
            if (keys.length === 0) {
                createNewSession();
            } else {
                currentSessionId = keys[keys.length - 1];
                renderSidebarSessions();
                restoreActiveSession();
            }
        }

        function createNewSession() {
            const id = 'session_' + Date.now();
            sessions[id] = {
                id: id,
                title: 'New Chat',
                messages: []
            };
            currentSessionId = id;
            saveSessions();
            renderSidebarSessions();
            switchTab('chat');
            restoreActiveSession();
        }

        function saveSessions() {
            /* Memory management: cap at 20 sessions, 50 messages each to avoid localStorage bloat */
            const MAX_SESSIONS = 20;
            const MAX_MSGS_PER_SESSION = 50;

            // Enforce per-session message limit
            Object.values(sessions).forEach(s => {
                if (s.messages && s.messages.length > MAX_MSGS_PER_SESSION) {
                    s.messages = s.messages.slice(-MAX_MSGS_PER_SESSION);
                }
            });

            // Enforce total session limit (keep newest)
            let keys = Object.keys(sessions);
            if (keys.length > MAX_SESSIONS) {
                keys.slice(0, keys.length - MAX_SESSIONS).forEach(k => delete sessions[k]);
            }

            localStorage.setItem('copilot_chat_sessions', JSON.stringify(sessions));
        }

        function renderSidebarSessions() {
            const list = document.getElementById('recentBookingsList');
            const keys = Object.keys(sessions).reverse(); // Newest first
            
            if (keys.length === 0) {
                list.innerHTML = `<li style="padding: 0.75rem; font-size: 0.8rem; color: var(--text-muted); text-align: center;">No previous chats.</li>`;
                return;
            }
            
            list.innerHTML = keys.map(k => {
                const s = sessions[k];
                const activeClass = s.id === currentSessionId ? 'history-item active-session' : 'history-item';
                return `
                    <li class="${activeClass}" onclick="switchToSession('${s.id}')" title="${escapeHTML(s.title)}">
                        <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 190px;">${escapeHTML(s.title)}</span>
                        <span style="font-size: 0.75rem; color: var(--text-muted); cursor: pointer;" onclick="deleteSession(event, '${s.id}')">✕</span>
                    </li>
                `;
            }).join('');
        }

        function deleteSession(event, id) {
            event.stopPropagation();
            delete sessions[id];
            saveSessions();
            
            const keys = Object.keys(sessions);
            if (keys.length === 0) {
                createNewSession();
            } else {
                if (currentSessionId === id) {
                    currentSessionId = keys[keys.length - 1];
                }
                renderSidebarSessions();
                restoreActiveSession();
            }
        }

        function switchToSession(id) {
            currentSessionId = id;
            switchTab('chat');
            renderSidebarSessions();
            restoreActiveSession();
        }

        function restoreActiveSession() {
            const s = sessions[currentSessionId];
            const feed = document.getElementById('chatFeed');
            feed.innerHTML = '';
            
            if (s.messages.length === 0) {
                // Render welcome screen
                feed.innerHTML = `
                    <div class="welcome-panel" id="welcomePanel">
                        <div class="welcome-title">How can I help you book a trade today?</div>
                        <div class="welcome-sub">Type your request below or select one of the examples:</div>
                        <div class="chips-grid">
                            <button class="suggestion-chip" onclick="submitChip('Buy 100 MW tomorrow at $47 from Shell')">
                                <span class="chip-header">Scenario 1</span>
                                <span>Buy 100 MW tomorrow at $47 from Shell</span>
                            </button>
                            <button class="suggestion-chip" onclick="submitChip('Please sell 80 MW tomorrow at $51 to BP')">
                                <span class="chip-header">Scenario 2 (Missing hub)</span>
                                <span>Please sell 80 MW tomorrow at $51 to BP</span>
                            </button>
                            <button class="suggestion-chip" onclick="submitChip('Maybe buy some power next week from Conoco')">
                                <span class="chip-header">Scenario 3 (Ambiguous)</span>
                                <span>Maybe buy some power next week from Conoco</span>
                            </button>
                            <button class="suggestion-chip" onclick="submitChip('Buy 100 MW tomorrow at -$5 from Shell in MISO')">
                                <span class="chip-header">Scenario 4 (Conflict Validation)</span>
                                <span>Buy 100 MW tomorrow at -$5 from Shell in MISO</span>
                            </button>
                        </div>
                    </div>
                `;
                return;
            }
            
            s.messages.forEach(m => {
                if (m.sender === 'user') {
                    appendUserMessageUI(m.text);
                } else {
                    if (m.type === 'text') {
                        appendAssistantTextMessageUI(m.text);
                    } else if (m.type === 'card') {
                        appendAssistantCardUI(m.data, m.id, m.state);
                    }
                }
            });
            feed.scrollTop = feed.scrollHeight;
        }

        function updateCardStateInSession(msgId, key, value) {
            const s = sessions[currentSessionId];
            if (!s) return;
            const msg = s.messages.find(m => m.id === msgId);
            if (msg && msg.type === 'card') {
                if (!msg.state) msg.state = {};
                msg.state[key] = value;
                saveSessions();
            }
        }

        function saveCardDraftState(cardId) {
            const direction = document.getElementById(`${cardId}_direction`).value;
            const quantity_mw = document.getElementById(`${cardId}_quantity_mw`).value;
            const price_per_mwh = document.getElementById(`${cardId}_price_per_mwh`).value;
            const counterparty = document.getElementById(`${cardId}_counterparty`).value;
            const delivery_start = document.getElementById(`${cardId}_delivery_start`).value;
            const delivery_end = document.getElementById(`${cardId}_delivery_end`).value;
            const hub = document.getElementById(`${cardId}_hub`).value;
            const notes = document.getElementById(`${cardId}_notes`).value;

            const s = sessions[currentSessionId];
            if (!s) return;
            const msg = s.messages.find(m => m.id === cardId);
            if (msg && msg.type === 'card') {
                if (!msg.state) msg.state = {};
                msg.state.edited_values = {
                    direction,
                    quantity_mw,
                    price_per_mwh,
                    counterparty,
                    delivery_start,
                    delivery_end,
                    hub,
                    notes
                };
                saveSessions();
            }
        }

        async function checkConnection() {
            const bypassEl = document.getElementById('bypassChaos');
            const bypass = bypassEl ? bypassEl.checked : true;
            const badge = document.getElementById('connectionBadge');
            if (!badge) return; // Guard: DOM not ready
            const controller = new AbortController();
            // 4 s timeout — backend health check takes ~700 ms, give it headroom
            const timeoutId = setTimeout(() => controller.abort(), 4000);
            try {
                const res = await fetch(`/api/health?bypass_chaos=${bypass}`, { signal: controller.signal });
                clearTimeout(timeoutId);
                const data = await res.json();
                if (data.repository_status === 'ok') {
                    badge.innerHTML = '<span class="badge-dot" style="background: var(--accent-emerald)"></span>Connected';
                } else {
                    badge.innerHTML = '<span class="badge-dot" style="background: var(--accent-amber)"></span>Repo Chaos Active';
                }
            } catch (e) {
                clearTimeout(timeoutId);
                badge.innerHTML = '<span class="badge-dot" style="background: var(--accent-rose)"></span>Disconnected';
            }
        }

        function handleInputKeydown(event) {
            const key = event.key || event.code;
            const code = event.keyCode || event.which;
            if ((key === 'Enter' || code === 13) && !event.shiftKey) {
                event.preventDefault();
                submitUserMessage();
            }
        }

        function submitChip(text) {
            document.getElementById('chatInput').value = text;
            submitUserMessage();
        }

        function switchTab(tab) {
            activeTab = tab;
            const chatBtn = document.getElementById('tabBtnChat');
            const ledgerBtn = document.getElementById('tabBtnLedger');
            const chatView = document.getElementById('chatView');
            const ledgerView = document.getElementById('ledgerView');
            
            if (tab === 'chat') {
                chatBtn.classList.add('active');
                ledgerBtn.classList.remove('active');
                chatView.style.display = 'flex';
                ledgerView.style.display = 'none';
            } else {
                chatBtn.classList.remove('active');
                ledgerBtn.classList.add('active');
                chatView.style.display = 'none';
                ledgerView.style.display = 'flex';
                renderLedgerTable(document.getElementById('ledgerSearch').value);
            }
        }

        function handleHistoryItemClick(tradeId) {
            switchTab('ledger');
            
            // Highlight the selected row
            setTimeout(() => {
                const row = document.getElementById(`ledger_row_${tradeId}`);
                if (row) {
                    const highlighted = document.querySelectorAll('.highlighted-row');
                    highlighted.forEach(el => el.classList.remove('highlighted-row'));
                    
                    row.classList.add('highlighted-row');
                    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);
        }

        async function fetchAllTrades() {
            const bypass = document.getElementById('bypassChaos').checked;
            try {
                const res = await fetch(`/api/trades?bypass_chaos=${bypass}`);
                allTradesCache = await res.json();
                
                const countBadge = document.getElementById('ledgerCount');
                if (countBadge) {
                    countBadge.textContent = `(${allTradesCache.length})`;
                }
                
                if (activeTab === 'ledger') {
                    renderLedgerTable(document.getElementById('ledgerSearch').value);
                }
            } catch (e) {
                console.error("Failed to load trades:", e);
            }
        }

        function renderLedgerTable(filterText = '') {
            const tbody = document.getElementById('ledgerTbody');
            const term = filterText.toLowerCase().trim();
            
            let filtered = allTradesCache;
            if (term) {
                filtered = allTradesCache.filter(t => 
                    (t.trade_id && String(t.trade_id).includes(term)) ||
                    (t.counterparty && t.counterparty.toLowerCase().includes(term)) ||
                    (t.hub && t.hub.toLowerCase().includes(term)) ||
                    (t.direction && t.direction.toLowerCase().includes(term)) ||
                    (t.notes && t.notes.toLowerCase().includes(term))
                );
            }
            
            if (filtered.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">
                            No trades found.
                        </td>
                    </tr>
                `;
                return;
            }
            
            tbody.innerHTML = filtered.map(t => {
                const tagClass = t.direction === 'BUY' ? 'ledger-tag-buy' : 'ledger-tag-sell';
                const formattedPrice = t.price_per_mwh ? `$${parseFloat(t.price_per_mwh).toFixed(2)}` : '$0.00';
                
                return `
                    <tr id="ledger_row_${t.trade_id}">
                        <td><strong>#${t.trade_id}</strong></td>
                        <td>${escapeHTML(t.counterparty || 'N/A')}</td>
                        <td><span class="ledger-tag ${tagClass}">${t.direction}</span></td>
                        <td><strong>${t.quantity_mw} MW</strong></td>
                        <td>${formattedPrice}</td>
                        <td><span style="font-weight: 500;">${escapeHTML(t.hub || 'N/A')}</span></td>
                        <td style="font-size: 0.8rem; color: var(--text-muted);">
                            <div>${t.delivery_start || 'N/A'}</div>
                            <div style="font-size: 0.7rem; color: #666; margin-top: 0.1rem;">to ${t.delivery_end || 'N/A'}</div>
                        </td>
                        <td style="font-size: 0.8rem; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHTML(t.notes || '')}">
                            ${escapeHTML(t.notes || '—')}
                        </td>
                    </tr>
                `;
            }).join('');
        }

        async function submitUserMessage() {
            const textInput = document.getElementById('chatInput');
            const query = textInput.value.trim();
            if (!query || isProcessing) return;

            // Clear welcome panel if present
            const welcome = document.getElementById('welcomePanel');
            if (welcome) welcome.remove();

            isProcessing = true;
            textInput.value = '';
            textInput.style.height = 'auto';
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;

            try {
                // 1. Save and Append User Bubble
                appendUserMessage(query);

                // ── FAST CLIENT-SIDE ROUTER ────────────────────────────────────
                // Classify intent locally. Conversational queries answered instantly
                // with zero network latency (no Gemini call needed).
                const localIntent = fastClassifyIntent(query);

                if (localIntent === 'conversational') {
                    setTimeout(() => {
                        try {
                            appendAssistantTextMessage(getConversationalReply(query));
                        } finally {
                            isProcessing = false;
                            sendBtn.disabled = textInput.value.trim() === '';
                        }
                    }, 120); // Tiny delay for natural feel
                    return;
                }
                // ──────────────────────────────────────────────────────────

                // 2. Append Assistant Typing Indicator
                const typingId = appendAssistantTyping();

                // 3. Make Translation API call (Gemini classifies + parses in one pass)
                const bypass = document.getElementById('bypassChaos').checked;
                try {
                    const res = await fetch('/api/parse', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: query, bypass_chaos: bypass })
                    });

                    const typingBubble = document.getElementById(typingId);
                    if (typingBubble) typingBubble.remove();

                    if (!res.ok) {
                        throw new Error('HTTP ' + res.status);
                    }

                    const data = await res.json();

                    /* Route by query_type returned from Gemini classification */
                    if (data.query_type === 'off_topic') {
                        appendOffTopicMessage(data.message);
                    } else if (data.query_type === 'market_research') {
                        appendResearchCard(data.research_query);
                    } else {
                        appendAssistantCard(data);
                    }

                    checkConnection();
                } catch (e) {
                    const typingBubble = document.getElementById(typingId);
                    if (typingBubble) typingBubble.remove();
                    appendAssistantTextMessage(`Failed to process request: ${e.message}. Please check connection or logs.`);
                }
            } catch (outerErr) {
                console.error("Uncaught exception in submitUserMessage:", outerErr);
            } finally {
                // For non-conversational path, ensure isProcessing and sendBtn are restored
                if (fastClassifyIntent(query) !== 'conversational') {
                    isProcessing = false;
                    sendBtn.disabled = textInput.value.trim() === '';
                }
            }
        }

        /**
         * fastClassifyIntent — O(1) regex-based intent classification.
         * Runs entirely client-side, zero API/network latency.
         * @param {string} text
         * @returns {'conversational'|'trade_likely'|'research_likely'|'unknown'}
         */
        function fastClassifyIntent(text) {
            const t = text.trim().toLowerCase();

            // ── Conversational patterns ─────────────────────────────────────────
            if (/^(hi+|hello+|hey+|howdy|yo+|sup|good\s*(morning|afternoon|evening|day)|what'?s up)[\s!.?,]*$/.test(t)) return 'conversational';
            if (/^(how are you|how r u|how you doing|how's it going|how do you do)[\s!.?,]*$/.test(t)) return 'conversational';
            if (/^(thanks?[\s!.,]*|thank you[\s!.,]*|thx|ty|cheers)$/.test(t)) return 'conversational';
            if (/^(bye+|goodbye+|see\s*you|cya|later|take care)[\s!.?,]*$/.test(t)) return 'conversational';
            if (/^(ok+|okay|got it|alright|sounds good|cool|great|nice|perfect|understood)[\s!.?,]*$/.test(t)) return 'conversational';
            if (/^(who are you|what are you|what can you do|help|what do you do)[\s!.?,]*$/.test(t)) return 'conversational';
            if (/^(what is this|tell me about yourself)[\s!.?,]*$/.test(t)) return 'conversational';

            // ── Trade signals ────────────────────────────────────────────────
            if (/\b(buy|sell|purchase|acquire)\b/.test(t) && /\b(mw|mwh|megawatt)\b/.test(t)) return 'trade_likely';
            if (/\b(pjm|miso|ercot|spp|caiso)\b/.test(t) && /\b(buy|sell|trade|book)\b/.test(t)) return 'trade_likely';
            if (/\b(counterparty|shell|bp|conoco|vitol|nextera|edf|engie)\b/.test(t) && /\b(buy|sell|mw|power|electricity)\b/.test(t)) return 'trade_likely';

            // ── Research / market data signals ───────────────────────────────
            if (/\b(stock\s*price|share\s*price|ticker|etf|mutual\s*fund|index\s*fund|crypto|bitcoin|ethereum)\b/.test(t)) return 'research_likely';
            if (/\b(crude\s*oil|gold\s*price|silver|commodity|natural\s*gas)\b/.test(t)) return 'research_likely';
            if (/^(what'?s?\s*(the\s*)?price\s*of|how\s*much\s*is|current\s*price\s*of)\b/.test(t)) return 'research_likely';

            return 'unknown'; // Let Gemini classify + handle
        }

        /**
         * getConversationalReply — returns a curated reply for common conversational patterns.
         * No API call required — instant, deterministic response.
         * @param {string} text
         * @returns {string}
         */
        function getConversationalReply(text) {
            const t = text.trim().toLowerCase();

            if (/\b(who are you|what are you|tell me about yourself|what is this)\b/.test(t)) {
                return "I'm your Power Trade Copilot \u2014 an AI assistant for booking and managing electricity trades. I can:\n\n\u2022 Book power trades (BUY or SELL electricity at a hub)\n\u2022 Look up current stock, ETF, or commodity prices\n\u2022 Research energy market data\n\nTry: \"Buy 100 MW tomorrow at $47 from Shell in PJM\"";
            }
            if (/\b(what can you do|help)\b/.test(t)) {
                return "Here's what I can help with:\n\u2022 Book energy trades \u2014 tell me direction, MW, price, counterparty, hub and delivery window\n\u2022 Look up stock prices \u2014 e.g. \"What's the price of AAPL?\"\n\u2022 Research commodity prices \u2014 e.g. \"crude oil price today\"\n\u2022 Review previous trades in the Trade Book tab";
            }
            if (/^(thanks?|thank you|thx|ty|cheers)/.test(t)) {
                return "You're welcome! Let me know if you'd like to book another trade or look up any market data.";
            }
            if (/^(bye|goodbye|see\s*you|cya|later|take care)/.test(t)) {
                return "Goodbye! Come back whenever you need to book trades or check market data.";
            }
            if (/^(ok+|okay|got it|alright|sounds good|cool|great|nice|perfect|understood)/.test(t)) {
                return "Got it. What would you like to do next? I can book a trade or look up market data for you.";
            }
            if (/^(how are you|how r u|how you doing|how's it going)/.test(t)) {
                return "Running smoothly! Ready to help \u2014 what trade would you like to book, or what market data do you need?";
            }
            // Default greeting
            return "Hello! I'm your Power Trade Copilot. Ready to book trades or research the market \u2014 what can I help you with today?";
        }


        /**
         * appendOffTopicMessage — renders a styled amber-bordered message explaining
         * the assistant cannot help with off-topic queries.
         */
        function appendOffTopicMessage(text) {
            const s = sessions[currentSessionId];
            if (s) {
                s.messages.push({ sender: 'ai', type: 'text', text: text });
                saveSessions();
            }
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            row.className = 'msg-row';
            row.innerHTML = `
                <div class="avatar ai">AI</div>
                <div class="msg-content">
                    <div class="offtopic-card">
                        <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#f59e0b;margin-bottom:0.5rem;">Out of Scope</div>
                        ${escapeHTML(text)}
                    </div>
                </div>
            `;
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
        }

        /**
         * appendResearchCard — renders a research result card.
         * Calls /api/market-data asynchronously and populates a rich price card.
         */
        async function appendResearchCard(researchQuery) {
            const feed = document.getElementById('chatFeed');
            const cardId = 'research_' + Date.now();

            const row = document.createElement('div');
            row.className = 'msg-row';
            row.id = cardId + '_row';
            row.innerHTML = `
                <div class="avatar ai">AI</div>
                <div class="msg-content">
                    <div class="research-card" id="${cardId}">
                        <div class="research-card-header">Market Data &bull; ${escapeHTML(researchQuery)}</div>
                        <div style="color:var(--text-muted);font-size:0.85rem;">Fetching market data...</div>
                    </div>
                </div>
            `;
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;

            try {
                const res = await fetch('/api/market-data?' + new URLSearchParams({ query: researchQuery }));
                const data = await res.json();
                const card = document.getElementById(cardId);
                if (!card) return;

                if (data.type === 'stock' && data.price) {
                    // Rich stock price card
                    const changeDir = data.change > 0 ? 'up' : 'down';
                    const changeClass = data.change > 0 ? 'research-change-up' : 'research-change-down';
                    const changeSign = data.change > 0 ? '+' : '';
                    card.innerHTML = `
                        <div class="research-card-header">Market Data &bull; ${escapeHTML(data.source)}</div>
                        <div class="research-price-block">
                            <span class="research-price-value">${data.currency} ${data.price.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
                            <span class="research-price-ticker">${escapeHTML(data.ticker)}</span>
                        </div>
                        <div style="margin-bottom:0.5rem;">
                            <span class="${changeClass}">${changeSign}${data.change?.toFixed(4)} (${changeSign}${data.change_pct?.toFixed(2)}%)</span>
                            &nbsp;<span style="font-size:0.78rem;color:#555;">vs prev close ${data.currency} ${data.prev_close?.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
                        </div>
                        <div class="research-summary">${escapeHTML(data.name || '')} &bull; ${escapeHTML(data.exchange || '')}</div>
                        <div class="research-source">Source: ${escapeHTML(data.source)} &bull; Data may be delayed 15-20 min.</div>
                    `;
                } else if (data.summary) {
                    // General knowledge / DuckDuckGo answer
                    card.innerHTML = `
                        <div class="research-card-header">Market Research &bull; ${escapeHTML(data.source || 'Web')}</div>
                        <div class="research-summary">${escapeHTML(data.summary)}</div>
                        ${ data.related && data.related.length > 0
                            ? '<div style="margin-top:0.5rem;">' +
                                data.related.map(r => `<div style="font-size:0.8rem;color:var(--text-muted);margin-top:0.3rem;">• ${escapeHTML(r)}</div>`).join('') +
                              '</div>'
                            : '' }
                        <div class="research-source">Source: ${escapeHTML(data.source || 'DuckDuckGo')} &bull; Verify before trading.</div>
                    `;
                } else {
                    card.innerHTML = `
                        <div class="research-card-header">Market Research</div>
                        <div class="research-summary">${escapeHTML(data.summary || 'No data available.')}</div>
                    `;
                }

                // Save summary to session
                const s = sessions[currentSessionId];
                if (s) {
                    s.messages.push({ sender: 'ai', type: 'text', text: data.summary || 'No data found.' });
                    saveSessions();
                }
            } catch (err) {
                const card = document.getElementById(cardId);
                if (card) card.innerHTML = `<div class="research-summary" style="color:var(--accent-rose);">Failed to fetch market data: ${escapeHTML(err.message)}</div>`;
            }

            feed.scrollTop = feed.scrollHeight;
        }

        function appendUserMessage(text) {
            const s = sessions[currentSessionId];
            if (s) {
                s.messages.push({ sender: 'user', text: text });
                
                // Update title on first query
                if (s.title === 'New Chat' || s.title === '') {
                    s.title = text.substring(0, 30) + (text.length > 30 ? '...' : '');
                }
                saveSessions();
                renderSidebarSessions();
            }
            appendUserMessageUI(text);
        }

        function appendUserMessageUI(text) {
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            row.className = 'msg-row user';
            row.innerHTML = `
                <div class="msg-content">${escapeHTML(text)}</div>
            `;
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
        }

        function appendAssistantTextMessage(text) {
            const s = sessions[currentSessionId];
            if (s) {
                s.messages.push({ sender: 'ai', type: 'text', text: text });
                saveSessions();
            }
            appendAssistantTextMessageUI(text);
        }

        function appendAssistantTextMessageUI(text) {
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            row.className = 'msg-row';
            row.innerHTML = `
                <div class="avatar ai">AI</div>
                <div class="msg-content">${escapeHTML(text)}</div>
            `;
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
        }

        function appendAssistantTyping() {
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            const id = 'typing_' + Date.now();
            row.id = id;
            row.className = 'msg-row';
            row.innerHTML = `
                <div class="avatar ai">AI</div>
                <div class="msg-content">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
            return id;
        }

        function playSuccessJingle() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                
                // First Note: C5 (523.25 Hz)
                const osc1 = ctx.createOscillator();
                const gain1 = ctx.createGain();
                osc1.type = 'sine';
                osc1.frequency.setValueAtTime(523.25, ctx.currentTime);
                gain1.gain.setValueAtTime(0, ctx.currentTime);
                gain1.gain.linearRampToValueAtTime(0.08, ctx.currentTime + 0.03);
                gain1.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
                osc1.connect(gain1);
                gain1.connect(ctx.destination);
                osc1.start(ctx.currentTime);
                osc1.stop(ctx.currentTime + 0.35);
                
                // Second Note: G5 (783.99 Hz) triggered slightly later
                const osc2 = ctx.createOscillator();
                const gain2 = ctx.createGain();
                osc2.type = 'sine';
                osc2.frequency.setValueAtTime(783.99, ctx.currentTime + 0.08);
                gain2.gain.setValueAtTime(0, ctx.currentTime + 0.08);
                gain2.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.11);
                gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
                osc2.connect(gain2);
                gain2.connect(ctx.destination);
                osc2.start(ctx.currentTime + 0.08);
                osc2.stop(ctx.currentTime + 0.5);
            } catch (e) {
                console.warn("Could not play synthesized success jingle: ", e);
            }
        }

        function validateCardInputs(cardId) {
            const dirEl = document.getElementById(`${cardId}_direction`);
            const qtyEl = document.getElementById(`${cardId}_quantity_mw`);
            const priceEl = document.getElementById(`${cardId}_price_per_mwh`);
            const cpEl = document.getElementById(`${cardId}_counterparty`);
            const startEl = document.getElementById(`${cardId}_delivery_start`);
            const endEl = document.getElementById(`${cardId}_delivery_end`);
            const hubEl = document.getElementById(`${cardId}_hub`);
            
            const dirErr = document.getElementById(`${cardId}_direction_err`);
            const qtyErr = document.getElementById(`${cardId}_quantity_mw_err`);
            const priceErr = document.getElementById(`${cardId}_price_per_mwh_err`);
            const cpErr = document.getElementById(`${cardId}_counterparty_err`);
            const startErr = document.getElementById(`${cardId}_delivery_start_err`);
            const endErr = document.getElementById(`${cardId}_delivery_end_err`);
            const hubErr = document.getElementById(`${cardId}_hub_err`);
            
            const submitBtn = document.querySelector(`#${cardId}_actions button.card-btn-success`);
            const alertDiv = document.getElementById(`${cardId}_alert`);
            
            let errorsCount = 0;
            
            const setError = (el, errEl, condition, msg) => {
                if (condition) {
                    el.classList.add('validation-error-border');
                    errEl.textContent = msg;
                    errEl.style.display = 'block';
                    errorsCount++;
                } else {
                    el.classList.remove('validation-error-border');
                    el.classList.remove('missing-warning');
                    errEl.style.display = 'none';
                }
            };
            
            // Direction check
            setError(dirEl, dirErr, !dirEl.value, "Direction is required.");
            
            // Quantity Check
            const qtyVal = parseFloat(qtyEl.value);
            setError(qtyEl, qtyErr, isNaN(qtyVal) || qtyVal <= 0 || qtyVal > 10000, 
                     isNaN(qtyVal) ? "Quantity is required." : "Quantity must be > 0 and <= 10000 MW.");
                     
            // Price Check
            const priceVal = parseFloat(priceEl.value);
            setError(priceEl, priceErr, isNaN(priceVal) || priceVal <= 0 || priceVal > 1000, 
                     isNaN(priceVal) ? "Price is required." : "Price must be > 0 and <= $1000/MWh.");
                     
            // Counterparty Check
            setError(cpEl, cpErr, !cpEl.value.trim(), "Counterparty name is required.");
            
            // Hub Check
            setError(hubEl, hubErr, !hubEl.value.trim(), "Grid hub is required.");
            
            // Start/End Dates Check
            let startValid = false;
            let endValid = false;
            let startDt, endDt;
            
            const startStr = startEl.value.trim();
            if (!startStr) {
                setError(startEl, startErr, true, "Start date-time is required.");
            } else {
                try {
                    startDt = new Date(startStr.replace("Z", ""));
                    if (isNaN(startDt.getTime())) throw new Error();
                    setError(startEl, startErr, false);
                    startValid = true;
                } catch (e) {
                    setError(startEl, startErr, true, "Format must be YYYY-MM-DDTHH:MM:SSZ.");
                }
            }
            
            const endStr = endEl.value.trim();
            if (!endStr) {
                setError(endEl, endErr, true, "End date-time is required.");
            } else {
                try {
                    endDt = new Date(endStr.replace("Z", ""));
                    if (isNaN(endDt.getTime())) throw new Error();
                    setError(endEl, endErr, false);
                    endValid = true;
                } catch (e) {
                    setError(endEl, endErr, true, "Format must be YYYY-MM-DDTHH:MM:SSZ.");
                }
            }
            
            if (startValid && endValid) {
                setError(endEl, endErr, endDt <= startDt, "Delivery end must be chronologically after start.");
            }
            
            // Handle card warning card state
            if (errorsCount > 0) {
                if (submitBtn) {
                    submitBtn.disabled = true;
                }
                if (alertDiv) {
                    alertDiv.className = "card-alert error";
                    alertDiv.innerHTML = `<strong>Invalid Parameters:</strong> Please correct the highlighted errors below before booking.`;
                }
            } else {
                if (submitBtn) {
                    submitBtn.disabled = false;
                }
                if (alertDiv) {
                    alertDiv.className = "card-alert";
                    alertDiv.style.backgroundColor = "rgba(16, 185, 129, 0.04)";
                    alertDiv.style.borderLeftColor = "var(--accent-emerald)";
                    alertDiv.style.color = "var(--accent-emerald)";
                    alertDiv.innerHTML = `<strong>Ready to Book:</strong> Validation passed. Click below to persist the trade.`;
                }
            }

            // Save the edited values locally to this card's session state
            saveCardDraftState(cardId);
        }

        function appendAssistantCard(data) {
            const s = sessions[currentSessionId];
            const msgId = 'msg_' + Date.now();
            if (s) {
                s.messages.push({
                    sender: 'ai',
                    type: 'card',
                    id: msgId,
                    data: data,
                    state: { booked: false, trade_id: null, cancelled: false }
                });
                saveSessions();
            }
            appendAssistantCardUI(data, msgId, { booked: false, trade_id: null, cancelled: false });
        }

        function appendAssistantCardUI(data, cardId, state = {}) {
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            row.className = 'msg-row';
            
            const currentDraft = state.edited_values || data.parsed_trade;
            const isBooked = state.booked || false;
            const isCancelled = state.cancelled || false;
            
            if (isCancelled) {
                row.innerHTML = `
                    <div class="avatar ai">AI</div>
                    <div class="msg-content">
                        <p style="color:var(--text-muted); font-style:italic;">Trade draft cancelled.</p>
                    </div>
                `;
                feed.appendChild(row);
                feed.scrollTop = feed.scrollHeight;
                return;
            }

            const confClass = data.confidence_score >= 0.70 ? 'conf-high' : 'conf-low';
            const confLabel = data.confidence_score >= 0.70 ? 'High Confidence' : 'Low Confidence';
            const percentage = Math.round(data.confidence_score * 100) + '%';
            
            // Warnings/Alert inside bubble
            let alertHtml = '';
            if (isBooked) {
                alertHtml = ''; // Hidden, replaced by success badge in actions
            } else if (!data.ready_for_booking) {
                alertHtml = `
                    <div class="card-alert" id="${cardId}_alert">
                        <strong>Missing/Invalid Fields:</strong> Please correct or fill in the highlighted parameters below.
                    </div>
                `;
            } else {
                alertHtml = `
                    <div class="card-alert" style="background-color:rgba(16,185,129,0.04); border-left:3px solid var(--accent-emerald); color:var(--accent-emerald);" id="${cardId}_alert">
                        <strong>Ready to Book:</strong> Trade details verified.
                    </div>
                `;
            }

            // Input form fields
            const getFieldClass = (f) => data.missing_fields.includes(f) ? 'form-input missing-warning' : 'form-input';
            
            const directionsOptions = `
                <select class="${getFieldClass('direction')}" id="${cardId}_direction" onchange="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                    <option value="" ${!currentDraft.direction ? 'selected' : ''}>-- Choose --</option>
                    <option value="BUY" ${currentDraft.direction === 'BUY' ? 'selected' : ''}>BUY</option>
                    <option value="SELL" ${currentDraft.direction === 'SELL' ? 'selected' : ''}>SELL</option>
                </select>
            `;

            let formStyle = (data.ready_for_booking && !isBooked) ? 'display: none;' : 'display: table;';
            let summaryStyle = (data.ready_for_booking && !isBooked) ? 'display: block;' : 'display: none;';

            let actionsHtml = '';
            if (isBooked) {
                actionsHtml = `
                    <div class="status-badge-inline success">
                        ✓ Persisted to Repository as Trade ID: <strong>#${state.trade_id}</strong>
                    </div>
                `;
            } else {
                actionsHtml = `
                    <div class="card-actions" id="${cardId}_actions">
                        <button class="card-btn card-btn-success" onclick="bookTradeFromCard('${cardId}')">${data.ready_for_booking ? 'Confirm & Book' : 'Submit corrections'}</button>
                        <button class="card-btn card-btn-secondary" onclick="cancelCard('${cardId}')">Cancel</button>
                    </div>
                `;
            }

            row.innerHTML = `
                <div class="avatar ai">AI</div>
                <div class="msg-content">
                    <p style="margin-bottom:0.5rem;">Here is the trade extraction draft:</p>
                    <div class="trade-card" id="${cardId}">
                        <div class="trade-card-header">
                            <span class="card-title">Structured Trade Draft</span>
                            <span class="conf-badge ${confClass}">${confLabel} (${percentage})</span>
                        </div>
                        
                        ${alertHtml}

                        <!-- Simplified Text Summary -->
                        <div id="${cardId}_summary" style="${summaryStyle} padding: 1rem; border-bottom: 1px solid var(--border-color); line-height: 1.6;">
                            <div style="font-size: 1.05rem; font-weight: 600; color: var(--accent-emerald); margin-bottom: 0.4rem;">
                                Ready to book trade draft
                            </div>
                            <div style="font-size: 0.95rem; margin-bottom: 0.4rem;">
                                <strong>${currentDraft.direction || 'TRADE'}</strong> of <strong>${currentDraft.quantity_mw || 0} MW</strong> @ <strong>$${currentDraft.price_per_mwh ? parseFloat(currentDraft.price_per_mwh).toFixed(2) : '0.00'}/MWh</strong> with <strong>${currentDraft.counterparty || 'Unknown'}</strong> on <strong>${currentDraft.hub || 'Unknown'}</strong>.
                            </div>
                            <div style="font-size: 0.8rem; color: var(--text-muted);">
                                Delivery window: ${currentDraft.delivery_start || 'N/A'} to ${currentDraft.delivery_end || 'N/A'}
                            </div>
                            <a href="#" style="color: #6854e4; font-size: 0.8rem; text-decoration: none; display: inline-block; margin-top: 0.5rem; font-weight: 500;" onclick="toggleCardForm(event, '${cardId}')">Edit parameters</a>
                        </div>

                        <table class="trade-form-table" id="${cardId}_form_table" style="${formStyle}">
                            <tbody>
                                <tr>
                                    <td class="field-label">Direction</td>
                                    <td>
                                        ${directionsOptions}
                                        <div class="field-error-text" id="${cardId}_direction_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Quantity (MW)</td>
                                    <td>
                                        <input type="number" step="any" class="${getFieldClass('quantity_mw')}" id="${cardId}_quantity_mw" value="${currentDraft.quantity_mw || ''}" placeholder="E.g. 120" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                                        <div class="field-error-text" id="${cardId}_quantity_mw_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Price ($/MWh)</td>
                                    <td>
                                        <input type="number" step="any" class="${getFieldClass('price_per_mwh')}" id="${cardId}_price_per_mwh" value="${currentDraft.price_per_mwh || ''}" placeholder="E.g. 45.50" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                                        <div class="field-error-text" id="${cardId}_price_per_mwh_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Counterparty</td>
                                    <td>
                                        <input type="text" class="${getFieldClass('counterparty')}" id="${cardId}_counterparty" value="${currentDraft.counterparty || ''}" placeholder="E.g. Shell" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                                        <div class="field-error-text" id="${cardId}_counterparty_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Delivery Start</td>
                                    <td>
                                        <input type="text" class="${getFieldClass('delivery_start')}" id="${cardId}_delivery_start" value="${currentDraft.delivery_start || ''}" placeholder="YYYY-MM-DDTHH:MM:SSZ" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                                        <div class="field-error-text" id="${cardId}_delivery_start_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Delivery End</td>
                                    <td>
                                        <input type="text" class="${getFieldClass('delivery_end')}" id="${cardId}_delivery_end" value="${currentDraft.delivery_end || ''}" placeholder="YYYY-MM-DDTHH:MM:SSZ" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                                        <div class="field-error-text" id="${cardId}_delivery_end_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Hub</td>
                                    <td>
                                        <input type="text" class="${getFieldClass('hub')}" id="${cardId}_hub" value="${currentDraft.hub || ''}" placeholder="E.g. MISO, PJM" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}>
                                        <div class="field-error-text" id="${cardId}_hub_err"></div>
                                    </td>
                                </tr>
                                <tr>
                                    <td class="field-label">Notes</td>
                                    <td><input type="text" class="form-input" id="${cardId}_notes" value="${currentDraft.notes || ''}" oninput="validateCardInputs('${cardId}')" ${isBooked ? 'disabled' : ''}></td>
                                </tr>
                            </tbody>
                        </table>

                        <div class="card-reasoning">
                            <strong>Gemini reasoning:</strong> ${escapeHTML(data.reasoning)}
                        </div>

                        ${actionsHtml}
                    </div>
                </div>
            `;
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
            
            // Only validate card inputs if not booked to avoid rendering validation marks on finished cards
            if (!isBooked) {
                validateCardInputs(cardId);
            }
        }

        function toggleCardForm(event, cardId) {
            event.preventDefault();
            document.getElementById(cardId + '_summary').style.display = 'none';
            document.getElementById(cardId + '_form_table').style.display = 'table';
            validateCardInputs(cardId);
        }

        /**
         * bookTradeFromCard — reads current card field values and shows the confirmation modal.
         * The modal displays a trade summary and fetches a market reference price before booking.
         * @param {string} cardId - The ID of the trade card element.
         */
        function bookTradeFromCard(cardId) {
            // Read all field values from the card form
            const direction    = document.getElementById(`${cardId}_direction`).value;
            const quantity_mw  = document.getElementById(`${cardId}_quantity_mw`).value;
            const price_per_mwh = document.getElementById(`${cardId}_price_per_mwh`).value;
            const counterparty = document.getElementById(`${cardId}_counterparty`).value.trim();
            const delivery_start = document.getElementById(`${cardId}_delivery_start`).value.trim();
            const delivery_end   = document.getElementById(`${cardId}_delivery_end`).value.trim();
            const hub   = document.getElementById(`${cardId}_hub`).value.trim();
            const notes = document.getElementById(`${cardId}_notes`).value.trim();

            // Store pending payload globally so modalProceedBooking can retrieve it
            window._pendingBooking = {
                cardId,
                payload: {
                    direction,
                    quantity_mw: quantity_mw ? parseFloat(quantity_mw) : null,
                    price_per_mwh: price_per_mwh ? parseFloat(price_per_mwh) : null,
                    counterparty,
                    delivery_start,
                    delivery_end,
                    hub,
                    notes: notes || null
                }
            };

            // Populate modal trade summary rows
            const dirClass = direction === 'BUY' ? 'modal-direction-buy' : 'modal-direction-sell';
            document.getElementById('modalTitle').textContent =
                `Confirm ${direction || 'Trade'} — ${counterparty || 'Unknown Counterparty'}`;
            document.getElementById('modalTradeDetails').innerHTML = `
                <div class="modal-trade-row">
                    <span class="modal-trade-key">Direction</span>
                    <span class="modal-trade-value ${dirClass}">${direction || '—'}</span>
                </div>
                <div class="modal-trade-row">
                    <span class="modal-trade-key">Quantity</span>
                    <span class="modal-trade-value">${quantity_mw ? quantity_mw + ' MW' : '—'}</span>
                </div>
                <div class="modal-trade-row">
                    <span class="modal-trade-key">Your Price</span>
                    <span class="modal-trade-value">${price_per_mwh ? '$' + parseFloat(price_per_mwh).toFixed(2) + ' / MWh' : '—'}</span>
                </div>
                <div class="modal-trade-row">
                    <span class="modal-trade-key">Counterparty</span>
                    <span class="modal-trade-value">${escapeHTML(counterparty || '—')}</span>
                </div>
                <div class="modal-trade-row">
                    <span class="modal-trade-key">Hub</span>
                    <span class="modal-trade-value">${escapeHTML(hub || '—')}</span>
                </div>
                <div class="modal-trade-row">
                    <span class="modal-trade-key">Delivery Window</span>
                    <span class="modal-trade-value" style="font-size:0.8rem;">${delivery_start || '—'} → ${delivery_end || '—'}</span>
                </div>
            `;

            // Reset reference price section and fetch async
            const refEl = document.getElementById('modalRefPrice');
            refEl.innerHTML = '<span style="color:var(--text-muted);">Fetching market reference price...</span>';
            fetchReferencePrice(hub, direction, parseFloat(price_per_mwh) || null, refEl);

            // Show the modal
            document.getElementById('confirmModal').style.display = 'flex';
            document.getElementById('modalConfirmBtn').disabled = false;
        }

        /** Close the confirmation modal without booking. */
        function closeConfirmModal() {
            document.getElementById('confirmModal').style.display = 'none';
            window._pendingBooking = null;
        }

        /**
         * Fetches a market reference price for the power hub or a related equity.
         * Electricity spot prices require commercial data feeds; we display a note about this
         * and fall back to a publicly-quoted proxy or historical range.
         */
        async function fetchReferencePrice(hub, direction, tradePrice, refEl) {
            // Map common power hubs to publicly visible proxy tickers or known ranges
            const hubProxies = {
                'PJM':   { query: 'PJM electricity day-ahead price', range: '$25–$85' },
                'MISO':  { query: 'MISO electricity spot price', range: '$20–$75' },
                'ERCOT': { query: 'ERCOT real-time electricity price', range: '$15–$120' },
                'SPP':   { query: 'SPP electricity market price', range: '$18–$70' },
                'CAISO': { query: 'CAISO California electricity price', range: '$30–$120' },
            };

            const proxy = hub ? hubProxies[hub.toUpperCase()] : null;

            // Build a context-aware reference display
            let baseMsg = '';
            if (tradePrice) {
                const dir = direction === 'BUY' ? 'buying' : 'selling';
                baseMsg = `<div style="margin-bottom:0.5rem;">You are <strong>${dir}</strong> at <strong>$${tradePrice.toFixed(2)}/MWh</strong>.</div>`;
            }

            if (proxy) {
                refEl.innerHTML = `
                    ${baseMsg}
                    <div>Typical ${hub.toUpperCase()} spot range: <strong>${proxy.range}/MWh</strong></div>
                    <div style="font-size:0.75rem;color:#555;margin-top:0.4rem;">Live electricity spot prices require a commercial data subscription (ICE, EIA). This range is indicative only.</div>
                `;
            } else {
                refEl.innerHTML = `
                    ${baseMsg}
                    <div style="font-size:0.8rem;color:#555;">Live electricity prices require a commercial data feed (ICE, EIA RTO). Please verify current market price independently before proceeding.</div>
                `;
            }
        }

        /**
         * modalProceedBooking — called when the user confirms in the modal.
         * Reads the pending booking payload and posts it to /api/confirm-book.
         */
        async function modalProceedBooking() {
            const pending = window._pendingBooking;
            if (!pending) return;

            // Close modal and set loading state
            document.getElementById('confirmModal').style.display = 'none';
            const { cardId, payload } = pending;
            window._pendingBooking = null;

            const actionsDiv = document.getElementById(`${cardId}_actions`);
            const alertDiv   = document.getElementById(`${cardId}_alert`);
            const bypass = document.getElementById('bypassChaos').checked;

            actionsDiv.innerHTML = `
                <div style="width:100%; display:flex; align-items:center; justify-content:center; gap:0.5rem; color:var(--text-muted); font-size:0.85rem;">
                    <div class="spinner"></div> Booking trade...
                </div>
            `;

            try {
                const res = await fetch('/api/confirm-book', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ trade_data: payload, bypass_chaos: bypass })
                });

                const data = await res.json();

                if (!res.ok) {
                    let msg = data.detail || 'Booking failed.';
                    if (Array.isArray(data.detail)) {
                        msg = data.detail.map(d => `${d.loc.join('.')}: ${d.msg}`).join(', ');
                    }
                    throw new Error(msg);
                }

                // Success — replace actions with booked badge
                if (alertDiv) alertDiv.outerHTML = '';
                actionsDiv.outerHTML = `
                    <div class="status-badge-inline success">
                        Trade confirmed and persisted as <strong>ID #${data.trade_id}</strong>
                    </div>
                `;

                // Play synthesised audio success jingle
                playSuccessJingle();

                // Make all form inputs read-only
                const cardEl = document.getElementById(cardId);
                const inputs = cardEl.querySelectorAll('input, select');
                inputs.forEach(el => el.disabled = true);

                // Update session messages state
                updateCardStateInSession(cardId, 'booked', true);
                updateCardStateInSession(cardId, 'trade_id', data.trade_id);

                fetchAllTrades(); // Refresh ledger view cache and count
                checkConnection();

            } catch (e) {
                // Restore actions and show error
                if (alertDiv) {
                    alertDiv.outerHTML = `
                        <div class="card-alert error" id="${cardId}_alert">
                            <strong>Booking failed:</strong> ${e.message}
                        </div>
                    `;
                }
                actionsDiv.innerHTML = `
                    <button class="card-btn card-btn-success" onclick="bookTradeFromCard('${cardId}')">Retry Booking</button>
                    <button class="card-btn card-btn-secondary" onclick="cancelCard('${cardId}')">Cancel</button>
                `;
            }
        }

        function cancelCard(cardId) {
            updateCardStateInSession(cardId, 'cancelled', true);
            const cardEl = document.getElementById(cardId);
            const parent = cardEl.parentElement;
            parent.innerHTML = `<p style="color:var(--text-muted); font-style:italic;">Trade draft cancelled.</p>`;
        }

        // Live Log Polling
        let lastLogCount = 0;
        function startLogPolling() {
            setInterval(async () => {
                try {
                    const res = await fetch('/api/logs');
                    const logs = await res.json();
                    const panel = document.getElementById('logsPanel');
                    
                    if (logs.length !== lastLogCount) {
                        lastLogCount = logs.length;
                        panel.innerHTML = logs.map(l => {
                            let lvlClass = 'log-level-info';
                            if (l.level === 'WARNING') lvlClass = 'log-level-warn';
                            if (l.level === 'ERROR') lvlClass = 'log-level-err';
                            return `
                                <div class="log-line">
                                    <span class="log-time">[${l.time}]</span>
                                    <span class="${lvlClass}">[${l.level}]</span>
                                    <span>${escapeHTML(l.msg)}</span>
                                </div>
                            `;
                        }).join('');
                        panel.scrollTop = panel.scrollHeight;
                    }
                } catch (e) {}
            }, 1000);
        }

        function escapeHTML(str) {
            if (str === null || str === undefined) return '';
            return String(str)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
    </script>
</body>
</html>
"""
