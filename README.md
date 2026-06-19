# Axso Power Trade AI Copilot

> **Candidate Implementation** — Axso AI Engineer Coding Challenge

A natural-language power trade orchestration platform. Translates free-form chat or email-style trade requests into structured, validated power trade bookings — with a real-time web dashboard, Gemini-powered NLU, and full resilience testing support.

---

## Architecture

```
Browser (Dashboard)
    │  WebSocket-free SSE polling
    ▼
FastAPI (src/main.py)        ← Candidate's server (port 8001)
    │  Rate-limited API, Gemini AI, guardrails
    ▼
Trade Repository (src/trade_repository/)  ← Provided fixture (port 8000)
    │  In-memory store, optional chaos/fault injection
```

### Key Modules (candidate-owned)

| File | Purpose |
|---|---|
| `src/main.py` | FastAPI server + full HTML/CSS/JS web dashboard (single-file SSR) |
| `src/translator.py` | Gemini 2.0 Flash powered NLU — classifies intent & parses trade fields |
| `src/client.py` | `TradeRepositoryClient` — HTTP client with retries, timeouts, chaos bypass |
| `tests/test_candidate_logic.py` | Unit tests for translator, client, guardrails |
| `tests/test_trades_api.py` | Integration tests for booking, parsing, ledger endpoints |

---

## Features

- **Natural Language → Trade** — Gemini 2.0 Flash parses `"buy 50 MW from Axpo at $42 delivery Jan 2026"` into a structured trade card
- **Client-side Fast Router** — Zero-latency intent classifier handles greetings/off-topic before hitting the API
- **Market Research** — Web-search fallback for price queries (`"what is AAPL price?"`)
- **Confirmation Modal** — Pre-booking review screen with auto-populated reference price
- **Chaos Mode Toggle** — Bypass or enable fault injection in the repository for resilience testing
- **Live System Logs** — In-memory rolling log buffer streamed to the dashboard terminal panel
- **Trade Ledger** — Scrollable table of all booked trades with status badges
- **Chat Session History** — localStorage-backed session management with sidebar navigation
- **Rate Limiting** — 20 req/min per IP on parse and book endpoints (slowapi)
- **Guardrails** — Off-topic queries blocked gracefully; no hallucinated trades

---

## Quick Start

### Prerequisites

- Python 3.11+
- A `GEMINI_API_KEY` environment variable (Google AI Studio)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Trade Repository (port 8000)

```bash
uvicorn src.trade_repository.main:app --reload --port 8000
```

### 3. Start the AI Copilot Dashboard (port 8001)

```bash
uvicorn src.main:app --reload --port 8001
```

### 4. Open the Dashboard

- **Dashboard**: http://localhost:8001/
- **Repository Swagger**: http://localhost:8000/docs
- **Copilot Swagger**: http://localhost:8001/docs

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 8 tests cover: translator parsing, client retries, guardrails, booking validation, ledger, and API integration.

---

## Example Queries

```
"Buy 100 MW from Vattenfall at €45/MWh, delivery from 2026-01-01 to 2026-03-31, hub DE-LU"
"Sell 250 MW to RWE at €38.50, Jan-Mar 2026 at TTF hub"
"What is the current price of Tesla stock?"
"Show me all booked trades"
```

---

## Documentation

- [docs/CHALLENGE.md](docs/CHALLENGE.md) — Challenge brief and scope
- [docs/API_SPEC.md](docs/API_SPEC.md) — Repository API contract
- [docs/ACCEPTANCE_SCENARIOS.md](docs/ACCEPTANCE_SCENARIOS.md) — Acceptance scenarios

---

## Contact

Questions? [lokesh.balu@axpo.com](mailto:lokesh.balu@axpo.com) | [jayanta.biswas@axpo.com](mailto:jayanta.biswas@axpo.com)
