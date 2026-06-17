# Candidate Challenge: NL to Power Trade Booking

## Objective

Build a candidate-owned component layer that transforms natural-language user requests (chat conversation, email text, or free-form query) into power trades and persists them through the provided repository CRUD API routes.

## What is provided

- A running repository API with fixed trade CRUD routes only.
- A simplified power trade model with validation.
- In-memory storage.
- Repository API source code located under `src/trade_repository`.
- Fault injection behavior for evaluation (intermittent `503`, delay, and `504` across API routes).

## What the candidate must build

Implement candidate-owned components under `src` 

1. Input adapters for (pick any one):
- chat-style text
- email-style text
- generic free-form text

2. A translation/orchestration component that:
- extracts trade intent and fields,
- handles ambiguity and confidence,
- maps parsed output into repository API payloads,
- calls CRUD API routes in the correct sequence.

3. Error handling and recovery logic that:
- requests clarification when required fields are missing,
- avoids creating/updating trades when confidence is too low,
- should validate with the user before proceeding with trade creation,
- handles API validation and conflict errors gracefully.

4. AI Usage
- you are free to use AI inorder to assist you in the scafolding etc.

## Trade model (required fields)

- direction: BUY | SELL
- quantity_mw: > 0
- price_per_mwh: > 0
- counterparty: non-empty
- delivery_start: datetime
- delivery_end: datetime and greater than delivery_start
- hub: non-empty
- status: DRAFT or BOOKED

## API workflow expectation

Typical happy path:
1. Candidate component receives user text.
2. Candidate component converts text into structured payload.
3. Candidate component calls `POST /api/trades`.
4. Candidate component calls `GET /api/trades/{trade_id}` or `PUT /api/trades/{trade_id}` as needed.

## Non-goals

- Do not modify repository API internals for candidate submissions.
- No production-grade auth, deployment, or external market data required.
- No persistent database required.
- Do not aim for production ready setup, rather minimal working app with the requirements satisfied
- FAST API is AI generated minimalistic web API for the challenge purpose

## Timebox

Target completion time: 4 to 8 hours, ideally.
