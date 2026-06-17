# Power Trade AI Engineer Challenge

This repository is the backend fixture for an AI engineer coding challenge.

The goal for the candidate is to build components that take natural-language trade requests from sources like chat, email, or free-form text, translate them into structured power trades, and interact with the provided repository API.

## Repository purpose

This repo intentionally contains only the repository-side FastAPI service:
- a simple in-memory power trade store
- CRUD endpoints for trades
- optional evaluation fault injection behavior for resiliency testing

Candidate-owned implementation should live separately under `src`, outside `src/trade_repository`.

## Quick start

From the repository root:

1. Install dependencies.
2. Start the API.
3. Open Swagger UI.

```bash
uvicorn src.trade_repository.main:app --reload
```

Swagger UI:
- `http://localhost:8000/docs`

Health endpoint:
- `http://localhost:8000/health`

## Documentation

The detailed challenge materials are under `docs/`:

- [docs/CHALLENGE.md](docs/CHALLENGE.md): candidate brief, scope, expectations, and timebox
- [docs/API_SPEC.md](docs/API_SPEC.md): repository API contract and evaluation-fault behavior
- [docs/ACCEPTANCE_SCENARIOS.md](docs/ACCEPTANCE_SCENARIOS.md): example scenarios and expected candidate behavior

## Clarifications and contact

Incase you have any questions, feel free to send an email to lokesh.balu@axpo.com or jayanta.biswas@axpo.com

