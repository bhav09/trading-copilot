from __future__ import annotations

import asyncio
from threading import Lock
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from src.trade_repository.models.trade import (
    Trade,
    TradeCreate,
    TradeStatus,
    TradeUpdate,
)

app = FastAPI(
    title="Power Trade Repository API",
    version="0.1.0",
    description="Repository-only API for power trade CRUD and booking workflows",
)

# In-memory trade store for challenge speed.
trade_store: dict[int, Trade] = {}
next_trade_id = 1

# Fault-injection controls for evaluation environments.
request_counter = 0
counter_lock = Lock()
TRANSIENT_ERROR_EVERY = 5
DELAY_EVERY = 7
TIMEOUT_EVERY = 11
DELAY_SECONDS = 1.25
TIMEOUT_SECONDS = 2.0
BYPASS_HEADER = "x-eval-bypass-chaos"


@app.middleware("http")
async def inject_faults_for_eval(request: Request, call_next):
    if request.headers.get(BYPASS_HEADER, "").lower() == "true":
        return await call_next(request)

    global request_counter
    with counter_lock:
        request_counter += 1
        current_count = request_counter

    if current_count % TIMEOUT_EVERY == 0:
        await asyncio.sleep(TIMEOUT_SECONDS)
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"detail": "Simulated timeout for evaluation"},
        )

    if current_count % TRANSIENT_ERROR_EVERY == 0:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "1"},
            content={"detail": "Simulated transient failure for evaluation"},
        )

    if current_count % DELAY_EVERY == 0:
        await asyncio.sleep(DELAY_SECONDS)

    return await call_next(request)


def _get_trade_or_404(trade_id: int) -> Trade:
    trade = trade_store.get(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    return trade


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/trades", response_model=Trade, status_code=status.HTTP_201_CREATED)
def create_trade(payload: TradeCreate) -> Trade:
    global next_trade_id
    trade = Trade.model_validate({**payload.model_dump(), "trade_id": next_trade_id})
    trade_store[trade.trade_id] = trade
    next_trade_id += 1
    return trade


@app.get("/api/trades", response_model=list[Trade])
def list_trades(
    status_filter: Optional[TradeStatus] = Query(default=None, alias="status"),
    counterparty: Optional[str] = None,
    hub: Optional[str] = None,
) -> list[Trade]:
    trades = list(trade_store.values())
    if status_filter is not None:
        trades = [t for t in trades if t.status == status_filter]
    if counterparty is not None:
        trades = [t for t in trades if t.counterparty.lower() == counterparty.lower()]
    if hub is not None:
        trades = [t for t in trades if t.hub.lower() == hub.lower()]
    return trades


@app.get("/api/trades/{trade_id}", response_model=Trade)
def get_trade(trade_id: int) -> Trade:
    return _get_trade_or_404(trade_id)


@app.put("/api/trades/{trade_id}", response_model=Trade)
def update_trade(trade_id: int, payload: TradeUpdate) -> Trade:
    current = _get_trade_or_404(trade_id)

    data = current.model_dump()
    data.update(payload.model_dump(exclude_unset=True))
    data["updated_at"] = datetime.utcnow()
    updated = Trade.model_validate(data)
    trade_store[trade_id] = updated
    return updated


@app.delete("/api/trades/{trade_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_trade(trade_id: int) -> Response:
    _get_trade_or_404(trade_id)

    del trade_store[trade_id]
    return Response(status_code=status.HTTP_204_NO_CONTENT)
