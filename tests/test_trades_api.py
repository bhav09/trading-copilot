from fastapi.testclient import TestClient
from src.trade_repository.main import app

client = TestClient(app, headers={"x-eval-bypass-chaos": "true"})


def test_create_and_get_trade_flow() -> None:
    create_response = client.post(
        "/api/trades",
        json={
            "direction": "BUY",
            "quantity_mw": 100,
            "price_per_mwh": 45.5,
            "counterparty": "Shell",
            "delivery_start": "2026-06-18T00:00:00Z",
            "delivery_end": "2026-06-19T00:00:00Z",
            "hub": "MISO",
        },
    )

    assert create_response.status_code == 201
    trade_id = create_response.json()["trade_id"]

    get_response = client.get(f"/api/trades/{trade_id}")
    assert get_response.status_code == 200
    assert get_response.json()["trade_id"] == trade_id


def test_update_trade() -> None:
    create_response = client.post(
        "/api/trades",
        json={
            "direction": "SELL",
            "quantity_mw": 50,
            "price_per_mwh": 52,
            "counterparty": "BP",
            "delivery_start": "2026-06-18T00:00:00Z",
            "delivery_end": "2026-06-19T00:00:00Z",
            "hub": "PJM",
        },
    )
    trade_id = create_response.json()["trade_id"]

    update_response = client.put(f"/api/trades/{trade_id}", json={"price_per_mwh": 60})
    assert update_response.status_code == 200
    assert update_response.json()["price_per_mwh"] == 60


def test_delete_trade() -> None:
    create_response = client.post(
        "/api/trades",
        json={
            "direction": "BUY",
            "quantity_mw": 40,
            "price_per_mwh": 48,
            "counterparty": "Shell",
            "delivery_start": "2026-06-18T00:00:00Z",
            "delivery_end": "2026-06-19T00:00:00Z",
            "hub": "MISO",
        },
    )
    trade_id = create_response.json()["trade_id"]

    delete_response = client.delete(f"/api/trades/{trade_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/api/trades/{trade_id}")
    assert get_response.status_code == 404
