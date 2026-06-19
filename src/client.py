"""
Repository API Client for Axso Power Trade Backend.
Implements HTTP requests to the backend store with transient failure retries.
"""

from __future__ import annotations
import logging
import time
import random
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("trade_client")

class TradeRepositoryClient:
    """
    Client for interacting with the backend Power Trade Repository API.
    Handles communication, headers, and resilient retry logic for 503 and 504 faults.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", bypass_chaos: bool = False):
        self.base_url = base_url
        self.bypass_chaos = bypass_chaos
        # Default timeouts: 5.0 seconds for general requests
        self.timeout = httpx.Timeout(5.0)

    def _get_headers(self) -> Dict[str, str]:
        """Generate headers, optionally adding the evaluation chaos bypass."""
        headers = {"Content-Type": "application/json"}
        if self.bypass_chaos:
            headers["x-eval-bypass-chaos"] = "true"
        return headers

    def _request_with_retry(
        self, 
        method: str, 
        path: str, 
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff_base: float = 0.5
    ) -> httpx.Response:
        """
        Executes an HTTP request with exponential backoff and jitter for 503 and 504 faults.
        """
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = self._get_headers()
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Sending {method} request to {url} (Attempt {attempt}/{max_retries})")
                
                with httpx.Client(timeout=self.timeout) as client:
                    if method.upper() == "GET":
                        response = client.get(url, headers=headers, params=params)
                    elif method.upper() == "POST":
                        response = client.post(url, headers=headers, json=json_data)
                    elif method.upper() == "PUT":
                        response = client.put(url, headers=headers, json=json_data)
                    elif method.upper() == "DELETE":
                        response = client.delete(url, headers=headers)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                # If successful or a non-chaos client error (like 400, 404, 422), return immediately
                if response.status_code not in (status_503 := 503, status_504 := 504):
                    return response

                # Check if it is a transient 503 or 504 chaos error
                logger.warning(
                    f"Received transient error {response.status_code} from {url} (Attempt {attempt})"
                )
                
                if attempt == max_retries:
                    logger.error(f"Max retries reached. Returning transient error response.")
                    return response

                # Handle Retry-After header if present for 503
                sleep_time = 0.0
                if response.status_code == 503:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_time = float(retry_after)
                
                # If no sleep_time set or not a 503, calculate exponential backoff with jitter
                if sleep_time <= 0:
                    sleep_time = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.3)

                logger.info(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                logger.warning(f"Network error on {url} (Attempt {attempt}): {exc}")
                if attempt == max_retries:
                    raise exc
                
                # Calculate backoff for timeouts/network errors
                sleep_time = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.3)
                logger.info(f"Sleeping for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
                
        raise httpx.ConnectError("Max retries exceeded without getting a response")

    def health(self, timeout: float = 1.5, max_retries: int = 1) -> Dict[str, str]:
        """Check the API liveness status."""
        old_timeout = self.timeout
        self.timeout = httpx.Timeout(timeout)
        try:
            response = self._request_with_retry("GET", "/health", max_retries=max_retries)
            if response.status_code == 200:
                return response.json()
            response.raise_for_status()
            return {"status": "error"}
        finally:
            self.timeout = old_timeout

    def create_trade(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new draft trade."""
        response = self._request_with_retry("POST", "/api/trades", json_data=payload)
        response.raise_for_status()
        return response.json()

    def get_trade(self, trade_id: int) -> Dict[str, Any]:
        """Retrieve a specific trade by ID."""
        response = self._request_with_retry("GET", f"/api/trades/{trade_id}")
        response.raise_for_status()
        return response.json()

    def list_trades(self, status: Optional[str] = None, counterparty: Optional[str] = None, hub: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all trades with optional filters."""
        params = {}
        if status:
            params["status"] = status
        if counterparty:
            params["counterparty"] = counterparty
        if hub:
            params["hub"] = hub
            
        response = self._request_with_retry("GET", "/api/trades", params=params)
        response.raise_for_status()
        return response.json()

    def update_trade(self, trade_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update a trade's details (valid only for DRAFT status)."""
        response = self._request_with_retry("PUT", f"/api/trades/{trade_id}", json_data=payload)
        response.raise_for_status()
        return response.json()

    def delete_trade(self, trade_id: int) -> bool:
        """Delete a trade."""
        response = self._request_with_retry("DELETE", f"/api/trades/{trade_id}")
        if response.status_code == 204:
            return True
        response.raise_for_status()
        return False
