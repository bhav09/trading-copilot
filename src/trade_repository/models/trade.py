from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(str, Enum):
    DRAFT = "DRAFT"
    BOOKED = "BOOKED"


class TradeBase(BaseModel):
    direction: Direction
    quantity_mw: float = Field(gt=0, le=10000)
    price_per_mwh: float = Field(gt=0, le=1000)
    counterparty: str = Field(min_length=1, max_length=120)
    delivery_start: datetime
    delivery_end: datetime
    hub: str = Field(min_length=1, max_length=40)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    notes: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_delivery_window(self) -> "TradeBase":
        if self.delivery_end <= self.delivery_start:
            raise ValueError("delivery_end must be after delivery_start")
        return self


class TradeCreate(TradeBase):
    pass


class TradeUpdate(BaseModel):
    direction: Optional[Direction] = None
    quantity_mw: Optional[float] = Field(default=None, gt=0, le=10000)
    price_per_mwh: Optional[float] = Field(default=None, gt=0, le=1000)
    counterparty: Optional[str] = Field(default=None, min_length=1, max_length=120)
    delivery_start: Optional[datetime] = None
    delivery_end: Optional[datetime] = None
    hub: Optional[str] = Field(default=None, min_length=1, max_length=40)
    currency: Optional[str] = Field(default=None, min_length=3, max_length=3)
    notes: Optional[str] = Field(default=None, max_length=500)


class Trade(TradeBase):
    trade_id: int = Field(gt=0)
    status: TradeStatus = TradeStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
