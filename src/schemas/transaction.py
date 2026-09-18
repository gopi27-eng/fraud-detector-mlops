from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

# Valid transaction categories based on the PaySim dataset
TransactionType = Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


class TransactionPayload(BaseModel):
    """Data contract representing a single incoming transaction for inference."""

    # Reject any undeclared fields automatically
    model_config = ConfigDict(extra="forbid")

    step: int = Field(
        ...,
        ge=0,
        description="Maps a unit of time in the real world (1 step is 1 hour)",
    )
    type: TransactionType = Field(
        ...,
        description="Transaction category: PAYMENT, TRANSFER, CASH_OUT, DEBIT, or CASH_IN",
    )
    amount: float = Field(
        ...,
        gt=0.0,
        description="Amount of the transaction in local currency (must be positive)",
    )
    oldbalanceOrg: float = Field(
        ...,
        ge=0.0,
        description="Initial balance of origin account before transaction",
    )
    newbalanceOrig: float = Field(
        ...,
        ge=0.0,
        description="New balance of origin account after transaction",
    )
    oldbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="Initial balance of recipient account before transaction",
    )
    newbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="New balance of recipient account after transaction",
    )