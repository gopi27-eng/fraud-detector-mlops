import pytest
from pydantic import ValidationError
from src.schemas.transaction import TransactionPayload


@pytest.fixture
def valid_transaction_dict() -> dict:
    """Provides a baseline valid payload for test cases."""
    return {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


def test_valid_payload_instantiation(valid_transaction_dict: dict):
    """Assert valid inputs pass schema validation without alteration."""
    tx = TransactionPayload(**valid_transaction_dict)
    assert tx.step == 1
    assert tx.type == "PAYMENT"
    assert tx.amount == 9839.64
    assert tx.oldbalanceOrg == 170136.0


def test_invalid_type_raises_validation_error(valid_transaction_dict: dict):
    """Assert unrecognized transaction type strings are rejected."""
    bad_payload = valid_transaction_dict.copy()
    bad_payload["type"] = "CRYPTO_SWAP"

    with pytest.raises(ValidationError) as exc_info:
        TransactionPayload(**bad_payload)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("type",) for err in errors)


def test_non_positive_amount_raises_validation_error(valid_transaction_dict: dict):
    """Assert transactions with amount <= 0 fail."""
    bad_payload = valid_transaction_dict.copy()
    bad_payload["amount"] = 0.0

    with pytest.raises(ValidationError) as exc_info:
        TransactionPayload(**bad_payload)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("amount",) for err in errors)


def test_negative_balances_raise_validation_error(valid_transaction_dict: dict):
    """Assert origin/dest balances cannot be negative."""
    bad_payload = valid_transaction_dict.copy()
    bad_payload["oldbalanceOrg"] = -50.0

    with pytest.raises(ValidationError) as exc_info:
        TransactionPayload(**bad_payload)

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("oldbalanceOrg",) for err in errors)


def test_extra_forbidden_fields_raise_validation_error(valid_transaction_dict: dict):
    """Assert extra arbitrary keys are rejected per model_config."""
    bad_payload = valid_transaction_dict.copy()
    bad_payload["untracked_attribute"] = "exploit"

    with pytest.raises(ValidationError) as exc_info:
        TransactionPayload(**bad_payload)

    errors = exc_info.value.errors()
    assert any("extra_forbidden" in err["type"] for err in errors)