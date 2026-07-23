from __future__ import annotations

import pytest
import torch

from src.redo_by_sara.training import create_model


def _artifact() -> dict[str, object]:
    return {
        "samples": torch.randn(4, 20, 256),
        "subject_to_class": {
            "001": 0,
            "002": 1,
            "003": 2,
            "004": 3,
            "005": 4,
            "007": 5,
            "008": 6,
        },
    }


@pytest.mark.parametrize("model_variant", ["simple", "deep"])
def test_centralized_classification_has_seven_outputs(
    model_variant: str,
) -> None:
    model = create_model(
        _artifact(),
        task="classification",
        model_variant=model_variant,
    )

    assert model(torch.randn(2, 20, 256)).shape == (2, 7)


@pytest.mark.parametrize("model_variant", ["simple", "deep"])
def test_centralized_regression_has_one_output(model_variant: str) -> None:
    model = create_model(
        _artifact(),
        task="regression",
        model_variant=model_variant,
    )

    assert model(torch.randn(2, 20, 256)).shape == (2, 1)


def test_unknown_centralized_model_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="model_variant"):
        create_model(_artifact(), task="classification", model_variant="unknown")
