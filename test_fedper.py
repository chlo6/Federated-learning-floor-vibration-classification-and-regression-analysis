from __future__ import annotations

import numpy as np
import torch

from src.redo_by_sara.federated import (
    get_base_parameters,
    get_head_state,
    set_base_parameters,
    set_head_state,
)
from src.redo_by_sara.models import SimpleCNN1D


def test_setting_fedper_base_does_not_replace_personal_head() -> None:
    source = SimpleCNN1D(in_channels=3, output_dim=4)
    client = SimpleCNN1D(in_channels=3, output_dim=4)
    head_before = get_head_state(client)

    set_base_parameters(client, get_base_parameters(source))

    for actual, expected in zip(
        get_base_parameters(client), get_base_parameters(source), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for key, value in client.head.state_dict().items():
        torch.testing.assert_close(value, head_before[key])


def test_setting_fedper_head_does_not_replace_shared_base() -> None:
    source = SimpleCNN1D(in_channels=3, output_dim=4)
    client = SimpleCNN1D(in_channels=3, output_dim=4)
    base_before = [value.copy() for value in get_base_parameters(client)]

    set_head_state(client, get_head_state(source))

    for actual, expected in zip(
        get_base_parameters(client), base_before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for key, value in client.head.state_dict().items():
        torch.testing.assert_close(value, source.head.state_dict()[key])
