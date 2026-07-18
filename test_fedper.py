from __future__ import annotations

import numpy as np
import torch

from src.redo_by_sara.federated import (
    IndexedArtifactDataset,
    client_class_ids,
    client_local_label_map,
    create_federated_model,
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


def _classification_artifact() -> dict[str, object]:
    return {
        "samples": torch.randn(3, 2, 256),
        "channel_mean": torch.zeros(1, 2, 1),
        "channel_std": torch.ones(1, 2, 1),
        "classification_targets": torch.tensor([0, 4, 6]),
        "subject_to_class": {
            "001": 0,
            "005": 4,
            "008": 6,
        },
    }


def test_client_dataset_remaps_global_classes_to_local_targets() -> None:
    artifact = _classification_artifact()
    class_ids = client_class_ids(artifact, ["001", "005", "008"])
    dataset = IndexedArtifactDataset(
        artifact=artifact,
        indices=[0, 1, 2],
        task="classification",
        class_ids=class_ids,
    )

    assert class_ids == [0, 4, 6]
    assert [int(dataset[index][1]) for index in range(3)] == [0, 1, 2]
    assert client_local_label_map(
        artifact,
        ["008", "001", "005"],
    ) == {
        "001": 0,
        "005": 1,
        "008": 2,
    }


def test_client_model_has_only_local_class_logits() -> None:
    artifact = _classification_artifact()
    model = create_federated_model(
        artifact,
        "classification",
        output_dim=3,
    )

    logits = model(torch.randn(2, 2, 256))

    assert logits.shape == (2, 3)
