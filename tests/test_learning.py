import torch

from wifi_csi_quality.learning import (
    CLASS_NAMES,
    confusion_matrix,
    create_model,
    metrics_from_outputs,
)


def test_all_models_return_four_logits():
    values = torch.randn(2, 500, 64)
    for name in ("mlp", "cnn1d", "cnn_lstm"):
        model = create_model(name)
        with torch.no_grad():
            logits = model(values)
        assert logits.shape == (2, len(CLASS_NAMES))


def test_metrics_are_perfect_for_perfect_predictions():
    labels = torch.tensor([0, 1, 2, 3])
    logits = torch.eye(4) * 10

    result = metrics_from_outputs(logits, labels)

    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["confusion_matrix"] == torch.eye(4, dtype=torch.int64).tolist()


def test_confusion_matrix_orientation_is_true_by_predicted():
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])

    matrix = confusion_matrix(labels, predictions)

    assert matrix[0, 0] == 1
    assert matrix[0, 1] == 1
    assert matrix[1, 1] == 2
