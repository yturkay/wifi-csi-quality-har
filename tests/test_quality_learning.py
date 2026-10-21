import torch

from wifi_csi_quality.quality_learning import GatedQualityCNNLSTM, QualityAwareCNNLSTM


def test_quality_aware_model_returns_four_logits():
    model = QualityAwareCNNLSTM()
    values = torch.randn(2, 500, 64)
    quality = torch.randn(2, 500, 4)

    with torch.no_grad():
        logits = model(values, quality)

    assert logits.shape == (2, 4)


def test_gated_quality_model_returns_four_logits():
    model = GatedQualityCNNLSTM()
    values = torch.randn(2, 500, 64)
    quality = torch.randn(2, 500, 4)

    with torch.no_grad():
        logits = model(values, quality)

    assert logits.shape == (2, 4)
