"""Focused checks for the professor-requested single-frame baseline."""
import torch

from src.models.single_frame import MobileNetSingleFrame


def test_single_frame_uses_only_last_frame():
    model = MobileNetSingleFrame(pretrained=False, num_classes=2).eval()
    x = torch.randn(1, 16, 3, 64, 64)
    with torch.no_grad():
        original = model(x)
        changed = x.clone()
        changed[:, :-1] = torch.randn_like(changed[:, :-1])
        assert torch.equal(original, model(changed))
        assert original.shape == (1, 2)


def test_single_frame_freezes_backbone_and_trains_head():
    model = MobileNetSingleFrame(pretrained=False, num_classes=2)
    model.train()
    assert not model.encoder.training
    assert model.head.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.head.parameters())
