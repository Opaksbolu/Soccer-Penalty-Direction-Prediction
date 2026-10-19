from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import src.training.train as training
from src.training.checkpoint import load_checkpoint


class TinyModel(nn.Module):
    """Fast replacement that exercises real optimizer, RNG and checkpoint code."""

    def __init__(self, pretrained=False, num_classes=3):
        super().__init__()
        self.encoder = nn.Identity()
        self.gru = nn.Linear(1, 4)
        self.head = nn.Linear(4, num_classes)

    def forward(self, frames):
        features = frames.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        return self.head(torch.relu(self.gru(features)))


def test_interrupted_training_matches_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MobileNetGRU", TinyModel)
    monkeypatch.setattr(training, "validation_partition", lambda rows, seed: (rows[:6], rows[6:]))
    monkeypatch.setattr(training, "_loader", lambda rows, batch_size, shuffle, augment, generator=None, class_order=None:
                        [(torch.ones(len(rows), 1, 1, 1, 1),
                          torch.tensor([("left", "center", "right").index(r["direction"]) for r in rows]))])
    rows = [{"kick_id": str(i), "direction": ("left", "center", "right")[i % 3]} for i in range(9)]
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    training.train_fold(rows, rows[:3], 0, 0.2, full, device_name="cpu", epochs=3, pretrained=False)
    training.train_fold(rows, rows[:3], 0, 0.2, resumed, device_name="cpu", epochs=1, pretrained=False)
    training.train_fold(rows, rows[:3], 0, 0.2, resumed, device_name="cpu", epochs=3,
                        pretrained=False, resume=True)
    a = load_checkpoint(full / "last.pt", torch.device("cpu"))
    b = load_checkpoint(resumed / "last.pt", torch.device("cpu"))
    assert a["epoch"] == b["epoch"] == 2
    assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert a["optimizer"]["state"] == b["optimizer"]["state"] if not a["optimizer"]["state"] else all(
        torch.equal(a["optimizer"]["state"][index]["exp_avg"], b["optimizer"]["state"][index]["exp_avg"])
        for index in a["optimizer"]["state"])


def test_final_training_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MobileNetGRU", TinyModel)
    monkeypatch.setattr(training, "_loader", lambda rows, batch_size, shuffle, augment, generator=None, class_order=None:
                        [(torch.ones(len(rows), 1, 1, 1, 1),
                          torch.tensor([class_order.index(r["direction"]) for r in rows]))])
    rows = [{"kick_id": str(i), "direction": ("left", "right")[i % 2]} for i in range(8)]
    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    training.train_final(rows, 0.2, complete, 3, "cpu", 4, 42, False, ("left", "right"))
    real_save = training.atomic_save

    def interrupt_after_epoch(payload, path):
        real_save(payload, path)
        if path.name == "final_0.2_last.pt" and payload["epoch"] == 0:
            raise KeyboardInterrupt

    monkeypatch.setattr(training, "atomic_save", interrupt_after_epoch)
    with pytest.raises(KeyboardInterrupt):
        training.train_final(rows, 0.2, interrupted, 3, "cpu", 4, 42, False, ("left", "right"))
    monkeypatch.setattr(training, "atomic_save", real_save)
    training.train_final(rows, 0.2, interrupted, 3, "cpu", 4, 42, False,
                         ("left", "right"), resume=True)
    a = load_checkpoint(complete / "final_0.2.pt", torch.device("cpu"))
    b = load_checkpoint(interrupted / "final_0.2.pt", torch.device("cpu"))
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
