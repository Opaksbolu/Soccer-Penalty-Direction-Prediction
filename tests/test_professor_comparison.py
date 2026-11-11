"""Fast integration checks; actual video training is run on the user's Mac."""
import json
from pathlib import Path

import pytest

from src.data.common import HORIZONS, write_jsonl
from src.compare_single_frame import run


def test_paired_comparison_resumes_without_retraining(tmp_path, monkeypatch):
    manifest = []
    clips = []
    for i in range(20):
        label = "left" if i % 2 == 0 else "right"
        record = {"kick_id": f"kick{i}", "direction": label,
                  "player_id": f"player{i}", "match_id": f"match{i}"}
        manifest.append(record)
        for h in HORIZONS:
            clips.append({**record, "horizon": h, "clip": f"unused_{i}_{h}.mp4"})
    mp, cp = tmp_path / "manifest.jsonl", tmp_path / "clips.jsonl"
    write_jsonl(mp, manifest)
    write_jsonl(cp, clips)
    calls = []

    def baseline(name, train, test, manifest, class_order):
        return [class_order.index(row["direction"]) for row in test]

    def fake_train(train_rows, test_rows, fold, horizon, output, **kwargs):
        calls.append((fold, horizon, kwargs["architecture"]))
        preds = [{"kick_id": row["kick_id"], "truth": int(row["direction"] == "right"),
                  "prediction": int(row["direction"] == "right")} for row in test_rows]
        result = {"predictions": preds, "prediction_latency_seconds": .01}
        output.mkdir(parents=True, exist_ok=True)
        (output / "test_metrics.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("src.compare_single_frame._baseline_predictions", baseline)
    monkeypatch.setattr("src.compare_single_frame.train_fold", fake_train)
    output = tmp_path / "out"
    kwargs = dict(device="cpu", epochs=1, batch_size=2, pretrained=False)
    result = run(cp, mp, output, **kwargs)
    assert result["complete_requested_run"]
    assert len(calls) == 30
    assert result["summary"]["0.2_gru"]["accuracy"] == 1
    assert (output / "comparison.csv").exists()
    assert (output / "macro_f1_comparison.png").exists()
    again = run(cp, mp, output, resume=True, **kwargs)
    assert again["complete_requested_run"]
    assert len(calls) == 30
    with pytest.raises(FileExistsError):
        run(cp, mp, output, **kwargs)
    with pytest.raises(ValueError, match="different experiment"):
        run(cp, mp, output, resume=True, seed=9, **{k:v for k,v in kwargs.items() if k != "seed"})
