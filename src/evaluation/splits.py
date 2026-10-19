"""Persistent, shared stratified folds with optional player/match grouping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def group_keys(records: list[dict]) -> list[str]:
    """Join kicks sharing either a player or a match, including transitive links."""
    parent = list(range(len(records)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for field in ("player_id", "match_id"):
            value = record.get(field)
            if value is None or str(value).strip() == "":
                continue
            key = (field, str(value))
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    return [f"component:{root(i)}" for i in range(len(records))]


def make_folds(records: list[dict], output: Path, seed: int = 42, folds: int = 5) -> list[dict]:
    ids = [row["kick_id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("one record per kick is required")
    labels = np.asarray([row["direction"] for row in records])
    groups = np.asarray(group_keys(records))
    if min(np.unique(labels, return_counts=True)[1]) < folds:
        raise ValueError(f"Each direction needs at least {folds} kicks for {folds}-fold CV")
    if len(set(groups)) < folds:
        raise ValueError("not enough independent groups for cross-validation")
    has_shared_group = len(set(groups)) < len(groups)
    splitter = (StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
                if has_shared_group else StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed))
    iterator = splitter.split(ids, labels, groups) if has_shared_group else splitter.split(ids, labels)
    result = []
    for number, (train, test) in enumerate(iterator):
        train_groups = {groups[i] for i in train}
        test_groups = {groups[i] for i in test}
        if train_groups & test_groups:
            raise AssertionError("group leakage detected")
        result.append({"fold": number, "train": [ids[i] for i in train],
                       "test": [ids[i] for i in test]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seed": seed, "group_policy": "connected player/match groups; kick otherwise",
                                  "folds": result}, indent=2), encoding="utf-8")
    return result
