import json
from pathlib import Path

import pytest

from sane.data.splitter import ModelWidthGroupSplitter


class SyntheticCheckpointDataset:
    def __init__(self, model_ids: list[str], checkpoints_per_model: int = 2) -> None:
        self._model_ids = model_ids
        self._checkpoint_model_ids = [
            model_id
            for model_id in model_ids
            for _ in range(checkpoints_per_model)
        ]

    def __len__(self) -> int:
        return len(self._checkpoint_model_ids)

    def get_model_id(self, index: int) -> str:
        return self._checkpoint_model_ids[index]


def _write_model_dir(root: Path, name: str, params: dict) -> str:
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")
    return str(model_dir)


def _split_model_ids(dataset: SyntheticCheckpointDataset, splits) -> list[set[str]]:
    return [
        {dataset.get_model_id(index) for index in subset.indices}
        for subset in splits
    ]


def _widths_for_models(model_ids: set[str]) -> set[str]:
    widths = set()
    for model_id in model_ids:
        params = json.loads((Path(model_id) / "params.json").read_text(encoding="utf-8"))
        widths.add(str(params["model_width"]))
    return widths


def test_model_width_group_splitter_keeps_models_together_and_covers_widths_in_train(tmp_path: Path):
    model_ids = [
        _write_model_dir(tmp_path, "width16_seed1", {"model_width": 16, "seed": 1}),
        _write_model_dir(tmp_path, "width16_seed2", {"model_width": 16, "seed": 2}),
        _write_model_dir(tmp_path, "width32_seed1", {"model_width": 32, "seed": 1}),
        _write_model_dir(tmp_path, "width32_seed2", {"model_width": 32, "seed": 2}),
        _write_model_dir(tmp_path, "width64_seed1", {"model_width": 64, "seed": 1}),
        _write_model_dir(tmp_path, "width64_seed2", {"model_width": 64, "seed": 2}),
    ]
    dataset = SyntheticCheckpointDataset(model_ids, checkpoints_per_model=3)

    splits = ModelWidthGroupSplitter(seed=123).split(dataset, [0.5, 0.25, 0.25])
    split_model_ids = _split_model_ids(dataset, splits)

    assert _widths_for_models(split_model_ids[0]) == {"16", "32", "64"}

    assigned = {}
    for split_idx, subset in enumerate(splits):
        for index in subset.indices:
            model_id = dataset.get_model_id(index)
            if model_id in assigned:
                assert assigned[model_id] == split_idx
            else:
                assigned[model_id] = split_idx


def test_model_width_group_splitter_assigns_singleton_width_to_train(tmp_path: Path):
    model_ids = [
        _write_model_dir(tmp_path, "width16_seed1", {"model_width": 16, "seed": 1}),
        _write_model_dir(tmp_path, "width32_seed1", {"model_width": 32, "seed": 1}),
        _write_model_dir(tmp_path, "width32_seed2", {"model_width": 32, "seed": 2}),
    ]
    dataset = SyntheticCheckpointDataset(model_ids, checkpoints_per_model=2)

    splits = ModelWidthGroupSplitter(seed=7).split(dataset, [0.7, 0.15, 0.15])
    train_model_ids = _split_model_ids(dataset, splits)[0]

    assert _widths_for_models(train_model_ids) == {"16", "32"}


def test_model_width_group_splitter_raises_when_width_cannot_be_inferred(tmp_path: Path):
    model_id = _write_model_dir(tmp_path, "model_without_width", {"seed": 1, "lr": 0.1})
    dataset = SyntheticCheckpointDataset([model_id], checkpoints_per_model=1)

    with pytest.raises(ValueError, match="Could not determine model width"):
        ModelWidthGroupSplitter(seed=0).split(dataset, [1.0])
