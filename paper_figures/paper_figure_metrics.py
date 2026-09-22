"""Load/save paper figure metrics produced by figures-creation.ipynb."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GROUPS = ["left", "middle", "right", "overall"]
CLASS_KEYS = [0, 1, 2]
CLASS_JSON_KEYS = ["clear", "person", "other"]


def _class_dict_to_json(class_dict: dict[int, float]) -> dict[str, float]:
    return {CLASS_JSON_KEYS[k]: float(class_dict[k]) for k in CLASS_KEYS}


def _class_dict_from_json(class_dict: dict[str, float]) -> dict[int, float]:
    return {i: float(class_dict[CLASS_JSON_KEYS[i]]) for i in CLASS_KEYS}


def serialize_models_data_by_class(models_data_by_class: dict) -> dict[str, Any]:
    return {
        model_name: {
            direction: _class_dict_to_json(by_dir[direction])
            for direction in GROUPS
        }
        for model_name, by_dir in models_data_by_class.items()
    }


def deserialize_models_data_by_class(data: dict[str, Any]) -> dict:
    return {
        model_name: {
            direction: _class_dict_from_json(by_dir[direction])
            for direction in GROUPS
        }
        for model_name, by_dir in data.items()
    }


def export_figure_metrics(
    path: Path | str,
    models_data_by_class: dict,
    noobj_f1: list[float],
    person_f1: list[float],
    other_f1: list[float],
    fig1_caption: str,
    fig2_caption: str,
    source: str = "figures-creation.ipynb",
) -> Path:
    """Write metrics JSON after the notebook evaluation cells finish."""
    path = Path(path)
    payload = {
        "source": source,
        "models_data_by_class": serialize_models_data_by_class(models_data_by_class),
        "fig2": {
            "clear_f1": [float(v) for v in noobj_f1],
            "person_f1": [float(v) for v in person_f1],
            "other_f1": [float(v) for v in other_f1],
        },
        "captions": {"fig1": fig1_caption, "fig2": fig2_caption},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_figure_metrics(path: Path | str) -> dict[str, Any]:
    """Load metrics JSON exported by figures-creation.ipynb."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run figures-creation.ipynb through Section 13 on Kaggle "
            "(or locally with models/data) to export figure_metrics.json."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "models_data_by_class": deserialize_models_data_by_class(
            payload["models_data_by_class"]
        ),
        "clear_f1": payload["fig2"]["clear_f1"],
        "person_f1": payload["fig2"]["person_f1"],
        "other_f1": payload["fig2"]["other_f1"],
        "fig1_caption": payload["captions"]["fig1"],
        "fig2_caption": payload["captions"]["fig2"],
        "source": payload.get("source", "unknown"),
    }
