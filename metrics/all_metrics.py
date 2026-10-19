#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_results(path: Path) -> Dict[str, List[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        return {name: trials for name, trials in data.items() if isinstance(trials, list)}

    if isinstance(data, list):
        merged: Dict[str, List[dict]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            for name, trials in item.items():
                if not isinstance(trials, list):
                    continue
                merged.setdefault(name, []).extend(trials)
        return merged

    raise ValueError("Unsupported JSON format. Expected a dict or a list of dicts.")


def collect_metrics(
    trials: Iterable[dict], success_only: bool = False
) -> Tuple[List[float], List[float]]:
    steps_list: List[float] = []
    time_list: List[float] = []

    for trial in trials:
        if not isinstance(trial, dict):
            continue
        if success_only and trial.get("success") is not True:
            continue
        if "steps" not in trial or "time" not in trial:
            continue

        try:
            steps_list.append(float(trial["steps"]))
            time_list.append(float(trial["time"]))
        except (TypeError, ValueError):
            continue

    return steps_list, time_list


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate average steps and time from all_results.json."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="all_results.json",
        help="Path to the JSON results file. Default: all_results.json",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only include trials where success is true.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    results = load_results(json_path)

    summary = {"objects": {}, "overall": {}}
    all_steps: List[float] = []
    all_times: List[float] = []

    for object_name in sorted(results):
        steps_list, time_list = collect_metrics(
            results[object_name], success_only=args.success_only
        )
        if not steps_list:
            continue

        all_steps.extend(steps_list)
        all_times.extend(time_list)
        summary["objects"][object_name] = {
            "count": len(steps_list),
            "avg_steps": average(steps_list),
            "avg_time": average(time_list),
        }

    summary["overall"] = {
        "count": len(all_steps),
        "avg_steps": average(all_steps),
        "avg_time": average(all_times),
    }
    summary["success_only"] = args.success_only

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
