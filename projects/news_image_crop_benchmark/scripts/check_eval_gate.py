#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_gate(summary: dict, expected_tasks: int) -> None:
    overall = summary["overall"]
    failures = []
    if overall["tasks"] != expected_tasks:
        failures.append(f"tasks={overall['tasks']} (expected {expected_tasks})")
    if overall["generation_success_count"] != expected_tasks:
        failures.append(
            f"generation_success_count={overall['generation_success_count']} (expected {expected_tasks})"
        )
    if overall["retry_exhausted_count"] != 0:
        failures.append(f"retry_exhausted_count={overall['retry_exhausted_count']} (expected 0)")
    if overall["canonical_format_count"] != expected_tasks:
        failures.append(
            f"canonical_format_count={overall['canonical_format_count']} (expected {expected_tasks})"
        )
    if overall["judge_completed_count"] != expected_tasks:
        failures.append(f"judge_completed_count={overall['judge_completed_count']} (expected {expected_tasks})")
    if overall["judge_failed_count"] != 0:
        failures.append(f"judge_failed_count={overall['judge_failed_count']} (expected 0)")
    if overall["judge_parse_fallback_count"] != 0:
        failures.append(
            f"judge_parse_fallback_count={overall['judge_parse_fallback_count']} (expected 0)"
        )
    if failures:
        raise RuntimeError("evaluation gate failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Require a complete and scoreable evaluation summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    if args.expected_tasks <= 0:
        raise ValueError("expected-tasks must be positive")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    check_gate(summary, args.expected_tasks)
    print(
        json.dumps(
            {
                "status": "pass",
                "model_name": summary.get("model_name"),
                "expected_tasks": args.expected_tasks,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()