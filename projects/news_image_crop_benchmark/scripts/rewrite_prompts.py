#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from news_crop_benchmark.data import build_prompt, load_policy_prompt_template


def rewrite_prompts(
    path: Path,
    *,
    policy_prompt_template: str | None = None,
    output_path: Path | None = None,
) -> dict:
    table = pq.read_table(path)
    rows = table.to_pylist()
    changed = 0
    for row in rows:
        extra_info = row["extra_info"]
        prompt = build_prompt(
            extra_info["title"],
            float(extra_info["target_ratio"]),
            policy_prompt_template,
        )
        if row["prompt"] != [{"role": "user", "content": prompt}]:
            row["prompt"] = [{"role": "user", "content": prompt}]
            changed += 1

    destination = output_path or path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.tmp")
    rewritten = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(rewritten, temporary_path, compression="zstd", row_group_size=1024)
    temporary_path.replace(destination)
    template = load_policy_prompt_template() if policy_prompt_template is None else policy_prompt_template
    return {
        "input_path": str(path.resolve()),
        "output_path": str(destination.resolve()),
        "rows": len(rows),
        "changed": changed,
        "policy_prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite converted Parquet prompts using the current prompt contract.")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-prompt-path", type=Path)
    args = parser.parse_args()

    policy_prompt_template = load_policy_prompt_template(args.policy_prompt_path)
    reports = [
        rewrite_prompts(
            path.expanduser(),
            policy_prompt_template=policy_prompt_template,
            output_path=(args.output_dir.expanduser() / path.name if args.output_dir else None),
        )
        for path in args.data
    ]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
