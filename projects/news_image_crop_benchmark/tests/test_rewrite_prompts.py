import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_rewriter_module():
    script_path = Path(__file__).parents[1] / "scripts" / "rewrite_prompts.py"
    spec = importlib.util.spec_from_file_location("test_rewrite_prompts_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rewrite_prompts_can_write_versioned_output_without_mutating_input(tmp_path):
    module = load_rewriter_module()
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "prompt_v1" / "input.parquet"
    original_prompt = [{"role": "user", "content": "<image>\nOld prompt"}]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": original_prompt,
                    "extra_info": {"title": "A title", "target_ratio": 1.59},
                }
            ]
        ),
        input_path,
    )
    template = "<image>\nHeadline: {title}\nRatio: {target_ratio}"

    report = module.rewrite_prompts(
        input_path,
        policy_prompt_template=template,
        output_path=output_path,
    )

    assert pq.read_table(input_path).to_pylist()[0]["prompt"] == original_prompt
    rewritten_prompt = pq.read_table(output_path).to_pylist()[0]["prompt"][0]["content"]
    assert rewritten_prompt == "<image>\nHeadline: A title\nRatio: 1.59"
    assert report["changed"] == 1
    assert report["output_path"] == str(output_path.resolve())
    assert len(report["policy_prompt_sha256"]) == 64
