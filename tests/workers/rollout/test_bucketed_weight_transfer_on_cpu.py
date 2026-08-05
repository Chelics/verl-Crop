# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.multiprocessing.reductions import reduce_tensor

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import rebuild_ipc


def test_rebuild_ipc_accepts_cpu_reducer_with_device_id() -> None:
    source = torch.arange(8, dtype=torch.float32)

    rebuilt = rebuild_ipc(reduce_tensor(source), device_id=0)

    assert rebuilt.device.type == "cpu"
    assert torch.equal(rebuilt, source)


def test_rebuild_ipc_replaces_named_storage_device() -> None:
    captured = {}

    def fake_rebuild_tensor(tensor_cls, storage_device, payload):
        captured["tensor_cls"] = tensor_cls
        captured["storage_device"] = storage_device
        return payload

    source = torch.tensor([1, 2, 3])
    rebuilt = rebuild_ipc((fake_rebuild_tensor, (torch.Tensor, 7, source)), device_id=2)

    assert captured == {"tensor_cls": torch.Tensor, "storage_device": 2}
    assert rebuilt is source