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

from verl.models.transformers.qwen3_5 import vision_rotary_embedding_forward


def test_vision_rotary_embedding_moves_inv_freq_to_input_device() -> None:
    class RecordingBuffer:
        def __init__(self):
            self.requested_device = None

        def to(self, *, device):
            self.requested_device = device
            return torch.tensor([1.0, 0.5], device=device)

    buffer = RecordingBuffer()
    module = type("VisionRotary", (), {"inv_freq": buffer})()
    position_ids = torch.tensor([[0, 1], [2, 3]])

    output = vision_rotary_embedding_forward(module, position_ids)

    assert buffer.requested_device == position_ids.device
    assert torch.equal(output, torch.tensor([[0.0, 0.0, 1.0, 0.5], [2.0, 1.0, 3.0, 1.5]]))