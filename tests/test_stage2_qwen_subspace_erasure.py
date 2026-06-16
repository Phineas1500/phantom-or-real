from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from scripts.stage2_qwen_subspace_erasure import make_hf_erasure_hook


def test_hf_erasure_hook_sets_projection_to_mean_for_tensor_output() -> None:
    hook, state = make_hf_erasure_hook(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        projection_mean=2.0,
        projection_std=4.0,
    )
    hidden = torch.tensor([[[5.0, 1.0], [-1.0, 3.0]]], dtype=torch.float32)

    patched = hook(None, None, hidden)

    assert_close(patched[:, :, 0], torch.tensor([[2.0, 2.0]]))
    assert_close(patched[:, :, 1], torch.tensor([[1.0, 3.0]]))
    assert state["calls"] == 1
    assert state["positions"] == 2
    assert state["abs_delta_sd_sum"] == pytest.approx((3.0 + 3.0) / 4.0)


def test_hf_erasure_hook_preserves_tuple_output_shape() -> None:
    hook, _state = make_hf_erasure_hook(
        vector=np.array([0.0, 1.0], dtype=np.float32),
        projection_mean=0.5,
        projection_std=1.0,
    )
    hidden = torch.tensor([[[2.0, 4.0]]], dtype=torch.float32)
    extra = torch.tensor([1.0])

    patched_hidden, patched_extra = hook(None, None, (hidden, extra))

    assert_close(patched_hidden, torch.tensor([[[2.0, 0.5]]]))
    assert patched_extra is extra
