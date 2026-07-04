import numpy as np
import torch

from scripts.stage2_subspace_erasure import (
    is_stack_kind,
    make_condition_plan,
    make_subspace_erasure_hook,
    parse_condition_kinds,
)


def test_condition_plan_expands_random_stack_draws() -> None:
    kinds = parse_condition_kinds("baseline,erase_raw,erase_readable_stack,erase_random_stack")
    plan = make_condition_plan(kinds, random_stack_draws=3)
    labels = [condition.label for condition in plan]
    assert labels == [
        "baseline",
        "erase_raw",
        "erase_readable_stack",
        "erase_random_stack_d1",
        "erase_random_stack_d2",
        "erase_random_stack_d3",
    ]
    assert is_stack_kind(plan[2].vector_kind)
    assert is_stack_kind(plan[3].vector_kind)
    assert not is_stack_kind(plan[1].vector_kind)
    assert not is_stack_kind(None)


def test_subspace_erasure_hook_clamps_projections_and_preserves_complement() -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((16, 3)))
    means = np.array([0.5, -1.0, 2.0])
    stds = np.array([1.0, 2.0, 0.5])
    hook_fn, state = make_subspace_erasure_hook(basis=q, projection_means=means, projection_stds=stds)

    act = torch.from_numpy(rng.standard_normal((1, 7, 16)).astype(np.float32))
    complement = np.eye(16) - q @ q.T
    orthogonal_before = act.numpy().astype(np.float64) @ complement

    out = hook_fn(act.clone(), None)
    proj_after = out.numpy().astype(np.float64) @ q
    np.testing.assert_allclose(proj_after, np.broadcast_to(means, proj_after.shape), atol=1e-4)
    orthogonal_after = out.numpy().astype(np.float64) @ complement
    np.testing.assert_allclose(orthogonal_after, orthogonal_before, atol=1e-4)

    assert state["calls"] == 1
    assert state["positions"] == 7
    assert state["prompt_projection_variance"] is not None
    assert len(state["prompt_projection_variance"]) == 3
    assert len(state["prompt_projection_mean"]) == 3
    prompt_variance = list(state["prompt_projection_variance"])

    decode_step = torch.from_numpy(rng.standard_normal((1, 1, 16)).astype(np.float32))
    hook_fn(decode_step, None)
    assert state["calls"] == 2
    assert state["positions"] == 8
    assert state["prompt_projection_variance"] == prompt_variance
