"""CPU checks for the SPIDER sampling update, without a GPU rollout."""

import torch

from planning.spider_mppi import compute_spider_weights, interp_knots, sample_ctrls


def test_interp_knots_matches_spider_linear_length():
    src = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]], dtype=torch.float32)
    stretched = interp_knots(src, 3)
    assert stretched.shape == (1, 9, 2)
    torch.testing.assert_close(stretched[:, 0], src[:, 0])
    torch.testing.assert_close(stretched[:, -1], src[:, -1])
    single = interp_knots(src[:, :1], 4)
    assert single.shape == (1, 4, 2)
    torch.testing.assert_close(single[:, 0], src[:, 0])


def test_sample_ctrls_adds_interpolated_knot_noise():
    torch.manual_seed(0)
    ctrls = torch.zeros(6, 2)
    noise_scale = torch.ones(5, 3, 2)
    samples = sample_ctrls(ctrls, noise_scale, knot_steps=2, global_noise_scale=0.5)
    assert samples.shape == (5, 6, 2)
    assert torch.isfinite(samples).all()
    assert float(samples.abs().mean()) > 0.0


def test_spider_weights_use_top_ten_percent_standardized_softmax():
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, float("nan"), 5.0, 4.0, -1.0, 8.0, 6.0])
    weights = compute_spider_weights(rewards, temperature=0.1)
    assert weights.shape == (10,)
    assert int(torch.count_nonzero(weights)) == 1
    assert int(torch.argmax(weights)) == 8
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))

    tied = torch.arange(20, dtype=torch.float32)
    weights = compute_spider_weights(tied, temperature=1.0)
    assert int(torch.count_nonzero(weights)) == 2
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert float(weights[-1]) > float(weights[-2]) > 0.0
