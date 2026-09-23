import numpy as np
import pytest

from maskfits.automask import (
    auto_mask_preview,
    background_stats,
    expand_mask,
    filter_by_group_size,
    fit_gaussian_to_histogram,
    neural_network_mask,
    sigma_clip_mask,
    valid_pixels,
)


def _background_with_source(shape=(60, 60), bg=100.0, sigma=5.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(bg, sigma, size=shape)
    # A bright, compact "source" well above the background - should get
    # clipped out of the background estimate and later flagged for masking.
    data[25:35, 25:35] = 500.0
    return data


def test_valid_pixels_excludes_zero_and_nan():
    data = np.array([1.0, 0.0, np.nan, -2.0, 3.0])
    assert list(valid_pixels(data)) == [True, False, False, True, True]


def test_sigma_clip_mask_removes_bright_source():
    data = _background_with_source()
    valid = valid_pixels(data)
    kept = sigma_clip_mask(data, valid, kappa=3.0)
    # The 10x10 bright block should be fully clipped out of the background set.
    assert not kept[25:35, 25:35].any()
    # Most of the surrounding background should survive.
    assert kept.sum() > 0.8 * valid.sum()


def test_sigma_clip_mask_zero_kappa_is_a_noop():
    data = _background_with_source()
    valid = valid_pixels(data)
    kept = sigma_clip_mask(data, valid, kappa=0.0)
    assert np.array_equal(kept, valid)


def test_background_stats_sigma_vs_sem():
    data = _background_with_source()
    valid = valid_pixels(data)
    kept = sigma_clip_mask(data, valid, kappa=3.0)
    bg, sigma_err = background_stats(data, kept, "sigma")
    _, sem_err = background_stats(data, kept, "sem")
    assert bg == pytest.approx(100.0, abs=1.0)
    assert sem_err < sigma_err  # SEM = sigma / sqrt(n), always <= sigma for n > 1
    assert sem_err == pytest.approx(sigma_err / np.sqrt(int(kept.sum())), rel=1e-6)


def test_background_stats_empty_kept_returns_zero():
    data = np.zeros((5, 5))
    kept = np.zeros((5, 5), dtype=bool)
    assert background_stats(data, kept, "sigma") == (0.0, 0.0)


def test_auto_mask_preview_flags_only_bright_source():
    data = _background_with_source()
    valid = valid_pixels(data)
    kept = sigma_clip_mask(data, valid, kappa=3.0)
    bg, err = background_stats(data, kept, "sigma")
    preview = auto_mask_preview(data, valid, bg, err, kappa=5.0)
    assert preview[25:35, 25:35].all()
    # Background-only region should be essentially unflagged.
    assert preview[0:10, 0:10].sum() < 5


def test_auto_mask_preview_excludes_zero_and_nan_even_above_threshold():
    data = np.full((5, 5), 100.0)
    data[0, 0] = 0.0
    data[0, 1] = np.nan
    valid = valid_pixels(data)
    preview = auto_mask_preview(data, valid, bg=-10.0, bg_err=0.0, kappa=1.0)  # threshold below 0
    assert not preview[0, 0]
    assert not preview[0, 1]


def test_auto_mask_preview_excludes_already_masked_pixels():
    data = np.full((5, 5), 100.0)
    existing_mask = np.zeros((5, 5), dtype=bool)
    existing_mask[2, 2] = True  # already masked, even though it's above threshold
    valid = valid_pixels(data) & ~existing_mask
    preview = auto_mask_preview(data, valid, bg=0.0, bg_err=0.0, kappa=1.0)
    assert not preview[2, 2], "an already-masked pixel must never be (re-)flagged"
    assert preview[0, 0], "an ordinary valid pixel above threshold should still be flagged"
    assert preview[1, 1]  # ordinary valid pixel above threshold


def test_expand_mask_pads_by_radius():
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    expanded = expand_mask(mask, radius=3)
    assert expanded[10, 13]  # 3px to the right, on the disk boundary
    assert not expanded[10, 14]  # just outside the disk
    assert expanded[10, 10]


def test_expand_mask_zero_radius_is_a_noop():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    expanded = expand_mask(mask, radius=0)
    assert np.array_equal(expanded, mask)
    assert expanded is mask  # explicitly returns the same object, not a copy


def test_neural_network_mask_is_an_unimplemented_placeholder():
    with pytest.raises(NotImplementedError):
        neural_network_mask(np.zeros((3, 3)), "some/model/path")


def test_fit_gaussian_to_histogram_recovers_known_parameters():
    rng = np.random.default_rng(2)
    mu, sigma = 50.0, 4.0
    values = rng.normal(mu, sigma, size=20000)
    fit = fit_gaussian_to_histogram(values, bins=80, hist_range=(mu - 8 * sigma, mu + 8 * sigma))
    assert fit is not None
    amplitude, fit_mu, fit_sigma = fit
    assert amplitude > 0
    assert fit_mu == pytest.approx(mu, abs=0.5)
    assert fit_sigma == pytest.approx(sigma, abs=0.5)


def test_fit_gaussian_to_histogram_too_few_points_returns_none():
    assert fit_gaussian_to_histogram(np.array([1.0, 2.0]), bins=10, hist_range=(0, 3)) is None


def test_fit_gaussian_to_histogram_falls_back_on_degenerate_input():
    # A single repeated value: curve_fit has nothing to converge on, but the
    # function must still return a usable (non-crashing) moment-based estimate.
    values = np.full(50, 7.0)
    fit = fit_gaussian_to_histogram(values, bins=10, hist_range=(0, 14))
    assert fit is not None
    amplitude, mu, sigma = fit
    assert mu == pytest.approx(7.0)
    assert sigma > 0


def test_filter_by_group_size_keeps_small_drops_large():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5, 5] = True  # a lone 1px group
    mask[10:13, 10:13] = True  # a compact 3x3 = 9px group
    mask[20:26, 20:26] = True  # a 6x6 = 36px group - too big at max_size=10
    filtered = filter_by_group_size(mask, max_size=10)
    assert filtered[5, 5]
    assert filtered[10:13, 10:13].all()
    assert not filtered[20:26, 20:26].any()


def test_filter_by_group_size_uses_8_connectivity():
    # Two single pixels touching only diagonally still count as one group.
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    mask[5, 5] = True
    filtered = filter_by_group_size(mask, max_size=1)
    assert not filtered.any(), "an 8-connected pair should be treated as one 2px group, dropped at max_size=1"
    filtered_ok = filter_by_group_size(mask, max_size=2)
    assert filtered_ok[4, 4] and filtered_ok[5, 5]


def test_filter_by_group_size_disabled_at_zero():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    filtered = filter_by_group_size(mask, max_size=0)
    assert np.array_equal(filtered, mask)


def test_filter_by_group_size_empty_mask_is_a_noop():
    mask = np.zeros((5, 5), dtype=bool)
    assert not filter_by_group_size(mask, max_size=10).any()
