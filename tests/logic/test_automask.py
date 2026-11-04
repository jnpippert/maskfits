import numpy as np
import pytest

from maskfits.automask import (
    auto_mask_preview,
    background_stats,
    expand_mask,
    filter_by_group_size,
    fit_gaussian_to_histogram,
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


def test_background_stats_median_and_std():
    data = _background_with_source()
    valid = valid_pixels(data)
    kept = sigma_clip_mask(data, valid, kappa=3.0)
    bg, err = background_stats(data, kept, "median")
    assert bg == pytest.approx(float(np.median(data[kept])))
    assert bg == pytest.approx(100.0, abs=1.0)
    assert err == pytest.approx(float(data[kept].std()))


def test_background_stats_defaults_to_median():
    data = _background_with_source()
    kept = valid_pixels(data)
    assert background_stats(data, kept) == background_stats(data, kept, "median")


def test_background_stats_mode_is_the_peak_bin_center():
    from maskfits.automask import MODE_BINS

    # Skewed background: a long tail of faint sources pulls the median up,
    # the mode stays at the true sky level.
    rng = np.random.default_rng(7)
    sky = rng.normal(100.0, 2.0, 50000)
    tail = 100.0 + rng.exponential(15.0, 20000)
    data = np.concatenate([sky, tail])
    kept = np.ones(data.shape, dtype=bool)

    mode, err = background_stats(data, kept, "mode")
    median, _ = background_stats(data, kept, "median")

    counts, edges = np.histogram(data, bins=MODE_BINS, range=(data.min(), data.max()))
    peak = int(np.argmax(counts))
    assert mode == pytest.approx((edges[peak] + edges[peak + 1]) / 2)
    assert abs(mode - 100.0) < abs(median - 100.0)
    assert err == pytest.approx(float(data.std()))


def test_background_stats_mode_of_a_constant_image():
    data = np.full((10, 10), 42.0)
    assert background_stats(data, np.ones((10, 10), dtype=bool), "mode") == (42.0, 0.0)


def test_background_stats_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        background_stats(np.ones((3, 3)), np.ones((3, 3), dtype=bool), "mean")


def test_background_stats_empty_kept_returns_zero():
    data = np.zeros((5, 5))
    kept = np.zeros((5, 5), dtype=bool)
    assert background_stats(data, kept, "median") == (0.0, 0.0)
    assert background_stats(data, kept, "mode") == (0.0, 0.0)


def test_auto_mask_preview_flags_only_bright_source():
    data = _background_with_source()
    valid = valid_pixels(data)
    kept = sigma_clip_mask(data, valid, kappa=3.0)
    bg, err = background_stats(data, kept)
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


def test_filter_by_group_size_min_drops_small_groups():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5, 5] = True  # 1px
    mask[10:13, 10:13] = True  # 9px
    mask[20:26, 20:26] = True  # 36px
    filtered = filter_by_group_size(mask, min_size=5)
    assert not filtered[5, 5]
    assert filtered[10:13, 10:13].all()
    assert filtered[20:26, 20:26].all()


def test_filter_by_group_size_keeps_only_the_range():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5, 5] = True  # 1px
    mask[10:13, 10:13] = True  # 9px
    mask[20:26, 20:26] = True  # 36px
    filtered = filter_by_group_size(mask, max_size=10, min_size=5)
    assert not filtered[5, 5]
    assert filtered[10:13, 10:13].all()
    assert not filtered[20:26, 20:26].any()


def test_filter_by_group_size_range_bounds_are_inclusive():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:8, 5:8] = True  # exactly 9px
    assert filter_by_group_size(mask, max_size=9, min_size=9)[5:8, 5:8].all()


def test_filter_by_group_size_min_of_one_or_less_is_no_lower_limit():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 2] = True
    assert np.array_equal(filter_by_group_size(mask, min_size=1), mask)
    assert np.array_equal(filter_by_group_size(mask, min_size=0), mask)


def test_filter_by_group_size_counts_pixels_on_the_image_border():
    # Gap-closing can erode border pixels; they must still belong to their
    # group instead of being dropped as "background".
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, 0:3] = True  # a 3px group on the top edge
    assert filter_by_group_size(mask, max_size=10)[0, 0:3].all()


# ------------------------------------------------------------------ shape


def _scene():
    """A round blob, a thin slanted streak and a small square, well apart."""
    from maskfits.automask import _bridged_groups  # noqa: F401 - imported for the tests below

    mask = np.zeros((200, 200), dtype=bool)
    yy, xx = np.mgrid[:200, :200]
    mask |= (xx - 50) ** 2 + (yy - 50) ** 2 <= 12 ** 2
    mask |= np.abs((yy - 150) - 0.3 * (xx - 100)) <= 2
    mask[180:184, 10:14] = True
    return mask


def test_group_elongation_round_vs_streak():
    from maskfits.automask import _bridged_groups, group_elongation

    labeled, num = _bridged_groups(_scene())
    elongation = group_elongation(labeled, num)
    round_blob = elongation[labeled[50, 50]]
    streak = elongation[labeled[150, 100]]
    square = elongation[labeled[181, 11]]
    assert round_blob == pytest.approx(1.0, abs=0.05)
    assert square == pytest.approx(1.0, abs=0.05)
    assert streak > 20


def test_group_elongation_of_a_bar_is_its_length_to_width_ratio():
    from maskfits.automask import group_elongation
    from scipy.ndimage import label

    mask = np.zeros((40, 80), dtype=bool)
    mask[10:14, 10:70] = True  # 60 x 4 bar
    labeled, num = label(mask)
    assert group_elongation(labeled, num)[1] == pytest.approx(15.0, rel=1e-6)


def test_group_elongation_single_pixel_is_round():
    from maskfits.automask import group_elongation
    from scipy.ndimage import label

    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    labeled, num = label(mask)
    assert group_elongation(labeled, num)[1] == pytest.approx(1.0)


def test_filter_by_shape_lines_keeps_only_the_streak():
    from maskfits.automask import filter_by_shape

    mask = _scene()
    lines = filter_by_shape(mask, "lines")
    assert lines[150, 100] and not lines[50, 50] and not lines[181, 11]
    assert not (lines & ~mask).any()  # never adds pixels


def test_filter_by_shape_circles_drops_only_the_streak():
    from maskfits.automask import filter_by_shape

    mask = _scene()
    circles = filter_by_shape(mask, "circles")
    assert not circles[150, 100] and circles[50, 50] and circles[181, 11]


def test_filter_by_shape_modes_are_complementary():
    from maskfits.automask import filter_by_shape

    mask = np.random.default_rng(4).random((60, 60)) < 0.08
    lines, circles = filter_by_shape(mask, "lines"), filter_by_shape(mask, "circles")
    assert not (lines & circles).any()
    assert np.array_equal(lines | circles, mask)


def test_filter_by_shape_treats_a_fragmented_streak_as_one_line():
    from maskfits.automask import filter_by_shape

    # A 4px-wide trail with ~45% of its pixels lost to noise - dozens of
    # separate specks before the gap-closing step bridges them.
    mask = np.zeros((40, 200), dtype=bool)
    mask[18:22, 10:190] = True
    mask &= np.random.default_rng(5).random(mask.shape) > 0.45
    assert np.array_equal(filter_by_shape(mask, "lines"), mask)


def test_filter_by_shape_rejects_an_unknown_mode():
    from maskfits.automask import filter_by_shape

    with pytest.raises(ValueError):
        filter_by_shape(np.ones((3, 3), dtype=bool), "squares")
