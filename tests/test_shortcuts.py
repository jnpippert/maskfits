from maskfits.shortcuts import (
    SHORTCUT_ACTIONS,
    SHORTCUT_ACTIONS_BY_ID,
    effective_keys,
    find_conflict,
    sanitize_overrides,
)


def test_every_action_has_a_unique_id():
    ids = [a.id for a in SHORTCUT_ACTIONS]
    assert len(ids) == len(set(ids))


def test_every_action_has_at_least_one_default():
    for action in SHORTCUT_ACTIONS:
        assert len(action.defaults) >= 1


def test_effective_keys_falls_back_to_defaults():
    action = SHORTCUT_ACTIONS[0]
    assert effective_keys(action.id, {}) == action.defaults


def test_effective_keys_uses_override_when_present():
    action = SHORTCUT_ACTIONS[0]
    override = {action.id: ["Z"]}
    assert effective_keys(action.id, override) == ("Z",)


def test_sanitize_overrides_drops_unknown_action_ids():
    raw = {"not_a_real_action": ["X"], "undo": ["Z"]}
    cleaned = sanitize_overrides(raw)
    assert "not_a_real_action" not in cleaned
    assert cleaned["undo"] == ["Z"]


def test_sanitize_overrides_drops_non_list_values():
    raw = {"undo": "Z"}  # a bare string, not a list
    assert sanitize_overrides(raw) == {}


def test_sanitize_overrides_drops_non_string_entries():
    raw = {"undo": ["Z", 5]}
    assert sanitize_overrides(raw) == {}


def test_sanitize_overrides_drops_empty_lists():
    raw = {"undo": []}
    assert sanitize_overrides(raw) == {}


def test_sanitize_overrides_rejects_non_dict_input():
    assert sanitize_overrides(["not", "a", "dict"]) == {}
    assert sanitize_overrides(None) == {}


def test_sanitize_overrides_keeps_valid_entries():
    raw = {"undo": ["Ctrl+Z", "U"], "redo": ["Y"]}
    assert sanitize_overrides(raw) == raw


def test_find_conflict_detects_a_shared_key():
    # "clear_mask" defaults to "R" - binding "undo"'s only key to "R" too
    # should be flagged as conflicting with clear_mask's label.
    overrides = {"undo": ["R"]}
    conflict = find_conflict("undo", "R", overrides)
    assert conflict == SHORTCUT_ACTIONS_BY_ID["clear_mask"].label


def test_find_conflict_none_when_key_is_unique():
    assert find_conflict("undo", "Ctrl+Alt+Shift+F12", {}) is None


def test_find_conflict_ignores_the_action_itself():
    # redo's own second default key ("Y") shouldn't conflict with itself.
    assert find_conflict("redo", "Y", {}) is None
