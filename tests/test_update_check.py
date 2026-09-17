import subprocess
from pathlib import Path

import pytest

from maskfits import update_check


def test_repo_root_finds_the_real_checkout():
    root = update_check._repo_root()
    assert root is not None
    assert (root / ".git").exists()


def test_repo_root_returns_none_outside_any_git_repo(tmp_path, monkeypatch):
    fake_module_path = tmp_path / "not_a_repo" / "src" / "maskfits" / "update_check.py"
    fake_module_path.parent.mkdir(parents=True)
    fake_module_path.write_text("")
    monkeypatch.setattr(update_check, "__file__", str(fake_module_path))
    assert update_check._repo_root() is None


def test_check_for_updates_against_the_real_repo_does_not_raise():
    # Exercises the real code path against this actual checkout - whatever
    # the answer is (up to date / behind / offline), it must come back as a
    # (bool, str) tuple, never an exception.
    available, message = update_check.check_for_updates()
    assert isinstance(available, bool)
    assert isinstance(message, str) and message


def test_reports_up_to_date_when_local_matches_remote(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["git", "fetch"]:
            return ""
        if args == ["git", "rev-parse", "origin/main"]:
            return "abc123"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "up to date" in message


def test_reports_update_available_with_commit_count(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["git", "fetch"]:
            return ""
        if args == ["git", "rev-parse", "origin/main"]:
            return "def456"
        if args[:2] == ["git", "rev-list"]:
            return "4"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is True
    assert "4 new commits" in message
    assert "git pull" in message


def test_detached_head_reports_cleanly(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "HEAD"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "detached" in message.lower()


def test_missing_git_reports_cleanly(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        raise FileNotFoundError("git")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "git isn't available" in message


def test_timeout_reports_cleanly(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        raise subprocess.TimeoutExpired(cmd=args, timeout=15)

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "timed out" in message.lower()


def test_no_git_repo_reports_cleanly(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: None)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "git clone" in message.lower()


def test_dns_failure_gets_a_friendly_network_headline(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["git", "fetch"]:
            raise subprocess.CalledProcessError(
                128, args, stderr="fatal: unable to access 'https://github.com/x/y.git/': "
                                  "Could not resolve host: github.com",
            )
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "no internet connection" in message.lower()


def test_unrelated_git_error_skips_the_network_headline(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["git", "fetch"]:
            raise subprocess.CalledProcessError(128, args, stderr="fatal: some unrelated git error")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_updates()
    assert available is False
    assert "no internet connection" not in message.lower()
    assert "some unrelated git error" in message


# ----------------------------------------------------------- major version


@pytest.mark.parametrize(
    "version_str, expected",
    [("1.0.0", 1), ("2.3.4", 2), ("10.0.1", 10), ("0.1.0", 0), ("1.0.0+unknown", 1), ("not-a-version", None)],
)
def test_major_version_parsing(version_str, expected):
    assert update_check._major_version(version_str) == expected


def _fake_run_for_major_check(local_sha, remote_sha, remote_toml):
    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return local_sha
        if args[:2] == ["git", "fetch"]:
            return ""
        if args == ["git", "rev-parse", "origin/main"]:
            return remote_sha
        if args == ["git", "show", "origin/main:pyproject.toml"]:
            return remote_toml
        raise AssertionError(f"unexpected git call: {args}")

    return fake_run


def test_major_update_silent_for_a_minor_bump(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))
    monkeypatch.setattr(update_check, "LOCAL_VERSION", "1.0.0")
    monkeypatch.setattr(
        update_check, "_run",
        _fake_run_for_major_check("abc123", "def456", '[project]\nversion = "1.1.0"\n'),
    )
    available, message = update_check.check_for_major_update()
    assert available is False
    assert "no major" in message.lower()


def test_major_update_silent_for_a_patch_bump(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))
    monkeypatch.setattr(update_check, "LOCAL_VERSION", "1.2.2")
    monkeypatch.setattr(
        update_check, "_run",
        _fake_run_for_major_check("abc123", "def456", '[project]\nversion = "1.2.3"\n'),
    )
    available, _message = update_check.check_for_major_update()
    assert available is False


def test_major_update_reports_true_for_a_major_bump(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))
    monkeypatch.setattr(update_check, "LOCAL_VERSION", "1.0.0")
    monkeypatch.setattr(
        update_check, "_run",
        _fake_run_for_major_check("abc123", "def456", '[project]\nversion = "2.0.0"\n'),
    )
    available, message = update_check.check_for_major_update()
    assert available is True
    assert "v2.0.0" in message
    assert "v1.0.0" in message
    assert "re-cloning" in message


def test_major_update_silent_when_remote_major_is_lower(monkeypatch):
    # Local is somehow ahead of the remote's own version bookkeeping -
    # should never happen in practice, but must not falsely report an
    # update either way.
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))
    monkeypatch.setattr(update_check, "LOCAL_VERSION", "3.0.0")
    monkeypatch.setattr(
        update_check, "_run",
        _fake_run_for_major_check("abc123", "def456", '[project]\nversion = "2.0.0"\n'),
    )
    available, _message = update_check.check_for_major_update()
    assert available is False


def test_major_update_silent_when_remote_pyproject_unreadable(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))
    monkeypatch.setattr(update_check, "LOCAL_VERSION", "1.0.0")

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["git", "fetch"]:
            return ""
        if args == ["git", "rev-parse", "origin/main"]:
            return "def456"
        if args == ["git", "show", "origin/main:pyproject.toml"]:
            raise subprocess.CalledProcessError(128, args, stderr="fatal: path does not exist")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, _message = update_check.check_for_major_update()
    assert available is False


def test_major_update_up_to_date_short_circuits_before_reading_pyproject(monkeypatch):
    monkeypatch.setattr(update_check, "_repo_root", lambda: Path("/fake/repo"))

    def fake_run(args, cwd):
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return "main"
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["git", "fetch"]:
            return ""
        if args == ["git", "rev-parse", "origin/main"]:
            return "abc123"
        raise AssertionError(f"unexpected git call: {args} - should never reach pyproject.toml")

    monkeypatch.setattr(update_check, "_run", fake_run)
    available, message = update_check.check_for_major_update()
    assert available is False
    assert "up to date" in message


def test_remote_pyproject_version_parses_the_version_line():
    toml = 'other = 1\n\n[project]\nname = "maskfits"\nversion = "3.4.5"\ndescription = "x"\n'

    def fake_run(args, cwd):
        return toml

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(update_check, "_run", fake_run)
        assert update_check._remote_pyproject_version(Path("/fake/repo"), "main") == "3.4.5"
