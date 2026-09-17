"""Two independent update checks, for two different audiences:

- check_for_updates(): Help -> Check New Repo Version... - compares the
  local git clone's current commit against its remote's tip via `git fetch`
  + `git rev-list`, without touching the working tree. Only meaningful for
  an editable/dev install run from an actual git clone (a regular `pip
  install maskfits` has no .git directory to compare against, so this
  cleanly reports that instead of guessing) - this is the "is the repo
  itself ahead of what I have checked out" check, for contributors/devs.

- check_for_major_pip_update() (see UpdateCheckWorker): the silent startup
  check, run for every install regardless of how it was installed - compares
  this install's own __version__ against the latest version published on
  PyPI, only reporting available=True for an actual major-version bump
  (e.g. 1.x -> 2.0.0), so routine minor/patch releases stay quiet. This is
  "is there a newer stable release", the one relevant to every end user.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from PySide6.QtCore import QObject, Signal

from maskfits import __version__ as LOCAL_VERSION

TIMEOUT_S = 15
PYPI_PACKAGE_NAME = "maskfits"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"


def _repo_root() -> Optional[Path]:
    """Walks up from this file to find the enclosing git repo root, if any."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


# Substrings git's own stderr uses for the common connectivity failures (no
# internet, DNS down, GitHub unreachable/down, SSL interception, ...) - used
# to put a plain-English headline above the raw git detail instead of just
# dumping "fatal: unable to access ...: Could not resolve host: github.com"
# on its own.
_NETWORK_ERROR_HINTS = (
    "could not resolve host",
    "network is unreachable",
    "connection refused",
    "connection reset",
    "could not connect",
    "timed out",
    "unable to access",
    "could not read from remote repository",
    "ssl",
    "certificate",
    "temporary failure in name resolution",
)


class _CheckFailure(Exception):
    """Internal only: carries a ready-to-show message for a failed
    connectivity/git step - caught by every public check_for_*() function so
    each one doesn't repeat the same except blocks."""


def _run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S, check=True,
    )
    return result.stdout.strip()


def get_clone_url() -> Optional[str]:
    """The repo's own `origin` URL (whatever the user actually cloned with -
    https or ssh) - used to build a copy-pasteable `git clone <url>` command
    in the update popup (see update_dialog.py). None if this isn't a git
    clone, or it has no `origin` remote, rather than guessing a URL."""
    root = _repo_root()
    if root is None:
        return None
    try:
        return _run(["git", "config", "--get", "remote.origin.url"], root) or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _fetch_remote_state(root: Path) -> tuple[str, str, str]:
    """Returns (branch, local_sha, remote_sha) after fetching - raises
    _CheckFailure with a ready-to-show message on any failure."""
    try:
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
        if branch == "HEAD":
            raise _CheckFailure("Not on a branch (detached HEAD) - nothing to compare against.")
        local_sha = _run(["git", "rev-parse", "HEAD"], root)
        _run(["git", "fetch", "origin", branch], root)
        remote_sha = _run(["git", "rev-parse", f"origin/{branch}"], root)
    except FileNotFoundError:
        raise _CheckFailure("git isn't available - can't check for updates.") from None
    except subprocess.TimeoutExpired:
        raise _CheckFailure(
            "Checking for updates timed out - no internet connection, or GitHub isn't reachable right now."
        ) from None
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        if any(hint in detail.lower() for hint in _NETWORK_ERROR_HINTS):
            raise _CheckFailure(f"No internet connection, or GitHub isn't reachable right now.\n\n{detail}") from None
        raise _CheckFailure(f"Could not check for updates:\n{detail}") from None
    return branch, local_sha, remote_sha


def _major_version(version_str: str) -> Optional[int]:
    match = re.match(r"\s*(\d+)", version_str)
    return int(match.group(1)) if match else None


def check_for_updates() -> tuple[bool, str]:
    """Returns (update_available, message) for ANY new commit on the
    remote - used by the manual Help -> Check New Repo Version... action,
    which reports every commit, not just major bumps (see
    check_for_major_pip_update for the quieter startup check, which looks at
    PyPI releases instead of raw commits). Never raises; any failure
    (offline, GitHub unreachable/down, no git, not a clone, ...) comes back
    as a message instead."""
    root = _repo_root()
    if root is None:
        return False, "Not running from a git clone - nothing to check."

    try:
        branch, local_sha, remote_sha = _fetch_remote_state(root)
    except _CheckFailure as exc:
        return False, str(exc)

    if remote_sha == local_sha:
        return False, f"You're up to date with origin/{branch}."

    try:
        count = int(_run(["git", "rev-list", f"{local_sha}..{remote_sha}", "--count"], root))
    except (subprocess.CalledProcessError, ValueError):
        count = None

    if count:
        plural = "s" if count != 1 else ""
        return True, (
            f"Update available: {count} new commit{plural} on origin/{branch}.\n\n"
            "Run `git pull` in the repo to update."
        )
    return True, f"origin/{branch} has diverged from your local {branch}.\n\nRun `git pull` in the repo to update."


def _latest_pypi_version() -> Optional[str]:
    """The latest version string published on PyPI for this package, or
    None on any failure (offline, PyPI unreachable/down, unexpected
    response, ...) - never raises."""
    request = Request(PYPI_JSON_URL, headers={"User-Agent": f"{PYPI_PACKAGE_NAME}-update-check"})
    try:
        with urlopen(request, timeout=TIMEOUT_S) as response:
            data = json.load(response)
        return data["info"]["version"]
    except (URLError, OSError, TimeoutError, ValueError, KeyError):
        return None


def check_for_major_pip_update() -> tuple[bool, str]:
    """Returns (update_available, message), but only True for an actual
    major-version bump (e.g. 1.x.y -> 2.0.0) on PyPI, compared against this
    install's own __version__. Used for the silent startup check (see
    UpdateCheckWorker), which should stay quiet for routine minor/patch
    releases and only ever interrupt the user for a major one. Works
    regardless of how maskfits was installed (pip or a git clone), unlike
    check_for_updates, which needs an actual git clone to compare against.
    Never raises."""
    latest = _latest_pypi_version()
    if latest is None:
        return False, "Could not check PyPI for updates - no internet connection, or PyPI isn't reachable right now."

    if latest == LOCAL_VERSION:
        return False, f"You're up to date (v{LOCAL_VERSION})."

    local_major = _major_version(LOCAL_VERSION)
    remote_major = _major_version(latest)
    if local_major is None or remote_major is None or remote_major <= local_major:
        return False, "No major version update available."

    return True, (
        f"maskfits v{latest} is available on PyPI (you're on v{LOCAL_VERSION}).\n\n"
        "This is a major version update, which may include breaking changes.\n\n"
        "Run `pip install --upgrade maskfits` to update."
    )


class UpdateCheckWorker(QObject):
    """Runs check_for_major_pip_update() off the GUI thread and signals the
    result back - see MaskFitsApp's silent startup check (run_gui() starts
    this on a plain daemon thread, not a QThread: emitting a Qt signal is
    safe from any thread, Qt auto-queues delivery onto the receiver's own
    thread, and `daemon=True` means a slow/hung network call never blocks
    app exit)."""

    finished = Signal(bool, str)

    def run(self) -> None:
        available, message = check_for_major_pip_update()
        self.finished.emit(available, message)
