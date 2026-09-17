"""maskfits package version - resolved from installed package metadata
(itself generated from pyproject.toml's `version` at install time), so this
never has to be hand-kept in sync with pyproject.toml. After bumping the
version there, re-run `pip install -e .` for this to pick it up - normal for
any Python package, not something maskfits-specific.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maskfits")
except PackageNotFoundError:
    # Running from source without an install (e.g. a plain git clone that
    # was never `pip install -e .`'d) - fall back rather than crash import.
    __version__ = "0.0.0+unknown"
