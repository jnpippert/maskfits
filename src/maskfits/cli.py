"""Command-line interface for maskfits."""

import argparse
import sys

from astropy.io import fits


def cmd_show(args: argparse.Namespace) -> int:
    with fits.open(args.file) as hdul:
        hdul.info()
        if args.header is not None:
            print(repr(hdul[args.header].header))
    return 0


def cmd_header(args: argparse.Namespace) -> int:
    with fits.open(args.file) as hdul:
        print(repr(hdul[args.extension].header))
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    with fits.open(args.file) as hdul:
        header = hdul[args.extension].header
        if args.keyword not in header:
            print(f"Keyword {args.keyword!r} not found in HDU {args.extension}", file=sys.stderr)
            return 1
        print(header[args.keyword])
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    with fits.open(args.file, mode="update") as hdul:
        hdul[args.extension].header[args.keyword] = args.value
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maskfits", description="Edit FITS file headers and data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="Show FITS file structure and header")
    show.add_argument("file", help="Path to the FITS file")
    show.add_argument("--header", type=int, default=None, help="Print header for the given HDU index")
    show.set_defaults(func=cmd_show)

    header = subparsers.add_parser("header", help="Show the FITS header for one extension")
    header.add_argument("file", help="Path to the FITS file")
    header.add_argument("-x", "--extension", type=int, default=0,
                         help="HDU extension index (default: 0)")
    header.set_defaults(func=cmd_header)

    get = subparsers.add_parser("get", help="Get a header keyword value")
    get.add_argument("file", help="Path to the FITS file")
    get.add_argument("keyword", help="Header keyword to read")
    # --hdu is kept as an alias for backwards compatibility - -x/--extension
    # is the name used everywhere else (the GUI's own -x flag, and header).
    get.add_argument("-x", "--extension", "--hdu", type=int, default=0,
                      help="HDU extension index (default: 0)")
    get.set_defaults(func=cmd_get)

    set_ = subparsers.add_parser("set", help="Set a header keyword value")
    set_.add_argument("file", help="Path to the FITS file")
    set_.add_argument("keyword", help="Header keyword to set")
    set_.add_argument("value", help="New value for the keyword")
    set_.add_argument("-x", "--extension", "--hdu", type=int, default=0,
                       help="HDU extension index (default: 0)")
    set_.set_defaults(func=cmd_set)

    return parser


SUBCOMMANDS = {"show", "header", "get", "set"}

# Friendly CLI names for -c/--colormap -> the actual internal colormap name
# (maskfits.colormaps.COLORMAP_NAMES) - kept as a plain literal table here,
# rather than importing colormaps.py, so the show/get/set subcommands (and
# --help) never pay for its import (colormaps.py pulls in maskfits.theme,
# which pulls in PySide6) just to parse arguments.
COLORMAP_CLI_CHOICES = {
    "grey": "Grayscale",
    "gray": "Grayscale",
    "viridis": "Viridis",
    "inferno": "Inferno",
    "midas": "Midas Rainbow",
    "isopy": "IsoPy",
}

# Friendly CLI names for --scale -> the internal scale_function name
# (linear/log/asinh stretch applied on top of the cut levels).
SCALE_CLI_CHOICES = {"lin": "linear", "log": "log", "asinh": "asinh"}


def _parse_cuts(value: str) -> str:
    """--cuts accepts 'zscale', 'minmax', or a percentile like 99.5 - mapped
    to the internal stretch string ('zscale'/'minmax'/'pct<value>') that
    Entry.apply_stretch already understands. Any percentile value works,
    not just the GUI's five preset buttons - percentile_cuts() itself
    doesn't care which one it's given."""
    normalized = value.strip().lower()
    if normalized in ("zscale", "minmax"):
        return normalized
    try:
        percent = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid cuts value {value!r} - use 'zscale', 'minmax', or a percentile like 99.5"
        ) from None
    if not 0 < percent <= 100:
        raise argparse.ArgumentTypeError(f"invalid cuts percentile {value!r} - must be between 0 and 100")
    return f"pct{percent}"


def build_gui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maskfits", description="Open the maskfits mask-editor GUI")
    parser.add_argument("files", nargs="*", help="FITS files to open")
    parser.add_argument("-z", "--zoom", type=float, default=None,
                         help="initial zoom multiplier relative to fit-to-window (e.g. 2 for 2x)")
    parser.add_argument("-m", "--mode", choices=["s", "e"], default=None,
                         help="initial mask mode: s=satellite, e=ellipse")
    parser.add_argument("--vmin", type=float, default=None,
                         help="initial lower cut level - overrides the value --cuts would otherwise pick")
    parser.add_argument("--vmax", type=float, default=None,
                         help="initial upper cut level - overrides the value --cuts would otherwise pick")
    parser.add_argument("--scale", choices=sorted(SCALE_CLI_CHOICES), default=None,
                         help="initial display stretch function")
    parser.add_argument("--cuts", type=_parse_cuts, default=None,
                         help="initial cut-levels algorithm: zscale, minmax, or a percentile like 99.5")
    parser.add_argument("-c", "--colormap", choices=sorted(COLORMAP_CLI_CHOICES), default=None,
                         help="initial colormap")
    parser.add_argument("-b", "--binning", type=int, default=None, help="initial bin factor (NxN)")
    parser.add_argument("-s", "--smooth", type=int, default=None, help="initial Gaussian smoothing sigma")
    parser.add_argument("-x", "--extension", type=int, default=None,
                         help="initial FITS extension/HDU to load (default: 0) - has no effect on a "
                              "file that doesn't have that extension, it just opens its own first one")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in SUBCOMMANDS:
        parser = build_parser()
        args = parser.parse_args(argv)
        return args.func(args)

    if argv and argv[0] in {"-h", "--help"}:
        print("usage: maskfits [-z ZOOM] [-m {s,e}] [-x N] [--vmin V] [--vmax V] [--scale {lin,log,asinh}]")
        print("                [--cuts CUTS] [-c COLORMAP] [-b N] [-s SIGMA] [IMAGE ...]")
        print("       maskfits {show,header,get,set} ...       header inspection/editing on the command line")
        print()
        print("  -z, --zoom ZOOM      initial zoom multiplier relative to fit-to-window (e.g. 2 for 2x)")
        print("  -m, --mode {s,e}     initial mask mode: s=satellite, e=ellipse")
        print("  -x, --extension N    initial FITS extension/HDU to load (default: 0)")
        print("  --vmin VMIN          initial lower cut level (overrides --cuts)")
        print("  --vmax VMAX          initial upper cut level (overrides --cuts)")
        print("  --scale {lin,log,asinh}  initial display stretch function")
        print("  --cuts CUTS          initial cut levels: zscale, minmax, or a percentile like 99.5")
        print(f"  -c, --colormap {{{','.join(sorted(COLORMAP_CLI_CHOICES))}}}")
        print("                       initial colormap")
        print("  -b, --binning N      initial bin factor (NxN)")
        print("  -s, --smooth SIGMA   initial Gaussian smoothing sigma")
        return 0

    gui_args = build_gui_parser().parse_args(argv)

    from maskfits.gui import run_gui

    return run_gui(
        gui_args.files,
        zoom=gui_args.zoom,
        mode=gui_args.mode,
        vmin=gui_args.vmin,
        vmax=gui_args.vmax,
        scale=SCALE_CLI_CHOICES.get(gui_args.scale) if gui_args.scale else None,
        cuts=gui_args.cuts,
        colormap=COLORMAP_CLI_CHOICES.get(gui_args.colormap) if gui_args.colormap else None,
        binning=gui_args.binning,
        smooth=gui_args.smooth,
        extension=gui_args.extension,
    )


if __name__ == "__main__":
    sys.exit(main())
