"""Command-line interface for fitsedit."""

import argparse
import sys

from astropy.io import fits


def cmd_show(args: argparse.Namespace) -> int:
    with fits.open(args.file) as hdul:
        hdul.info()
        if args.header is not None:
            print(repr(hdul[args.header].header))
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    with fits.open(args.file) as hdul:
        header = hdul[args.hdu].header
        if args.keyword not in header:
            print(f"Keyword {args.keyword!r} not found in HDU {args.hdu}", file=sys.stderr)
            return 1
        print(header[args.keyword])
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    with fits.open(args.file, mode="update") as hdul:
        hdul[args.hdu].header[args.keyword] = args.value
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fitsedit", description="Edit FITS file headers and data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="Show FITS file structure and header")
    show.add_argument("file", help="Path to the FITS file")
    show.add_argument("--header", type=int, default=None, help="Print header for the given HDU index")
    show.set_defaults(func=cmd_show)

    get = subparsers.add_parser("get", help="Get a header keyword value")
    get.add_argument("file", help="Path to the FITS file")
    get.add_argument("keyword", help="Header keyword to read")
    get.add_argument("--hdu", type=int, default=0, help="HDU index (default: 0)")
    get.set_defaults(func=cmd_get)

    set_ = subparsers.add_parser("set", help="Set a header keyword value")
    set_.add_argument("file", help="Path to the FITS file")
    set_.add_argument("keyword", help="Header keyword to set")
    set_.add_argument("value", help="New value for the keyword")
    set_.add_argument("--hdu", type=int, default=0, help="HDU index (default: 0)")
    set_.set_defaults(func=cmd_set)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
