"""Document-oriented screening CLI with findings-only output by default."""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO

from .base import BackendError
from .scanner import CATEGORIES, Scanner


def _csv(value: str) -> List[str]:
    names = [name.strip() for name in value.split(",")]
    if not all(names):
        raise ValueError("name lists must not contain empty entries")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pygarble-screen",
        description="Local secrets, PII and profanity screening of documents.",
    )
    parser.add_argument("command", choices=["scan", "redact"])
    parser.add_argument(
        "inputs", nargs="*", help="UTF-8 files; default: stdin"
    )
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument(
        "--backends", help="phonenumbers,stdnum,detect-secrets,gitleaks"
    )
    parser.add_argument(
        "--backend-options",
        default="{}",
        help='JSON options, e.g. {"phonenumbers":{"region":"GB"}}',
    )
    parser.add_argument("--no-builtin", action="store_true")
    parser.add_argument("--kinds")
    parser.add_argument("--exclude-kinds")
    parser.add_argument("--locales", default="us,uk,in")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-input-length", type=int, default=1_000_000)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["placeholder", "mask", "partial"],
        default="placeholder",
    )
    return parser


def _read(handle: TextIO, limit: int) -> str:
    text = handle.read(limit + 1)
    if len(text) > limit:
        raise ValueError("document exceeds max_input_length")
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_intermixed_args(argv)
    try:
        options = json.loads(args.backend_options)
        if not isinstance(options, dict) or any(
            not isinstance(value, dict) for value in options.values()
        ):
            raise ValueError(
                "backend-options must map backend names to objects"
            )
        scanner = Scanner(
            categories=_csv(args.categories),
            backends=() if args.backends is None else _csv(args.backends),
            backend_options=options,
            builtin=not args.no_builtin,
            kinds=None if args.kinds is None else _csv(args.kinds),
            exclude_kinds=(
                () if args.exclude_kinds is None else _csv(args.exclude_kinds)
            ),
            locales=_csv(args.locales),
            min_confidence=args.min_confidence,
            max_input_length=args.max_input_length,
        )
        flagged = False
        for source in args.inputs or ["-"]:
            if source == "-":
                text = _read(sys.stdin, args.max_input_length)
            else:
                with open(source, encoding="utf-8", newline="") as handle:
                    text = _read(handle, args.max_input_length)
            if args.command == "redact":
                sys.stdout.write(scanner.redact(text, mode=args.mode).text)
            else:
                report = scanner.scan(text)
                flagged = flagged or report.flagged
                row: Dict[str, Any] = report.to_dict()
                if args.include_text:
                    row["text"] = text
                print(json.dumps(row, ensure_ascii=True))
        return int(flagged)
    except (ValueError, TypeError, OSError, ImportError, BackendError):
        # File paths, dependency exceptions and subprocess output can contain
        # credentials. Do not echo them to stderr in a log-facing command.
        print(
            "screening failed: check configuration, optional dependencies "
            "and document size; no clean result is available",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
