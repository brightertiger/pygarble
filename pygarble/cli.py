"""Command-line interface: check, score, analyze and calibrate texts."""

import argparse
import io
import json
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

EXIT_OK = 0
EXIT_GARBLED = 1
EXIT_ERROR = 2


class CliError(Exception):
    """User-facing error; message goes to stderr with exit code 2."""


def load_allowlist(path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        raise CliError(f"cannot read allowlist {path}: {error}")
    words = []
    for line in lines:
        word = line.split("#", 1)[0].strip()
        if word:
            words.append(word)
    return words


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="pygarble",
        description="Deterministic gibberish detection for English text.",
        epilog="Input is UTF-8; invalid UTF-8 is replaced with U+FFFD.",
    )
    parser.add_argument(
        "--version", action="version", version=f"pygarble {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    def add_common(sub: argparse.ArgumentParser, default_format: str) -> None:
        sub.add_argument(
            "inputs",
            nargs="*",
            help="files to read, one text per line; '-' or none = stdin",
        )
        sub.add_argument(
            "-t",
            "--text",
            action="append",
            default=None,
            help="evaluate this text instead of reading inputs (repeatable)",
        )
        group = sub.add_mutually_exclusive_group()
        group.add_argument(
            "--profile",
            default=None,
            help="ensemble profile (default english)",
        )
        group.add_argument(
            "--strategy", default=None, help="single strategy name"
        )
        sub.add_argument("--threshold", type=float, default=0.5)
        sub.add_argument(
            "--allowlist", default=None, help="file of words never flagged"
        )
        sub.add_argument(
            "--format",
            choices=["text", "tsv", "jsonl"],
            default=default_format,
        )
        sub.add_argument(
            "--field",
            default=None,
            help="read JSON objects and evaluate this field; echo JSONL",
        )

    add_common(
        subparsers.add_parser("check", help="flag garbled lines"), "text"
    )
    add_common(subparsers.add_parser("score", help="print scores"), "text")
    add_common(
        subparsers.add_parser("analyze", help="print full analyses"), "jsonl"
    )
    cal = subparsers.add_parser("calibrate", help="recommend a threshold")
    cal.add_argument("--garbled", required=True, help="file of garbled lines")
    cal.add_argument("--clean", required=True, help="file of clean lines")
    group = cal.add_mutually_exclusive_group()
    group.add_argument("--profile", default=None)
    group.add_argument("--strategy", default=None)
    cal.add_argument("--allowlist", default=None)
    cal.add_argument("--objective", choices=["f1", "max_fpr"], default="f1")
    cal.add_argument("--max-fpr", type=float, default=None)
    cal.add_argument("--format", choices=["text", "jsonl"], default="text")
    cal.set_defaults(threshold=0.5)
    return parser


def make_detector(args: argparse.Namespace) -> Any:
    from . import EnsembleDetector, GarbleDetector, Strategy

    allowlist = load_allowlist(args.allowlist) if args.allowlist else None
    try:
        if args.strategy:
            return GarbleDetector(
                Strategy(args.strategy),
                threshold=args.threshold,
                allowlist=allowlist,
            )
        return EnsembleDetector(
            threshold=args.threshold,
            profile=args.profile or "english",
            allowlist=allowlist,
        )
    except ValueError as error:
        raise CliError(str(error))


def read_stdin() -> str:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        return sys.stdin.read()
    data: bytes = buffer.read()
    return data.decode("utf-8", errors="replace")


def iter_lines(inputs: List[str]) -> Iterator[str]:
    sources = inputs or ["-"]
    for source in sources:
        if source == "-":
            for line in read_stdin().splitlines():
                yield line
            continue
        try:
            with open(source, encoding="utf-8", errors="replace") as handle:
                for line in handle.read().splitlines():
                    yield line
        except OSError as error:
            raise CliError(f"cannot read {source}: {error}")


def analysis_row(text: str, analysis: Any) -> Dict[str, Any]:
    return {
        "text": text,
        "garbled": analysis.garbled,
        "score": analysis.score,
        "status": analysis.status,
        "profile": analysis.profile,
        "spans": [asdict(span) for span in analysis.spans],
        "signals": [
            {
                "strategy": s.strategy,
                "score": s.score,
                "applicable": s.applicable,
                "reason": s.reason,
            }
            for s in analysis.signals
        ],
    }


def format_row(command: str, fmt: str, text: str, analysis: Any) -> str:
    if fmt == "jsonl":
        return json.dumps(analysis_row(text, analysis), ensure_ascii=False)
    if fmt == "tsv":
        return "\t".join(
            [
                "1" if analysis.garbled else "0",
                f"{analysis.score:.4f}",
                analysis.status,
                text,
            ]
        )
    if command == "score":
        return f"{analysis.score:.4f}\t{text}"
    label = (
        "garbled"
        if analysis.garbled
        else (
            "insufficient"
            if analysis.status == "insufficient_evidence"
            else "clean"
        )
    )
    return f"{label}\t{text}"


def field_problem(obj: Any, field: str) -> Optional[str]:
    if not isinstance(obj, dict):
        return "not a JSON object"
    if field not in obj:
        return f"missing field '{field}'"
    if not isinstance(obj[field], str):
        return f"field '{field}' is not a string"
    return None


def run_texts(args: argparse.Namespace, out: Any, err: Any) -> int:
    detector = make_detector(args)
    any_garbled = False
    had_error = False
    if args.text is not None:
        pairs: Iterable[Tuple[int, Any]] = enumerate(args.text, 1)
    else:
        pairs = enumerate(iter_lines(args.inputs), 1)
    for number, line in pairs:
        if args.field is None:
            analysis = detector.analyze(line)
            any_garbled = any_garbled or analysis.garbled
            out.write(
                format_row(args.command, args.format, line, analysis) + "\n"
            )
            continue
        problem: Optional[str] = None
        try:
            obj = json.loads(line)
        except ValueError as error:
            problem = f"invalid JSON: {error}"
        else:
            problem = field_problem(obj, args.field)
        if problem is not None:
            err.write(f"line {number}: {problem}\n")
            had_error = True
            continue
        value = obj[args.field]
        analysis = detector.analyze(value)
        any_garbled = any_garbled or analysis.garbled
        obj["pygarble"] = analysis_row(value, analysis)
        del obj["pygarble"]["text"]
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if had_error:
        return EXIT_ERROR
    if args.command == "check" and any_garbled:
        return EXIT_GARBLED
    return EXIT_OK


def run_calibrate(args: argparse.Namespace, out: Any) -> int:
    from .calibration import calibrate

    detector = make_detector(args)
    garbled = [line for line in iter_lines([args.garbled]) if line.strip()]
    clean = [line for line in iter_lines([args.clean]) if line.strip()]
    try:
        report = calibrate(
            detector,
            garbled,
            clean,
            objective=args.objective,
            max_false_positive_rate=args.max_fpr,
        )
    except ValueError as error:
        raise CliError(str(error))
    if args.format == "jsonl":
        out.write(json.dumps(asdict(report)) + "\n")
        return EXIT_OK
    out.write("threshold\tprecision\trecall\tf1\tfpr\n")
    for p in report.points:
        out.write(
            f"{p.threshold:.4f}\t{p.precision:.3f}\t{p.recall:.3f}\t"
            f"{p.f1:.3f}\t{p.false_positive_rate:.3f}\n"
        )
    r = report.recommended
    out.write(
        f"recommended threshold: {r.threshold:.4f} "
        f"(f1={r.f1:.3f}, fpr={r.false_positive_rate:.3f})\n"
    )
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Write UTF-8 regardless of locale; unencodable text such as lone
    # surrogates is backslash-escaped instead of raising.
    buffer = getattr(sys.stdout, "buffer", None)
    wrapper: Optional[io.TextIOWrapper] = None
    out: Any = sys.stdout
    if buffer is not None:
        sys.stdout.flush()
        wrapper = io.TextIOWrapper(
            buffer,
            encoding="utf-8",
            errors="backslashreplace",
            line_buffering=True,
        )
        out = wrapper
    try:
        if args.command == "calibrate":
            return run_calibrate(args, out)
        return run_texts(args, out, sys.stderr)
    except CliError as error:
        sys.stderr.write(f"pygarble: {error}\n")
        return EXIT_ERROR
    finally:
        out.flush()
        if wrapper is not None:
            # Detach so collecting the wrapper never closes sys.stdout.
            wrapper.detach()
