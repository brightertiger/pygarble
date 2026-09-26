"""Command-line interface: check, score, analyze, calibrate, scan, redact."""

import argparse
import io
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

EXIT_OK = 0
EXIT_GARBLED = 1
EXIT_ERROR = 2


class CliError(Exception):
    """User-facing error; message goes to stderr with exit code 2."""


def read_lines(handle: Iterable[str]) -> Iterator[str]:
    """Yield lines split on "\n" only, dropping one trailing "\r".

    ``str.splitlines`` would also split on form feeds, information
    separators, NEL and the Unicode line and paragraph separators, hiding
    the very characters some strategies detect. Open handles with
    ``newline="\n"``; ``newline=""`` would still end lines at a bare "\r".
    """
    for chunk in handle:
        if chunk.endswith("\n"):
            chunk = chunk[:-1]
        if chunk.endswith("\r"):
            chunk = chunk[:-1]
        yield chunk


def open_text(path: str) -> Any:
    return open(path, encoding="utf-8", errors="replace", newline="\n")


def load_allowlist(path: str) -> List[str]:
    try:
        with open_text(path) as handle:
            lines = list(read_lines(handle))
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
    cal.add_argument(
        "--objective",
        choices=["f1", "max_fpr"],
        default=None,
        help="f1 (default) or max_fpr (implied by --max-fpr)",
    )
    cal.add_argument(
        "--max-fpr",
        type=float,
        default=None,
        help="highest acceptable false-positive rate",
    )
    cal.add_argument("--format", choices=["text", "jsonl"], default="text")
    cal.set_defaults(threshold=0.5)

    def add_scan_common(sub: argparse.ArgumentParser) -> None:
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
        sub.add_argument(
            "--field",
            default=None,
            help="read JSON objects and process this field; echo JSONL",
        )
        sub.add_argument(
            "--categories", default=None, help="comma list; default all"
        )
        sub.add_argument("--kinds", default=None, help="comma list of kinds")
        sub.add_argument(
            "--exclude-kinds", default=None, help="comma list of kinds"
        )
        sub.add_argument(
            "--locales", default=None, help="comma list of us,uk,in"
        )
        sub.add_argument("--min-confidence", type=float, default=0.5)
        sub.add_argument(
            "--profile", default="english", help="gibberish profile"
        )
        sub.add_argument(
            "--threshold", type=float, default=0.5, help="gibberish threshold"
        )
        sub.add_argument(
            "--allowlist", default=None, help="file of words never gibberish"
        )

    scan_parser = subparsers.add_parser(
        "scan", help="find secrets, PII, profanity and gibberish"
    )
    add_scan_common(scan_parser)
    scan_parser.add_argument(
        "--format", choices=["text", "tsv", "jsonl"], default="text"
    )
    scan_parser.add_argument(
        "--show-matches", action="store_true", help="include matched text"
    )
    redact_parser = subparsers.add_parser("redact", help="print redacted text")
    add_scan_common(redact_parser)
    redact_parser.add_argument(
        "--mode",
        choices=["placeholder", "mask", "partial"],
        default="placeholder",
    )
    redact_parser.add_argument("--placeholder", default="[{KIND}]")
    redact_parser.add_argument("--mask-char", default="*")
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


def read_stdin() -> Iterator[str]:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:
        yield from read_lines(sys.stdin)
        return
    wrapper = io.TextIOWrapper(
        buffer, encoding="utf-8", errors="replace", newline="\n"
    )
    try:
        yield from read_lines(wrapper)
    finally:
        # Detach so collecting the wrapper never closes sys.stdin.
        wrapper.detach()


def iter_lines(inputs: List[str]) -> Iterator[str]:
    sources = inputs or ["-"]
    for source in sources:
        if source == "-":
            yield from read_stdin()
            continue
        try:
            handle = open_text(source)
        except OSError as error:
            raise CliError(f"cannot read {source}: {error}")
        with handle:
            try:
                for line in read_lines(handle):
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

    objective = args.objective
    if objective is None:
        objective = "max_fpr" if args.max_fpr is not None else "f1"
    elif objective == "f1" and args.max_fpr is not None:
        raise CliError("--max-fpr requires --objective max_fpr")
    detector = make_detector(args)
    garbled = [line for line in iter_lines([args.garbled]) if line.strip()]
    clean = [line for line in iter_lines([args.clean]) if line.strip()]
    try:
        report = calibrate(
            detector,
            garbled,
            clean,
            objective=objective,
            max_false_positive_rate=args.max_fpr,
        )
    except ValueError as error:
        raise CliError(str(error))
    if args.format == "jsonl":
        out.write(json.dumps(asdict(report)) + "\n")
        return EXIT_OK
    out.write(f"objective: {report.objective}\n")
    if report.max_false_positive_rate is not None:
        out.write(f"max fpr: {report.max_false_positive_rate}\n")
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


def _split(
    value: Optional[str], option: str, allow_empty: bool = False
) -> Optional[List[str]]:
    """Comma list of names; None when the option was not given. An
    option given with no names is a usage error unless allow_empty."""
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names and not allow_empty:
        raise CliError(f"{option} needs at least one name")
    return names


def make_scanner(args: argparse.Namespace) -> Any:
    from .scanner import DEFAULT_CATEGORIES, Scanner

    categories = _split(args.categories, "--categories")
    kinds = _split(args.kinds, "--kinds")
    exclude = _split(args.exclude_kinds, "--exclude-kinds", allow_empty=True)
    locales = _split(args.locales, "--locales")
    allowlist = load_allowlist(args.allowlist) if args.allowlist else None
    try:
        if args.command == "redact" and args.mode == "placeholder":
            from .redaction import validate_placeholder

            # Up front, so a bad template fails even on empty input.
            validate_placeholder(args.placeholder)
        return Scanner(
            DEFAULT_CATEGORIES if categories is None else categories,
            min_confidence=args.min_confidence,
            kinds=kinds,
            exclude_kinds=exclude or (),
            locales=("us", "uk", "in") if locales is None else locales,
            profile=args.profile,
            threshold=args.threshold,
            allowlist=allowlist,
        )
    except ValueError as error:
        raise CliError(str(error))


def scan_row(text: str, report: Any, show: bool) -> Dict[str, Any]:
    findings = []
    for finding in report.findings:
        row = finding.to_dict()
        if show:
            row["match"] = text[finding.start : finding.end]
        findings.append(row)
    return {"flagged": report.flagged, "findings": findings}


def format_scan(
    fmt: str, text: str, report: Any, show: bool, min_confidence: float
) -> str:
    if fmt == "jsonl":
        row: Dict[str, Any] = {"text": text}
        row.update(scan_row(text, report, show))
        return json.dumps(row, ensure_ascii=False)
    # Text and TSV rows list only the findings that count towards flagged,
    # so a clean row never names a kind; JSONL keeps every finding.
    shown = [
        f
        for f in report.findings
        if f.confidence >= min_confidence or f.category == "gibberish"
    ]
    kinds = ",".join(sorted({f.kind for f in shown}))
    if fmt == "text":
        label = "flagged" if report.flagged else "clean"
        return f"{label}\t{kinds}\t{text}"
    return f"{int(report.flagged)}\t{len(shown)}\t{kinds}\t{text}"


def run_scan(args: argparse.Namespace, out: Any, err: Any) -> int:
    scanner = make_scanner(args)
    redacting = args.command == "redact"
    any_flagged = False
    had_error = False
    if args.text is not None:
        pairs: Iterable[Tuple[int, Any]] = enumerate(args.text, 1)
    else:
        pairs = enumerate(iter_lines(args.inputs), 1)

    def redact_line(value: str) -> str:
        try:
            redaction = scanner.redact(
                value,
                mode=args.mode,
                placeholder=args.placeholder,
                mask_char=args.mask_char,
            )
        except ValueError as error:
            raise CliError(str(error))
        return str(redaction.text)

    for number, line in pairs:
        if args.field is None:
            if redacting:
                out.write(redact_line(line) + "\n")
                continue
            report = scanner.scan(line)
            any_flagged = any_flagged or report.flagged
            out.write(
                format_scan(
                    args.format,
                    line,
                    report,
                    args.show_matches,
                    scanner.min_confidence,
                )
                + "\n"
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
        if redacting:
            obj[args.field] = redact_line(value)
        else:
            report = scanner.scan(value)
            any_flagged = any_flagged or report.flagged
            obj["pygarble"] = scan_row(value, report, args.show_matches)
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if had_error:
        return EXIT_ERROR
    if not redacting and any_flagged:
        return EXIT_GARBLED
    return EXIT_OK


def silence_stdout() -> None:
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        pass


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
        if args.command in ("scan", "redact"):
            return run_scan(args, out, sys.stderr)
        return run_texts(args, out, sys.stderr)
    except CliError as error:
        sys.stderr.write(f"pygarble: {error}\n")
        return EXIT_ERROR
    except BrokenPipeError:
        # The reader went away (e.g. `| head`). Point stdout at devnull so
        # the interpreter's final flush cannot raise again.
        silence_stdout()
        return EXIT_OK
    finally:
        try:
            out.flush()
        except BrokenPipeError:
            silence_stdout()
        if wrapper is not None:
            # Detach so collecting the wrapper never closes sys.stdout.
            wrapper.detach()
