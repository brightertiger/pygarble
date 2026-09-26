"""Gitleaks stdin scanning with isolated configuration and exact spans."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from ...findings import Finding, sort_key
from ...validation import finite_number
from ..base import BackendError, check_text

_RULE_ID = re.compile(r"[A-Za-z0-9_.-]{1,120}\Z")


class GitleaksDetector:
    """Requires Gitleaks >=8.19,<9. One process is started per document."""

    category = "secrets"
    kinds = frozenset({"gitleaks_secret"})

    def __init__(
        self,
        executable: str = "gitleaks",
        timeout: float = 10.0,
        config: Optional[str] = None,
    ) -> None:
        self.timeout = finite_number("timeout", timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        resolved = shutil.which(executable)
        if resolved is None:
            raise ImportError("install Gitleaks >=8.19,<9 to use this backend")
        self.executable = str(Path(resolved).resolve())
        self.config = None if config is None else str(Path(config).resolve())
        if self.config is not None and not Path(self.config).is_file():
            raise ValueError("Gitleaks config must be an existing file")

    def _findings(self, text: str, rows: Any) -> Tuple[Finding, ...]:
        if not isinstance(rows, list):
            raise BackendError("invalid Gitleaks report")
        raw = text.encode("utf-8")
        lines = raw.split(b"\n")
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)
        # Translate only reported byte offsets, in one pass below; repeatedly
        # decoding the prefix for each finding is quadratic on dense input.
        spans: List[Tuple[int, int, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise BackendError("invalid Gitleaks finding")
            coords = [
                row.get(key)
                for key in ("StartLine", "EndLine", "StartColumn", "EndColumn")
            ]
            if any(type(value) is not int for value in coords):
                raise BackendError("invalid Gitleaks coordinates")
            first, last, left, right = cast(List[int], coords)
            secret = row.get("Secret")
            rule = row.get("RuleID")
            if (
                not 1 <= first <= last <= len(lines)
                or not 1 <= left <= len(lines[first - 1]) + (first > 1)
                or not 1 <= right <= len(lines[last - 1]) + (last > 1)
                or not isinstance(secret, str)
                or not secret
                or not isinstance(rule, str)
                or not _RULE_ID.fullmatch(rule)
            ):
                raise BackendError("invalid Gitleaks finding")
            value = secret.encode("utf-8")
            # Gitleaks 8.x counts the preceding LF as column one on later
            # lines. Accept ordinary one-based columns too, but only when
            # the reported secret actually exists within the reported span.
            cursor = -1
            for preceding_lf in (True, False):
                start = offsets[first - 1] + left - 1
                end = offsets[last - 1] + right
                if preceding_lf:
                    start -= int(first > 1)
                    end -= int(last > 1)
                segment = raw[start:end]
                cursor = segment.find(value)
                if cursor >= 0:
                    break
            if cursor < 0:
                # Decoded/transformed secrets have no reliable source span.
                # A failed mapping must never yield a clean or partial scan.
                raise BackendError("Gitleaks value has no original-text span")
            while cursor >= 0:
                spans.append(
                    (start + cursor, start + cursor + len(value), rule)
                )
                cursor = segment.find(value, cursor + len(value))
        needed = {point for start, end, _ in spans for point in (start, end)}
        positions: Dict[int, int] = {}
        byte_offset = 0
        for index, char in enumerate(text):
            if byte_offset in needed:
                positions[byte_offset] = index
            byte_offset += len(char.encode("utf-8"))
        positions[byte_offset] = len(text)
        if needed - positions.keys():
            raise BackendError("Gitleaks offset splits a Unicode character")
        return tuple(
            sorted(
                {
                    Finding(
                        self.category,
                        "gitleaks_secret",
                        positions[start],
                        positions[end],
                        0.9,
                        "gitleaks:" + rule,
                    )
                    for start, end, rule in spans
                },
                key=sort_key,
            )
        )

    def detect(self, text: str) -> Tuple[Finding, ...]:
        check_text(text)
        if not text:
            return ()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"GITLEAKS_CONFIG", "GITLEAKS_CONFIG_TOML"}
        }
        try:
            with tempfile.TemporaryDirectory(
                prefix="pygarble-gitleaks-"
            ) as tmp:
                report = Path(tmp) / "report.json"
                command = [
                    self.executable,
                    "stdin",
                    "--no-banner",
                    "--no-color",
                    "--log-level",
                    "error",
                    "--exit-code",
                    "10",
                    "--ignore-gitleaks-allow",
                    "--max-decode-depth",
                    "0",
                    "--max-archive-depth",
                    "0",
                    "--report-format",
                    "json",
                    "--report-path",
                    str(report),
                ]
                if self.config is not None:
                    command += ["--config", self.config]
                result = subprocess.run(
                    command,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    cwd=tmp,
                    env=env,
                    timeout=self.timeout,
                    check=False,
                )
                if result.returncode not in (0, 10):
                    raise BackendError("Gitleaks process failed")
                rows = json.loads(report.read_text(encoding="utf-8"))
                findings = self._findings(text, rows)
                if bool(findings) != (result.returncode == 10):
                    raise BackendError("inconsistent Gitleaks result")
                return findings
        except (
            OSError,
            ValueError,
            TypeError,
            subprocess.SubprocessError,
            BackendError,
        ):
            # Subprocess output and reports may contain actual credentials.
            raise BackendError("Gitleaks scan failed or timed out") from None
