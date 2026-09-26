"""Replace flagged regions. Overlapping or touching findings become one."""

from string import Formatter
from typing import Iterable, List, Tuple

from .findings import Finding, Redaction, sort_key

MODES = ("placeholder", "mask", "partial")
REVEAL_LAST_FOUR = frozenset(
    {"credit_card", "phone", "iban", "ssn_us", "nhs_number", "aadhaar"}
)
PLACEHOLDER_FIELDS = frozenset({"KIND", "kind", "category"})
_TEMPLATE_ERROR = "placeholder may use only {KIND}, {kind} and {category}"

Region = Tuple[int, int, Finding, Tuple[Finding, ...]]


def merge_regions(findings: Iterable[Finding]) -> List[Region]:
    """Sorted, merged regions; each keeps its highest-confidence finding.

    Ties keep the earlier finding, so results never depend on input order.
    Each region also carries every finding merged into it.
    """
    regions: List[Region] = []
    for finding in sorted(findings, key=sort_key):
        if regions and finding.start <= regions[-1][1]:
            start, end, best, members = regions[-1]
            if finding.confidence > best.confidence:
                best = finding
            regions[-1] = (
                start,
                max(end, finding.end),
                best,
                members + (finding,),
            )
        else:
            regions.append((finding.start, finding.end, finding, (finding,)))
    return regions


def validate_placeholder(placeholder: str) -> None:
    """Reject anything but bare {KIND}, {kind} and {category} fields."""
    try:
        parsed = list(Formatter().parse(placeholder))
    except ValueError as error:
        raise ValueError(f"{_TEMPLATE_ERROR}: {error}") from None
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if field not in PLACEHOLDER_FIELDS or spec or conversion:
            raise ValueError(f"{_TEMPLATE_ERROR}: got {{{field}}}")


def reveals_tail(end: int, members: Tuple[Finding, ...]) -> bool:
    """True when the region ends with a revealable finding longer than four
    characters and no other kind of finding covers its last four."""
    tail = end - 4
    ends_revealable = any(
        m.kind in REVEAL_LAST_FOUR and m.end == end and m.end - m.start > 4
        for m in members
    )
    tail_covered = any(
        m.kind not in REVEAL_LAST_FOUR and m.end > tail and m.start < end
        for m in members
    )
    return ends_revealable and not tail_covered


def replacement(
    segment: str,
    finding: Finding,
    mode: str,
    placeholder: str,
    mask: str,
    reveal: bool = False,
) -> str:
    if mode == "placeholder":
        return placeholder.format(
            KIND=finding.kind.upper(),
            kind=finding.kind,
            category=finding.category,
        )
    if mode == "mask":
        return mask * len(segment)
    keep = 4 if reveal and len(segment) > 4 else 0
    return mask * (len(segment) - keep) + segment[len(segment) - keep :]


def render(
    text: str,
    findings: Iterable[Finding],
    mode: str,
    placeholder: str,
    mask_char: str,
) -> Redaction:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if not isinstance(mask_char, str) or len(mask_char) != 1:
        raise ValueError("mask_char must be a single character")
    if mode == "placeholder":
        validate_placeholder(placeholder)
    chosen = tuple(sorted(findings, key=sort_key))
    regions = merge_regions(chosen)
    pieces: List[str] = []
    cursor = 0
    for start, end, best, members in regions:
        pieces.append(text[cursor:start])
        pieces.append(
            replacement(
                text[start:end],
                best,
                mode,
                placeholder,
                mask_char,
                reveals_tail(end, members),
            )
        )
        cursor = end
    pieces.append(text[cursor:])
    return Redaction("".join(pieces), chosen, len(regions))
