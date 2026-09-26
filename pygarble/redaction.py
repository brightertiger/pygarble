"""Replace flagged regions. Overlapping or touching findings become one."""

from typing import Iterable, List, Tuple

from .findings import Finding, Redaction, sort_key

MODES = ("placeholder", "mask", "partial")
REVEAL_LAST_FOUR = frozenset(
    {"credit_card", "phone", "iban", "ssn_us", "nhs_number", "aadhaar"}
)

Region = Tuple[int, int, Finding]


def merge_regions(findings: Iterable[Finding]) -> List[Region]:
    """Sorted, merged regions; each keeps its highest-confidence finding.

    Ties keep the earlier finding, so results never depend on input order.
    """
    regions: List[Region] = []
    for finding in sorted(findings, key=sort_key):
        if regions and finding.start <= regions[-1][1]:
            start, end, best = regions[-1]
            if finding.confidence > best.confidence:
                best = finding
            regions[-1] = (start, max(end, finding.end), best)
        else:
            regions.append((finding.start, finding.end, finding))
    return regions


def replacement(
    segment: str, finding: Finding, mode: str, placeholder: str, mask: str
) -> str:
    if mode == "placeholder":
        try:
            return placeholder.format(
                KIND=finding.kind.upper(),
                kind=finding.kind,
                category=finding.category,
            )
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "placeholder may use only {KIND}, {kind} and {category}: "
                f"{error}"
            ) from None
    if mode == "mask":
        return mask * len(segment)
    keep = 4 if finding.kind in REVEAL_LAST_FOUR and len(segment) > 4 else 0
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
    chosen = tuple(sorted(findings, key=sort_key))
    regions = merge_regions(chosen)
    pieces: List[str] = []
    cursor = 0
    for start, end, best in regions:
        pieces.append(text[cursor:start])
        pieces.append(
            replacement(text[start:end], best, mode, placeholder, mask_char)
        )
        cursor = end
    pieces.append(text[cursor:])
    return Redaction("".join(pieces), chosen, len(regions))
