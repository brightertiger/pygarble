"""Phone extraction using the metadata-only libphonenumber port."""

from typing import Optional, Tuple

from ...findings import Finding
from ..base import check_text, optional_module


class PhoneNumbersDetector:
    category = "pii"
    kinds = frozenset({"phone"})

    def __init__(self, region: Optional[str] = None) -> None:
        self._phones = optional_module("phonenumbers", "phones")
        if region is not None:
            if not isinstance(region, str):
                raise ValueError("region must be an ISO country code")
            region = region.upper()
            if region not in self._phones.SUPPORTED_REGIONS:
                raise ValueError("unsupported phone region")
        self.region = region

    def detect(self, text: str) -> Tuple[Finding, ...]:
        check_text(text)
        return tuple(
            Finding(
                self.category,
                "phone",
                match.start,
                match.end,
                0.9,
                "phonenumbers:valid_number",
            )
            for match in self._phones.PhoneNumberMatcher(
                text,
                self.region,
                leniency=self._phones.Leniency.VALID,
                # Enough attempts for every source position: a fixed small
                # retry budget can silently miss later numbers in a document.
                max_tries=len(text) + 1,
            )
        )
