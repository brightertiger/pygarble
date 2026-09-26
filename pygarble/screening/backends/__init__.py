"""Optional local backends. Importing this module installs/loads no extras."""

from typing import Any, Dict, Type

from .detect_secrets import DetectSecretsDetector
from .gitleaks import GitleaksDetector
from .phones import PhoneNumbersDetector
from .stdnum import StdnumDetector

BACKENDS: Dict[str, Type[Any]] = {
    "phonenumbers": PhoneNumbersDetector,
    "stdnum": StdnumDetector,
    "detect-secrets": DetectSecretsDetector,
    "gitleaks": GitleaksDetector,
}

__all__ = [
    "PhoneNumbersDetector",
    "StdnumDetector",
    "DetectSecretsDetector",
    "GitleaksDetector",
]
