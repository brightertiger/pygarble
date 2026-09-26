"""Locations shared by study runners and publication tools."""

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPOSITORY = SCRIPTS.parents[2]
ROOT = REPOSITORY / "paper" / "study"
