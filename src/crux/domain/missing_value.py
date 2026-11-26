"""
Constants and utility functions for handling missing values.
"""

EMPTY_NA = "EMPTY_NA"

EMPTY_TOKENS: frozenset[str] = frozenset({EMPTY_NA, ""})

def none2empty_na(value: str | None) -> str:
    """Convert None to "EMPTY_NA"."""
    return value if value is not None else EMPTY_NA

def empty_na_to_none(value: str) -> str | None:
    """Convert EMPTY_NA or empty string to None."""
    value = value.strip()
    return None if value in EMPTY_TOKENS else value