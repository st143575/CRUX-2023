"""
CRUX: Claim Frame Extraction and Cross-Claim Relation Tagging System.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crux")
except PackageNotFoundError:
    # Run from a source checkout without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
