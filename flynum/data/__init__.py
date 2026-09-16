"""Data acquisition for the MaleCNS v1.0 release."""

from .download import (
    ApprovalRequired,
    download_file,
    download_required,
    head_size,
    verify_file,
)
from .manifest import Manifest

__all__ = [
    "ApprovalRequired",
    "download_file",
    "download_required",
    "head_size",
    "verify_file",
    "Manifest",
]
