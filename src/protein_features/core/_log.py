"""Logging setup shared by the CLI entry points."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Wire up root logging. INFO by default; DEBUG when ``verbose`` is set."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
