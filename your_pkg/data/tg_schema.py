"""Canonical schema definitions for TrustGuard-style datasets."""

from __future__ import annotations

SCHEMA = {
    "SRC": "src",
    "DST": "dst",
    "RATING": "rating",
    "TIME": "timestamp",
}


def label_from_rating(rating: float | int) -> int:
    """Return the default trust/distrust label derived from *rating*.

    Parameters
    ----------
    rating: float | int
        Opinion score from the raw dataset.

    Returns
    -------
    int
        ``1`` if the rating denotes trust (positive), otherwise ``0``.
    """

    return 1 if float(rating) > 0 else 0


__all__ = ["SCHEMA", "label_from_rating"]
