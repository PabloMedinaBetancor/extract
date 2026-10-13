"""Deterministic assignment of documents and pages to splits, bands and random streams.

Everything here is a pure function of a seed and an identifier, so the result never
depends on the order in which documents are met.
"""

import hashlib

import numpy as np


def stable_unit(seed: int, *parts: str) -> float:
    """A number in [0, 1) that depends only on the seed and the parts."""
    return _digest_int(seed, parts) / 2**64


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    return np.random.default_rng(_digest_int(seed, parts))


def assign_split(unit_id: str, seed: int, fractions: dict[str, float]) -> str:
    """Pick the split of a unit (a document, an invoice template) from a hash of its id."""
    if not fractions or any(share < 0 for share in fractions.values()):
        raise ValueError("fractions must be a non-empty mapping of non-negative shares")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must add up to 1, got {sum(fractions.values())}")
    point = stable_unit(seed, "split", unit_id)
    edge = 0.0
    name = ""
    for name, share in sorted(fractions.items()):
        edge += share
        if point < edge:
            return name
    return name


def _digest_int(seed: int, parts: tuple[str, ...]) -> int:
    payload = "\x1f".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
