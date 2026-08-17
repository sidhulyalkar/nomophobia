from __future__ import annotations
import hashlib


def stable_seed(base_seed: int, *parts: object, modulus: int = 2_000_000_000) -> int:
    """Deterministically derive a library-safe RNG seed."""
    payload = "|".join([str(int(base_seed)), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"s6e8seed").digest()
    return int((int.from_bytes(digest, "little") + int(base_seed)) % modulus)
