import math
import subprocess
import sys

import pytest

from motion_proj.resim.canonical_hash import (
    CanonicalizationError,
    canonical_sha256,
    canonicalize_quaternion_wxyz,
)


def test_key_order_unicode_nfc_and_negative_zero_are_stable():
    left = {"b": -0.0, "a": "e\u0301"}
    right = {"a": "é", "b": 0.0}
    assert canonical_sha256(left) == canonical_sha256(right)


def test_quaternion_sign_is_canonical():
    q = canonicalize_quaternion_wxyz([0.5, -0.5, 0.5, -0.5])
    negated = canonicalize_quaternion_wxyz([-0.5, 0.5, -0.5, 0.5])
    assert q == negated


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_float_fails_closed(value):
    with pytest.raises(CanonicalizationError, match="NaN/Inf"):
        canonical_sha256({"value": value})


def test_hash_is_stable_across_processes():
    code = (
        "from motion_proj.resim.canonical_hash import canonical_sha256;"
        "print(canonical_sha256({'z':1.25,'a':[0.0,'é']}))"
    )
    outputs = [
        subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        for _ in range(3)
    ]
    assert len(set(outputs)) == 1
