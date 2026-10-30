#!/usr/bin/env python3
"""Unit test for the recompute history (prefill replaces, decode appends)."""
import sys

sys.path.insert(0, "/workspace/ATOM-tritonmoe")
from atom.models.reference_mode import history_after  # noqa: E402

assert history_after([1, 2, 3]) == [1, 2, 3]
print("prefill replaces history OK")

assert history_after([4]) == [1, 2, 3, 4]
print("decode appends OK")

assert history_after([5]) == [1, 2, 3, 4, 5]
print("second decode appends OK")

assert history_after([7, 8]) == [7, 8]
print("new prefill resets history OK")

assert history_after([9]) == [7, 8, 9]
print("decode after new prefill OK")
print("### ALL PASS")
