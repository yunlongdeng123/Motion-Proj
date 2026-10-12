#!/usr/bin/env python3
"""Regression check: indexed removal equals the original strict radius rule."""
import numpy as np
from scipy.spatial import cKDTree


def original(points, actors):
    keep = np.ones(len(points), dtype=bool)
    for point in actors:
        keep[np.linalg.norm(points - point, axis=1) < 0.1] = False
    return keep


def indexed(points, actors):
    nearest, _ = cKDTree(actors).query(points, k=1, eps=0, distance_upper_bound=0.1, workers=4)
    return nearest >= 0.1


actors = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
points = np.array([[0.0, 0.0, 0.0], [0.099999, 0.0, 0.0],
                   [0.1, 0.0, 0.0], [0.100001, 0.0, 0.0], [3.0, 3.0, 3.0]])
expected = np.array([False, False, True, True, True])
assert np.array_equal(original(points, actors), expected)
assert np.array_equal(indexed(points, actors), expected)
rng = np.random.RandomState(20261011)
points, actors = rng.rand(4000, 3), rng.rand(257, 3)
assert np.array_equal(original(points, actors), indexed(points, actors))
print("PASS: strict radius boundary, duplicate actor points, 4,000-point brute-force parity")
