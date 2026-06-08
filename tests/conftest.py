"""Shared pytest fixtures for the ABT test suite."""
import random
import pytest

from abt.core import build_tree_from_list, Node


@pytest.fixture
def tree_h2():
    return build_tree_from_list([10, 6, 14, 4, 8, 12, 16])


@pytest.fixture
def tree_h1():
    return build_tree_from_list([20, 10, 30])


@pytest.fixture
def tree_h3():
    return build_tree_from_list([
        100, 50, 150, 25, 75, 125, 175,
        10, 40, 60, 90, 110, 140, 160, 190,
    ])


@pytest.fixture
def tree_float():
    return build_tree_from_list([1.0, 0.5, 1.5, 0.25, 0.75, 1.25, 1.75])


def make_random_perfect_tree(h: int, seed: int = 42) -> Node:
    """Generate a random perfect binary tree of height h."""
    rng = random.Random(seed)
    n = (1 << (h + 1)) - 1
    values = [rng.uniform(0.0, 1000.0) for _ in range(n)]
    return build_tree_from_list(values)


@pytest.fixture
def random_tree_h4():
    return make_random_perfect_tree(4)
