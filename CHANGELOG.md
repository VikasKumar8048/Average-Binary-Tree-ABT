# Changelog

All notable changes to the ABT reference implementation.

## [1.0.0] — 2026-06-01

### Added
- Core `build_abt()` implementing Algorithm 1 from the paper (O(n) BFS)
- `ABT` high-level facade class with `transform()`, `transform_k()`, `iterative_sequence()`
- `validate_perfect()` — perfect binary tree validator (Listing 3)
- `abt.math.theorems` — executable verification of all 7 theorems/properties
- `abt.math.definitions` — Definitions 1–9 as executable Python functions
- `abt.math.proofs` — computational proof demonstrations
- `abt.math.operators` — arithmetic mean, weighted mean, median, probabilistic, geometric, harmonic
- `abt.extensions.WABT` — Weighted Average Binary Tree (Definition 7, Eq. 2)
- `abt.extensions.MBT` — Median Binary Tree (Definition 8, Eq. 3)
- `abt.extensions.PBT` — Probabilistic Binary Tree (Definition 9, Eq. 4)
- `abt.viz` — ASCII, Graphviz DOT, NetworkX, Matplotlib, Plotly visualization
- `abt.benchmarks` — Table III reproduction: ABT vs STA vs APY
- `abt.applications.demos` — five application demos (Section VIII)
- `reproduce_paper.py` — single-command paper reproduction script
- 186 tests (unit, integration, property-based, fuzz) — 100% pass

### Paper
- Vikas Kumar, "Average Binary Tree (ABT): A Novel Formal Tree-to-Tree
  Transformation with Pairwise Sibling Aggregation,"
  IJIRT Volume 13 Issue 1, ISSN 2349-6002, June 2026.
