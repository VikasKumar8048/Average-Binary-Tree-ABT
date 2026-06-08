#!/usr/bin/env python3
"""
reproduce_paper.py
==================
Reproduce ALL experiments, tables, and test results from the paper
with a single command:

    python reproduce_paper.py

Sections reproduced:
  - Section VI-B: Six test categories with theorem verification
  - Section VI-C: Empirical benchmarks (Table III)
  - Section VII:  Extension demonstrations (WABT, MBT, PBT)
  - Section IV:   Full theorem verification suite

Author : Based on Vikas Kumar, IJIRT Vol 13 Issue 1, June 2026
"""

import math
import sys
import time
from typing import List

# ─── colour helpers ───────────────────────────────────────────────────────────
try:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except Exception:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


def banner(text: str) -> None:
    width = 70
    print(f"\n{BOLD}{CYAN}{'=' * width}{RESET}")
    print(f"{BOLD}{CYAN}{text.center(width)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * width}{RESET}\n")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {YELLOW}→{RESET} {msg}")


# ─── imports ──────────────────────────────────────────────────────────────────

from abt import ABT, Node, build_abt, build_tree_from_list, bfs_values, height, node_count
from abt.math.theorems import verify_all, print_verification_report
from abt.extensions.variants import WABT, MBT, PBT
from abt.viz.visualizer import ascii_tree, side_by_side_ascii


# ─── Section VI-B: Six test categories ───────────────────────────────────────

def section_vi_b() -> bool:
    banner("Section VI-B: Six Test Categories from the Paper")
    all_pass = True

    test_cases = [
        ([10, 6, 14, 4, 8, 12, 16],                                         "h=2 canonical"),
        ([20, 10, 30],                                                        "h=1 minimal"),
        ([100, 50, 150, 25, 75, 125, 175,
          10, 40, 60, 90, 110, 140, 160, 190],                               "h=3 stress"),
        ([1.0, 0.5, 1.5, 0.25, 0.75, 1.25, 1.75],                          "float values"),
    ]

    # ── Tests 1–4: valid trees ────────────────────────────────────────────
    for values, label in test_cases:
        print(f"\n{BOLD}Test: {label}{RESET}")
        T = build_tree_from_list(values)
        T_prime = build_abt(T)
        n = node_count(T)
        h_T = height(T)

        print(ascii_tree(T,       f"T  ({label})"))
        print(ascii_tree(T_prime, f"T' ({label})"))

        # Theorem 1
        exp1 = (n - 1) // 2
        got1 = node_count(T_prime) if T_prime else 0
        p1 = got1 == exp1
        (ok if p1 else fail)(f"Thm 1 |T'|={(n-1)/2:.0f}  got {got1}")

        # Theorem 2
        p2 = height(T_prime) == h_T - 1
        (ok if p2 else fail)(f"Thm 2 h(T')={h_T-1}  got {height(T_prime)}")

        # Theorem 4 for each ABT node
        abt_vals  = bfs_values(T_prime)
        orig_vals = bfs_values(T)
        all_contained = all(
            min(orig_vals) <= v <= max(orig_vals) for v in abt_vals
        )
        (ok if all_contained else fail)(
            f"Thm 4 value containment: [{min(orig_vals):.2g}, {max(orig_vals):.2g}]"
        )

        if not (p1 and p2 and all_contained):
            all_pass = False

    # ── Test 5: single-node edge case ─────────────────────────────────────
    print(f"\n{BOLD}Test: height-0 edge case{RESET}")
    T_single = build_tree_from_list([42.0])
    result = build_abt(T_single)
    p5 = result is None
    (ok if p5 else fail)(f"Height-0 input returns None: got {result!r}")
    if not p5:
        all_pass = False

    # ── Test 6: imperfect tree detection ──────────────────────────────────
    print(f"\n{BOLD}Test: imperfect tree error detection{RESET}")
    root = Node(10.0)
    root.left = Node(5.0)   # only one child → not perfect
    try:
        build_abt(root)
        fail("Expected ValueError — none raised")
        all_pass = False
    except ValueError as e:
        ok(f"ValueError raised correctly: {e}")

    return all_pass


# ─── Section IV: Full theorem verification ───────────────────────────────────

def section_iv() -> bool:
    banner("Section IV: Formal Theorem Verification Suite")
    trees = [
        (build_tree_from_list([10, 6, 14, 4, 8, 12, 16]),                    "h=2"),
        (build_tree_from_list([20, 10, 30]),                                  "h=1"),
        (build_tree_from_list([100, 50, 150, 25, 75, 125, 175,
                                10, 40, 60, 90, 110, 140, 160, 190]),         "h=3"),
        (build_tree_from_list([1.0, 0.5, 1.5, 0.25, 0.75, 1.25, 1.75]),     "float"),
    ]
    all_pass = True
    for T, label in trees:
        passed = print_verification_report(T, label=label)
        if not passed:
            all_pass = False
    return all_pass


# ─── Section VI-C: Empirical benchmarks (Table III) ──────────────────────────

def section_vi_c() -> bool:
    banner("Section VI-C: Empirical Benchmarks (Table III)")
    from abt.benchmarks.runner import ABTBenchmark, scalability_check

    print(f"Running benchmarks (heights 8, 10, 12, 14) — please wait...\n")
    bench = ABTBenchmark(heights=[8, 10, 12, 14], runs=3)
    report = bench.run()

    print("\n" + report.summary_table())
    print("\n" + scalability_check(report))

    # Save CSV
    csv_path = "benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write(report.to_csv())
    ok(f"CSV saved: {csv_path}")

    return True


# ─── Section VII: Extensions ──────────────────────────────────────────────────

def section_vii() -> bool:
    banner("Section VII: Extensions — WABT, MBT, PBT")
    values = [10, 6, 14, 4, 8, 12, 16]
    T = build_tree_from_list(values)

    # WABT
    print(f"\n{BOLD}WABT (omega=(2,1)):{RESET}")
    wabt = WABT(default_weights=(2.0, 1.0))
    T_wabt = wabt.build(T)
    print(ascii_tree(T_wabt, "T' (WABT omega=(2,1))"))
    expected_root = (2.0 * 6 + 1.0 * 14) / 3.0
    p_wabt = math.isclose(T_wabt.val, expected_root)
    (ok if p_wabt else fail)(
        f"WABT root = (2*6+1*14)/3 = {expected_root:.4f}  got {T_wabt.val:.4f}"
    )

    # MBT
    print(f"\n{BOLD}MBT:{RESET}")
    T_mbt = MBT.build(T)
    print(ascii_tree(T_mbt, "T' (MBT)"))
    ok(f"MBT size={node_count(T_mbt)}, height={height(T_mbt)}")

    # PBT
    print(f"\n{BOLD}PBT (p=(0.7, 0.3)):{RESET}")
    pbt = PBT(default_probs=(0.7, 0.3))
    T_pbt = pbt.build(T)
    print(ascii_tree(T_pbt, "T' (PBT p=(0.7,0.3))"))
    expected_pbt_root = 0.7 * 6.0 + 0.3 * 14.0
    p_pbt = math.isclose(T_pbt.val, expected_pbt_root)
    (ok if p_pbt else fail)(
        f"PBT root = 0.7*6+0.3*14 = {expected_pbt_root:.4f}  got {T_pbt.val:.4f}"
    )

    return p_wabt and p_pbt


# ─── Iterative contraction sequence ──────────────────────────────────────────

def iterative_sequence() -> bool:
    banner("Corollary 3: Iterative Contraction Sequence")
    values = [100, 50, 150, 25, 75, 125, 175,
               10, 40, 60, 90, 110, 140, 160, 190]
    tree = ABT.from_list(values)
    seq = tree.iterative_sequence()
    info(f"Initial tree: h={seq[0].height}, n={seq[0].size}")
    for i, t in enumerate(seq):
        print(f"  f^{i}(T): h={t.height}, n={t.size}  values={t.values()}")
    ok(f"Sequence length = {len(seq)} (expected h+1 = {tree.height + 1})")
    return len(seq) == tree.height + 1


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    t_start = time.perf_counter()

    print(f"\n{BOLD}Average Binary Tree (ABT) — Paper Reproducibility Script{RESET}")
    print("Reference: Vikas Kumar, IJIRT Vol 13 Issue 1, June 2026\n")

    results = {}
    results["Section VI-B (6 test categories)"]    = section_vi_b()
    results["Section IV  (theorem verification)"]  = section_iv()
    results["Section VII (extensions: WABT/MBT/PBT)"] = section_vii()
    results["Corollary 3 (iterative sequence)"]    = iterative_sequence()
    results["Section VI-C (benchmarks)"]           = section_vi_c()

    banner("Reproducibility Summary")
    all_pass = True
    for name, passed in results.items():
        (ok if passed else fail)(name)
        if not passed:
            all_pass = False

    elapsed = time.perf_counter() - t_start
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"\n  {BOLD}Overall: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}{RESET}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
