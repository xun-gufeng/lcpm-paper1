#!/usr/bin/env python3
"""
Exhaustive enumeration of all 5^9 = 1,953,125 configurations
on the Luoshu lattice to verify Theorem 4 (Perfect Matching).

Result: Only 5 USC configurations satisfy all 31 bonds (15 gen + 16 rest).
No non-USC configuration achieves 31/31.

Author: Ling Jinguo
"""

import numpy as np
from itertools import product

# PALACE_WX mapping: Palace -> Z_5 value
PALACE_WX = {0: 4, 1: 2, 2: 0, 3: 0, 4: 2, 5: 3, 6: 3, 7: 2, 8: 1}

# Generation links (step +1): 15 bonds
GEN_LINKS = [
    (0, 2), (0, 3),   # Water -> Wood
    (1, 5), (1, 6),   # Earth -> Metal
    (2, 8), (3, 8),   # Wood -> Fire
    (4, 5), (4, 6),   # Earth -> Metal
    (5, 0), (6, 0),   # Metal -> Water
    (7, 5), (7, 6),   # Earth -> Metal
    (8, 1), (8, 4), (8, 7),  # Fire -> Earth
]

# Restriction links (step +2): 16 bonds
REST_LINKS = [
    (0, 8),           # Water -> Fire
    (1, 0), (4, 0), (7, 0),  # Earth -> Water
    (2, 1), (2, 4), (2, 7),  # Wood -> Earth
    (3, 1), (3, 4), (3, 7),  # Wood -> Earth
    (5, 2), (5, 3),   # Metal -> Wood
    (6, 2), (6, 3),   # Metal -> Wood
    (8, 5), (8, 6),   # Fire -> Metal
]


def count_satisfied_bonds(config):
    """Count number of generation and restriction bonds satisfied."""
    gen_count = 0
    for i, j in GEN_LINKS:
        if config[j] == (config[i] + 1) % 5:
            gen_count += 1

    rest_count = 0
    for i, k in REST_LINKS:
        if config[k] == (config[i] + 2) % 5:
            rest_count += 1

    return gen_count, rest_count


def usc_config(k):
    """Generate USC configuration with offset k."""
    return tuple((PALACE_WX[p] + k) % 5 for p in range(9))


def verify_theorem4():
    """Exhaustive enumeration to verify Theorem 4."""
    print("=== Theorem 4 Verification: Exhaustive Enumeration ===")
    print(f"Total configurations to check: 5^9 = {5**9}")
    print()

    max_total = 0
    perfect_configs = []

    for idx, config in enumerate(product(range(5), repeat=9)):
        gen_count, rest_count = count_satisfied_bonds(config)
        total = gen_count + rest_count

        if total > max_total:
            max_total = total
            perfect_configs = [(config, gen_count, rest_count)]

        elif total == max_total:
            perfect_configs.append((config, gen_count, rest_count))

        if (idx + 1) % 500000 == 0:
            print(f"  Progress: {idx + 1}/{5**9} ({100*(idx+1)/5**9:.1f}%)")

    print()
    print(f"Maximum satisfied bonds: {max_total} / 31")
    print(f"Number of configurations achieving maximum: {len(perfect_configs)}")
    print()

    # Verify all perfect configs are USC
    usc_set = set(usc_config(k) for k in range(5))
    all_usc = True
    for config, gen_count, rest_count in perfect_configs:
        if config not in usc_set:
            all_usc = False
            print(f"  NON-USC perfect config found: {config}")

    if all_usc:
        print("OK: All perfect configurations are USC states (k=0,1,2,3,4)")
    else:
        print("FAIL: Non-USC perfect configuration exists!")

    # Verify USC states
    print()
    print("USC verification:")
    for k in range(5):
        config = usc_config(k)
        gen_count, rest_count = count_satisfied_bonds(config)
        total = gen_count + rest_count
        status = "OK" if total == 31 else "FAIL"
        print(f"  USC(k={k}): {config} -> gen={gen_count}/15, rest={rest_count}/16, total={total}/31 [{status}]")

    print()
    result = "VERIFIED" if max_total == 31 and len(perfect_configs) == 5 and all_usc else "FAILED"
    print(f"=== Result: Theorem 4 {result} ===")


if __name__ == "__main__":
    verify_theorem4()
