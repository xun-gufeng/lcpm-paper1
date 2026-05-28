#!/usr/bin/env python3
"""
Z5 All-or-Nothing Theorem verification.

Verifies Theorems A, B, C:
- Theorem A: External field all-or-nothing: sum_p delta(sigma, xi) = 9 * delta(k_t, k_ext)
- Theorem B: Temporal coupling all-or-nothing: sum_p delta(sigma_t, sigma_{t+1}) = 9 * delta(k_t, k_{t+1})
- Theorem C: Noise perturbation: Delta_A = 1 - 5*rho/4, Delta_B = Delta_A^2, rho_c = 0.80

Author: Ling Jinguo
"""

import numpy as np

# PALACE_WX mapping
PALACE_WX = np.array([4, 2, 0, 0, 2, 3, 3, 2, 1])


def verify_theorem_A():
    """Verify Theorem A: External field all-or-nothing."""
    print("=== Theorem A Verification: External Field All-or-Nothing ===")

    # Test all combinations of k_t and k_ext
    all_pass = True
    for k_t in range(5):
        for k_ext in range(5):
            sigma = (PALACE_WX + k_t) % 5  # USC with offset k_t
            xi = (PALACE_WX + k_ext) % 5    # External field USC with offset k_ext

            match_count = np.sum(sigma == xi)
            expected = 9 if k_t == k_ext else 0

            if match_count != expected:
                print(f"  FAIL: k_t={k_t}, k_ext={k_ext}: got {match_count}, expected {expected}")
                all_pass = False

    if all_pass:
        print("OK: All 25 (k_t, k_ext) combinations give match count in {0, 9}")
    print()

    # Verify PALACE_WX cancellation
    print("PALACE_WX cancellation:")
    for k_t in range(5):
        for k_ext in range(5):
            diff = ((PALACE_WX + k_t) - (PALACE_WX + k_ext)) % 5
            if not np.all(diff == diff[0]):
                print(f"  FAIL: difference not uniform for k_t={k_t}, k_ext={k_ext}")
    print("OK: sigma - xi = k_t - k_ext (mod 5) for all palaces, independent of PALACE_WX")
    print()


def verify_theorem_B():
    """Verify Theorem B: Temporal coupling all-or-nothing."""
    print("=== Theorem B Verification: Temporal Coupling All-or-Nothing ===")

    all_pass = True
    for k_t in range(5):
        for k_next in range(5):
            sigma_t = (PALACE_WX + k_t) % 5
            sigma_next = (PALACE_WX + k_next) % 5

            match_count = np.sum(sigma_t == sigma_next)
            expected = 9 if k_t == k_next else 0

            if match_count != expected:
                print(f"  FAIL: k_t={k_t}, k_next={k_next}: got {match_count}, expected {expected}")
                all_pass = False

    if all_pass:
        print("OK: All 25 (k_t, k_{t+1}) combinations give match count in {0, 9}")
    print()


def verify_theorem_C():
    """Verify Theorem C: Noise perturbation."""
    print("=== Theorem C Verification: Noise Perturbation ===")

    np.random.seed(42)
    N_trials = 100000
    N_palaces = 9

    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    print(f"{'rho':>6} {'Delta_A(theory)':>16} {'Delta_A(sim)':>12} {'Delta_B(theory)':>16} {'Delta_B(sim)':>12} {'Delta_B/Delta_A^2':>18}")

    for rho in rho_values:
        Delta_A_theory = 1 - 5 * rho / 4
        Delta_B_theory = Delta_A_theory ** 2

        # Simulate external field contrast
        matches_A_same = []  # k_t = k_ext
        matches_A_diff = []  # k_t != k_ext

        for _ in range(N_trials):
            k_t = np.random.randint(5)
            k_ext_same = k_t  # Same
            k_ext_diff = (k_t + np.random.randint(1, 5)) % 5  # Different

            noise = np.random.choice(5, size=N_palaces,
                                      p=[1-rho] + [rho/4]*4 if rho < 1 else [0.2]*5)

            # Same case
            sigma = (PALACE_WX + k_t + noise) % 5
            xi_same = (PALACE_WX + k_ext_same) % 5
            matches_A_same.append(np.mean(sigma == xi_same))

            # Different case
            noise2 = np.random.choice(5, size=N_palaces,
                                       p=[1-rho] + [rho/4]*4 if rho < 1 else [0.2]*5)
            sigma2 = (PALACE_WX + k_t + noise2) % 5
            xi_diff = (PALACE_WX + k_ext_diff) % 5
            matches_A_diff.append(np.mean(sigma2 == xi_diff))

        Delta_A_sim = np.mean(matches_A_same) - np.mean(matches_A_diff)

        # Simulate temporal coupling contrast
        matches_B_same = []
        matches_B_diff = []
        for _ in range(N_trials):
            k_t = np.random.randint(5)
            k_next_same = k_t
            k_next_diff = (k_t + np.random.randint(1, 5)) % 5

            noise_t = np.random.choice(5, size=N_palaces,
                                        p=[1-rho] + [rho/4]*4 if rho < 1 else [0.2]*5)
            noise_next = np.random.choice(5, size=N_palaces,
                                           p=[1-rho] + [rho/4]*4 if rho < 1 else [0.2]*5)

            sigma_t = (PALACE_WX + k_t + noise_t) % 5
            sigma_next_s = (PALACE_WX + k_next_same + noise_next) % 5
            sigma_next_d = (PALACE_WX + k_next_diff + noise_next) % 5

            matches_B_same.append(np.mean(sigma_t == sigma_next_s))
            matches_B_diff.append(np.mean(sigma_t == sigma_next_d))

        Delta_B_sim = np.mean(matches_B_same) - np.mean(matches_B_diff)

        ratio = Delta_B_sim / Delta_A_sim**2 if Delta_A_sim**2 > 1e-10 else float('inf')

        print(f"{rho:6.2f} {Delta_A_theory:16.6f} {Delta_A_sim:12.6f} {Delta_B_theory:16.6f} {Delta_B_sim:12.6f} {ratio:18.6f}")

    print()
    print("Key result: Delta_B = Delta_A^2 (exact identity)")
    print("Critical noise: rho_c = 4/5 = 0.80")
    print()


if __name__ == "__main__":
    verify_theorem_A()
    verify_theorem_B()
    verify_theorem_C()
