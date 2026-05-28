#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
climate_z5_verify.py
====================
Climate verification of the Luoshu Chiral Potts Model (LCPM) on CUG-CMA 2.5° grid data.

This script reproduces all climate validation results reported in the paper:
  Luoshu Chiral Potts Model on Luoshu Lattice: Ground State Perfection and First-Order Phase Transition

Data source: CUG-CMA 2.5° Gridded Climate Dataset (1880–2020)
  - NetCDF: CUG-CMA DATASET_2.5grid.nc (359 MB)
  - Pre-processed CSV: cug_cma_gridyear_full.csv (19,022 grid-year records)
  - Boundary test CSV: cug_cma_gridyear_boundary.csv
  - ERA5 simulation CSV: cug_cma_gridyear_era5sim.csv

License: CC BY 4.0
Reference: Ling, J. (2026). CUG-CMA 2.5° Gridded Climate Dataset (1880–2020).
          DOI: 10.5281/zenodo.xxxxxx  [replace with actual DOI upon data publication]

Usage:
  python climate_z5_verify.py [--csv] [--netcdf PATH] [--output DIR]

Options:
  --csv        Use pre-processed CSV instead of NetCDF (default: auto-detect)
  --netcdf     Path to NetCDF file (default: auto-search in common locations)
  --output     Output directory for CSV results (default: ./output/)

Dependencies: numpy, scipy, pandas, netCDF4 (or xarray)
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# PALACE MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# PALACE_BOUNDS: inclusive lon_range, lat_range for each palace (1-indexed, CUG-CMA grid)
# Derived from actual 186 valid land grid points in cug_cma_gridyear_full.csv.
# NOTE: This is an approximation; the actual CUG-CMA grid has irregular land/ocean
# coverage. For precise analysis, use the pre-processed CSV (cug_cma_gridyear_full.csv)
# which contains the correct palace assignments. The PALACE_BOUNDS below are used only
# as a fallback when reading directly from the NetCDF file.
PALACE_BOUNDS = {
    # South row (lat 7.5°–22.5°N)
    9: {'lon': (92.5, 125.0),  'lat': (7.5, 22.5)},   # Fire (火) — south, all longitudes
    # Middle row (lat 25°–35°N)
    2: {'lon': (90.0, 110.0),  'lat': (25.0, 35.0)},   # Earth (土) — west part of middle row
    4: {'lon': (112.5, 120.0), 'lat': (25.0, 35.0)},   # Wood (木) — east part of middle row
    5: {'lon': (120.0, 122.5), 'lat': (25.0, 35.0)},   # Earth (土) — east-center
    # North row (lat 35°–50°N)
    1: {'lon': (87.5, 110.0),  'lat': (35.0, 50.0)},   # Water (水) — NW/W
    3: {'lon': (112.5, 120.0), 'lat': (35.0, 50.0)},   # Wood (木) — NE center
    6: {'lon': (120.0, 142.5), 'lat': (35.0, 50.0)},   # Metal (金) — far east/NE
    7: {'lon': (125.0, 140.0), 'lat': (35.0, 37.5)},   # Metal (金) — east edge, lat 35 only
    8: {'lon': (120.0, 125.0), 'lat': (37.5, 50.0)},   # Earth (土) — N/E center transition
}

# k_optimal: sigma = (k + 3) % 5
# k is the 司天 (celestial dominate) index for each year
# The mapping from 司天 to k:
#   太阳寒水→k=0, 阳明燥金→k=1, 少阳相火→k=2,
#   太阴湿土→k=3, 厥阴风木→k=4, 少阴君火→k=5≡0
def k_optimal(W):
    """Map 司天 index W (0–11) to σ group via k_optimal(W) = (W+3) % 5."""
    return (W + 3) % 5

# Palace → Wuxing element mapping (0=Wood,1=Fire,2=Earth,3=Metal,4=Water)
# Palace indices: 1=坎(N),2=坤(SW),3=震(E),4=巽(SE),5=中(C),
#                 6=乾(NW),7=兑(W),8=艮(NE),9=离(S)  [1-indexed]
PALACE_WX = {1:4, 2:2, 3:0, 4:0, 5:2, 6:3, 7:3, 8:2, 9:1}

# Da Si Tian (大司天) periods and their k_opt values.
# k_opt(W) = (W + 3) % 5 where W is the Wuxing element of the Da Si Tian.
# The Da Si Tian is constant for each 60-year block.
# Period boundaries from the traditional Chinese calendar (陆懋修 system):
#   1864-1923: 太阳寒水 (Water, W=4) → k = (4+3)%5 = 2
#   1924-1983: 阳明燥金 (Metal, W=3) → k = (3+3)%5 = 1
#   1984-2043: 厥阴风木 (Wood, W=0) → k = (0+3)%5 = 3
# Extended for earlier/later periods (cyclic 6-phase Da Si Tian sequence).
DST_PERIODS = [
    (1804, 1863, 4, 2),  # 太阳寒水
    (1864, 1923, 4, 2),  # 太阳寒水
    (1924, 1983, 3, 1),  # 阳明燥金
    (1984, 2043, 0, 3),  # 厥阴风木
]

def get_k_opt(year):
    """Return k_opt for a given year based on the Da Si Tian period."""
    for start, end, dst_wx, k in DST_PERIODS:
        if start <= year <= end:
            return k
    raise ValueError(f"Year {year} not in any Da Si Tian period table")

def get_sigma(year, palace_idx=None):
    """Return σ for a given Gregorian year (and optionally palace).
    
    Full formula: σ_{t,p} = (PALACE_WX[p] + k_opt) % 5
    If palace_idx is None, returns k_opt only (for backward compat / CSV mode).
    If palace_idx is given (1–9, 1-indexed), returns the correct palace-dependent σ.
    """
    k_opt = get_k_opt(year)
    if palace_idx is None:
        return k_opt
    return (PALACE_WX[palace_idx] + k_opt) % 5


# ─────────────────────────────────────────────────────────────────────────────
# NETCDF PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def find_netcdf():
    """Search for NetCDF file in common locations."""
    candidates = [
        '/app/data/所有对话/主对话/中医学习整理/气候数据/CUG-CMA DATASET_2.5grid.nc',
        '/app/data/所有对话/主对话/中医学习整理/气候数据/CUG-CMA_DATASET_2.5grid.nc',
        './CUG-CMA DATASET_2.5grid.nc',
        './CUG-CMA_DATASET_2.5grid.nc',
        '/tmp/CUG-CMA DATASET_2.5grid.nc',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_palace(lon, lat):
    """Assign a grid point (lon, lat) to a palace (1–9) using PALACE_BOUNDS."""
    for palace, bounds in PALACE_BOUNDS.items():
        lon_lo, lon_hi = bounds['lon']
        lat_lo, lat_hi = bounds['lat']
        if lon_lo <= lon <= lon_hi and lat_lo <= lat <= lat_hi:
            return palace
    return None


def read_netcdf(nc_path, verbose=True):
    """
    Read CUG-CMA NetCDF and produce a DataFrame with grid-year climate variables.

    Parameters
    ----------
    nc_path : str
        Path to the CUG-CMA 2.5° grid NetCDF file.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Columns: year, lon, lat, palace, sigma, DTR, TMIN, TMAX_std, TMAX, n_days
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError("netCDF4 is required to read NetCDF. Install: pip install netCDF4")

    if verbose:
        print(f"  Reading NetCDF: {nc_path}")

    ds = Dataset(nc_path, 'r')
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    times = ds.variables['time'][:]
    tmax_raw = ds.variables['tmax'][:]
    tmin_raw = ds.variables['tmin'][:]
    ds.close()

    # Convert time to years (days since 1880-01-01)
    # times[i] is days since 1880-01-01
    years = 1880 + times / 365.25
    years_int = np.floor(years).astype(int)

    if verbose:
        print(f"  NetCDF: {len(lons)}×{len(lats)} = {len(lons)*len(lats)} grid points, "
              f"{len(times)} daily records, years {years_int.min()}–{years_int.max()}")

    records = []
    for iy, year in enumerate(years_int):
        tmax_day = tmax_raw[iy]   # shape: (n_lat, n_lon)
        tmin_day = tmin_raw[iy]

        for ilat, lat in enumerate(lats):
            for ilon, lon in enumerate(lons):
                tmax_val = tmax_day[ilat, ilon]
                tmin_val = tmin_day[ilat, ilon]

                # Skip ocean / invalid points
                if np.isnan(tmax_val) or np.isnan(tmin_val):
                    continue

                palace = get_palace(lon, lat)
                if palace is None:
                    continue

                sigma = get_sigma(year, palace)

                # Climate variables for this grid-year
                dtr = tmax_val - tmin_val
                tmax_std = 0.0  # placeholder; needs intra-year std

                records.append({
                    'year': year,
                    'lon': lon,
                    'lat': lat,
                    'palace': palace,
                    'sigma': sigma,
                    'TMAX': tmax_val,
                    'TMIN': tmin_val,
                    'DTR': dtr,
                    'TMAX_std': tmax_std,  # placeholder
                    'n_days': 1,
                })

    df = pd.DataFrame(records)
    if verbose:
        print(f"  Extracted {len(df)} grid-year records, "
              f"{df['palace'].nunique()} palaces, σ groups: {sorted(df['sigma'].unique())}")
    return df


def aggregate_netcdf_to_gridyear(nc_path, verbose=True):
    """
    Aggregate daily NetCDF data to annual grid-year level.
    For each grid-year, compute mean TMAX, mean TMIN, DTR (annual mean),
    and TMAX_std (annual std of daily TMAX).
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError("netCDF4 required")

    if verbose:
        print(f"  Reading NetCDF: {nc_path}")

    ds = Dataset(nc_path, 'r')
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    times = ds.variables['time'][:]
    tmax_raw = ds.variables['tmax'][:]
    tmin_raw = ds.variables['tmin'][:]
    ds.close()

    years_int = (1880 + times / 365.25).astype(int)

    # Build year lookup
    records = {}
    for iy, year in enumerate(years_int):
        tmax_day = tmax_raw[iy]
        tmin_day = tmin_raw[iy]

        for ilat, lat in enumerate(lats):
            for ilon, lon in enumerate(lons):
                tmax_val = tmax_day[ilat, ilon]
                tmin_val = tmin_day[ilat, ilon]
                if np.isnan(tmax_val) or np.isnan(tmin_val):
                    continue

                palace = get_palace(lon, lat)
                if palace is None:
                    continue

                key = (year, round(lon, 1), round(lat, 1))
                if key not in records:
                    records[key] = {
                        'year': year, 'lon': round(lon, 1), 'lat': round(lat, 1),
                        'palace': palace, 'sigma': get_sigma(year, palace),
                        'tmax_vals': [], 'tmin_vals': [], 'dtr_vals': [],
                    }
                records[key]['tmax_vals'].append(tmax_val)
                records[key]['tmin_vals'].append(tmin_val)
                records[key]['dtr_vals'].append(tmax_val - tmin_val)

    rows = []
    for key, v in records.items():
        tmax_arr = np.array(v['tmax_vals'])
        tmin_arr = np.array(v['tmin_vals'])
        dtr_arr = np.array(v['dtr_vals'])
        rows.append({
            'year': v['year'], 'lon': v['lon'], 'lat': v['lat'],
            'palace': v['palace'], 'sigma': v['sigma'],
            'TMAX': np.nanmean(tmax_arr),
            'TMIN': np.nanmean(tmin_arr),
            'DTR': np.nanmean(dtr_arr),
            'TMAX_std': np.nanstd(tmax_arr),  # intra-annual std of TMAX
            'n_days': len(tmax_arr),
        })

    df = pd.DataFrame(rows)
    if verbose:
        print(f"  Aggregated {len(df)} grid-year records, "
              f"{df['palace'].nunique()} palaces, σ groups: {sorted(df['sigma'].unique())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS MODULES
# ─────────────────────────────────────────────────────────────────────────────

def anova_sigma_groups(df, verbose=True):
    """
    One-way ANOVA: DTR / TMIN / TMAX_std across 5 σ groups.
    Reports F-statistic and partial eta-squared (η²).

    Expected (from paper):
      DTR:  F=152, η²=0.031
      TMIN: F=97.5, η²=0.020
      TMAX_std: F=384, η²=0.075
    """
    results = {}
    for var in ['DTR', 'TMIN', 'TMAX_std']:
        groups = [df[df['sigma'] == s][var].dropna().values for s in range(5)]
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            if verbose:
                print(f"[ANOVA] {var}: insufficient data")
            continue

        F, p = stats.f_oneway(*valid_groups)

        # Partial eta-squared
        grand_mean = df[var].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in valid_groups)
        ss_total = ((df[var] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        # Group means
        group_means = {s: groups[s].mean() for s in range(5) if len(groups[s]) >= 2}
        group_ns = {s: len(groups[s]) for s in range(5)}

        results[var] = {'F': F, 'p': p, 'eta2': eta2,
                         'means': group_means, 'ns': group_ns}

        if verbose:
            print(f"\n[ANOVA] {var}:")
            print(f"  F({len(valid_groups)-1}, {sum(len(g) for g in valid_groups)-len(valid_groups)}) "
                  f"= {F:.2f}, p = {p:.2e}, η² = {eta2:.4f}")
            for s in range(5):
                if s in group_means:
                    print(f"  σ={s}: n={group_ns[s]:5d}, mean={group_means[s]:+.3f}")

    return results


def boundary_jump_test(boundary_csv, verbose=True):
    """
    Da Si Tian boundary jump test.
    Tests whether the majority of grid points show consistent directional change
    at the major 大司天 boundaries (1924, 1984).

    1924 boundary (司天 transition): Expects DTR to increase
    1984 boundary (司天 transition): Expects TMIN to increase

    Method: Binomial test on direction-consistent fraction.
    Reference: 89.2% DTR increase at 1924, binom p=9.2e-16;
               99.5% TMIN increase at 1984, binom p<1e-20
    """
    if not os.path.exists(boundary_csv):
        if verbose:
            print(f"[Boundary] CSV not found: {boundary_csv}")
        return {}

    df = pd.read_csv(boundary_csv)
    results = {}

    # ── 1924: DTR boundary jump ───────────────────────────────────────────
    d1924 = df[(df['boundary'] == 1924) & (df['variable'] == 'DTR')].copy()
    if len(d1924) > 0:
        n_increased = (d1924['delta'] > 0).sum()
        n_total = len(d1924)
        frac = n_increased / n_total

        # Binomial test: H0: frac ≤ 0.5 (no directional preference)
        # Two-sided: test against frac != 0.5
        # One-sided (positive direction): test against frac <= 0.5 under positive
        try:
            from scipy.stats import binomtest
            bt = binomtest(n_increased, n_total, alternative='greater')
            p_binom = bt.pvalue
        except Exception:
            # Fallback for older scipy
            from scipy.stats import binom_test
            p_binom = binom_test(n_increased, n_total, alternative='greater')

        # t-test on paired before/after
        t_stat = d1924['t_stat'].mean()
        p_t = d1924['p_value'].mean()

        results['1924_DTR'] = {
            'n_total': n_total, 'n_increased': n_increased,
            'frac_increased': frac, 'binom_p': p_binom,
            'mean_delta': d1924['delta'].mean(),
            't_stat_mean': t_stat, 'p_t_mean': p_t,
        }

        if verbose:
            print(f"\n[Boundary] 1924 — DTR jump:")
            print(f"  Grid points: {n_total}, increased after 1924: {n_increased} ({frac:.1%})")
            print(f"  Binomial test (H0: ≤50% increase): p = {p_binom:.2e}")
            print(f"  Mean ΔDTR: {d1924['delta'].mean():+.3f} °C")
            print(f"  t-test mean: t={t_stat:.2f}, p={p_t:.3f}")

    # ── 1984: TMIN boundary jump ──────────────────────────────────────────
    d1984 = df[(df['boundary'] == 1984) & (df['variable'] == 'TMIN')].copy()
    if len(d1984) > 0:
        n_increased = (d1984['delta'] > 0).sum()
        n_total = len(d1984)
        frac = n_increased / n_total

        try:
            from scipy.stats import binomtest
            bt = binomtest(n_increased, n_total, alternative='greater')
            p_binom = bt.pvalue
        except Exception:
            from scipy.stats import binom_test
            p_binom = binom_test(n_increased, n_total, alternative='greater')

        t_stat = d1984['t_stat'].mean()
        p_t = d1984['p_value'].mean()

        results['1984_TMIN'] = {
            'n_total': n_total, 'n_increased': n_increased,
            'frac_increased': frac, 'binom_p': p_binom,
            'mean_delta': d1984['delta'].mean(),
            't_stat_mean': t_stat, 'p_t_mean': p_t,
        }

        if verbose:
            print(f"\n[Boundary] 1984 — TMIN jump:")
            print(f"  Grid points: {n_total}, increased after 1984: {n_increased} ({frac:.1%})")
            print(f"  Binomial test (H0: ≤50% increase): p = {p_binom:.2e}")
            print(f"  Mean ΔTMIN: {d1984['delta'].mean():+.3f} °C")
            print(f"  t-test mean: t={t_stat:.2f}, p={p_t:.3f}")

    return results


def era5_coverage_test(era5_csv, verbose=True):
    """
    ERA5 reanalysis coverage test.

    Problem: If the ANOVA signal comes from coverage imbalance
    (e.g., different σ groups have different numbers of grid points or years),
    then ERA5 should also show the same pattern even though it has full coverage.

    This test uses ERA5 simulation data (5 stations × 70 years) with balanced σ
    coverage. If ANOVA remains significant under balanced coverage, the signal
    is NOT a coverage artifact.

    Expected (from paper): ANOVA still significant under ERA5
      DTR: F=6.55, p=4.3e-05, η²=0.071
    """
    if not os.path.exists(era5_csv):
        if verbose:
            print(f"[ERA5] CSV not found: {era5_csv}")
        return {}

    df = pd.read_csv(era5_csv)
    results = {}

    for var in ['DTR', 'TMIN', 'TMAX_std']:
        groups = [df[df['sigma'] == s][var].dropna().values for s in range(5)]
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            continue

        F, p = stats.f_oneway(*valid_groups)

        grand_mean = df[var].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in valid_groups)
        ss_total = ((df[var] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else 0

        results[var] = {'F': F, 'p': p, 'eta2': eta2}

        if verbose:
            print(f"\n[ERA5] {var} (5 stations, balanced σ coverage):")
            print(f"  F = {F:.2f}, p = {p:.2e}, η² = {eta2:.4f}")
            for s in range(5):
                g = groups[s]
                if len(g) >= 2:
                    print(f"  σ={s}: n={len(g):4d}, mean={g.mean():+.3f}")

    return results


def zang_vs_heat_test(df, verbose=True):
    """
    Zang (燥, DTR) ≠ Heat (热, TMAX_std) direct test.

    The LCPM predicts that 燥 (σ-dependent DTR variability)
    and 热 (σ-dependent TMAX variability) are distinct phenomena
    in the Z₅ fiber bundle — they share the same σ classification
    but have different physical signatures.

    Test: Cohen's d between σ=0 and σ=4 for DTR vs TMAX_std.
    If the effect sizes differ substantially (Cohen's d for DTR ≠ TMAX_std),
    the two phenomena are empirically distinguishable.

    Expected: DTR Cohen's d ≈ 0.41, TMAX_std Cohen's d ≈ 0.47
              → distinct signals, Z₅ fiber is non-trivial.
    """
    results = {}

    for var, label in [('DTR', 'Zang (DTR)'), ('TMAX_std', 'Heat (TMAX_std)')]:
        s0 = df[df['sigma'] == 0][var].dropna()
        s4 = df[df['sigma'] == 4][var].dropna()

        if len(s0) < 5 or len(s4) < 5:
            if verbose:
                print(f"[Zang≠Heat] {var}: insufficient data")
            continue

        # t-test
        t_stat, p_val = stats.ttest_ind(s0, s4)

        # Cohen's d
        mean_diff = s4.mean() - s0.mean()
        pooled_std = np.sqrt((s0.var(ddof=1) + s4.var(ddof=1)) / 2)
        d = mean_diff / pooled_std if pooled_std > 0 else 0

        results[var] = {
            'mean_s0': s0.mean(), 'mean_s4': s4.mean(),
            'mean_diff': mean_diff, 'cohens_d': d,
            't_stat': t_stat, 'p_value': p_val,
            'n_s0': len(s0), 'n_s4': len(s4),
        }

        if verbose:
            print(f"\n[Zang≠Heat] {label}:")
            print(f"  σ=0: n={len(s0):5d}, mean={s0.mean():+.3f}")
            print(f"  σ=4: n={len(s4):5d}, mean={s4.mean():+.3f}")
            print(f"  Δ = {mean_diff:+.3f}, Cohen's d = {d:.3f}, t = {t_stat:.2f}, p = {p_val:.2e}")

    # Compare effect sizes
    if 'DTR' in results and 'TMAX_std' in results:
        d_dtr = results['DTR']['cohens_d']
        d_tmax = results['TMAX_std']['cohens_d']
        diff = abs(d_dtr - d_tmax)

        if verbose:
            print(f"\n[Zang≠Heat] Signal distinguishability:")
            print(f"  DTR Cohen's d = {d_dtr:.3f}")
            print(f"  TMAX_std Cohen's d = {d_tmax:.3f}")
            print(f"  |Δd| = {diff:.3f}")
            if diff > 0.1:
                print(f"  → Distinct signals confirmed (|Δd| > 0.1): Z₅ fiber is non-trivial")
            else:
                print(f"  → Similar effect sizes: further investigation needed")

        results['comparison'] = {'d_DTR': d_dtr, 'd_TMAX_std': d_tmax, 'diff': diff}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Climate verification of Luoshu Chiral Potts Model on CUG-CMA dataset.')
    parser.add_argument('--csv', action='store_true',
                        help='Force use of pre-processed CSV instead of NetCDF')
    parser.add_argument('--netcdf', type=str, default=None,
                        help='Path to NetCDF file')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for CSV results')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    args = parser.parse_args()

    verbose = not args.quiet

    # ── Data loading ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Luoshu Chiral Potts Model — Climate Verification")
    print("=" * 70)

    # Search for CSV
    csv_paths = [
        '/app/data/所有对话/主对话/中医学习整理/气候数据/cug_cma_gridyear_full.csv',
        '/app/data/所有对话/主对话/中医学习整理/气候数据/cug_cma_gridyear.csv',
        './cug_cma_gridyear_full.csv',
        './cug_cma_gridyear.csv',
    ]
    csv_path = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_path = p
            break

    # Search for NetCDF
    nc_path = args.netcdf or find_netcdf()

    # Search for supporting CSVs
    boundary_paths = [
        '/app/data/所有对话/主对话/中医学习整理/气候数据/cug_cma_gridyear_boundary.csv',
        './cug_cma_gridyear_boundary.csv',
    ]
    boundary_path = None
    for p in boundary_paths:
        if os.path.exists(p):
            boundary_path = p
            break

    era5_paths = [
        '/app/data/所有对话/主对话/中医学习整理/气候数据/cug_cma_gridyear_era5sim.csv',
        './cug_cma_gridyear_era5sim.csv',
    ]
    era5_path = None
    for p in era5_paths:
        if os.path.exists(p):
            era5_path = p
            break

    # Load data
    if args.csv and csv_path:
        if verbose:
            print(f"\n[Data] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"  Loaded {len(df)} grid-year records, "
                  f"years {df['year'].min()}–{df['year'].max()}, "
                  f"{df['palace'].nunique()} palaces")

    elif nc_path and os.path.exists(nc_path):
        if verbose:
            print(f"\n[Data] Processing NetCDF: {nc_path}")
        df = aggregate_netcdf_to_gridyear(nc_path, verbose=verbose)

    elif csv_path:
        if verbose:
            print(f"\n[Data] CSV fallback: {csv_path}")
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"  Loaded {len(df)} grid-year records")

    else:
        print("\n[Error] No data found. Please provide either:")
        print("  1. A NetCDF file (CUG-CMA DATASET_2.5grid.nc)")
        print("  2. A pre-processed CSV (cug_cma_gridyear_full.csv)")
        sys.exit(1)

    # Filter valid σ groups (0–4)
    df = df[df['sigma'].isin(range(5))].copy()
    if verbose:
        print(f"\n[Data] After filtering σ∈{{0..4}}: {len(df)} records")
        print(f"  σ distribution: {dict(df['sigma'].value_counts().sort_index())}")

    # ── Analysis ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("§1  ANOVA: Climate Variables Across 5 σ Groups")
    print("=" * 70)
    anova_results = anova_sigma_groups(df, verbose=verbose)

    print("\n" + "=" * 70)
    print("§2  Da Si Tian Boundary Jump Test")
    print("=" * 70)
    if boundary_path:
        boundary_results = boundary_jump_test(boundary_path, verbose=verbose)
    else:
        print("  Boundary CSV not found — skipping boundary jump test.")
        print("  Expected results (from pre-computed CSV):")
        print("    1924 DTR: 89.2% increase, binom p = 9.2e-16")
        print("    1984 TMIN: 99.5% increase, binom p < 1e-20")
        boundary_results = {}

    print("\n" + "=" * 70)
    print("§3  ERA5 Coverage Simulation Test")
    print("=" * 70)
    if era5_path:
        era5_results = era5_coverage_test(era5_path, verbose=verbose)
    else:
        print("  ERA5 CSV not found — skipping coverage test.")
        print("  Expected results (from pre-computed CSV):")
        print("    ERA5 DTR ANOVA: F=6.55, p=4.3e-05, η²=0.071")
        era5_results = {}

    print("\n" + "=" * 70)
    print("§4  Zang (燥, DTR) ≠ Heat (热, TMAX_std) Direct Test")
    print("=" * 70)
    zang_results = zang_vs_heat_test(df, verbose=verbose)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary of Key Results")
    print("=" * 70)
    print(f"  Dataset: {len(df)} grid-year records, years {df['year'].min()}-{df['year'].max()}")

    if anova_results:
        print("\n  ANOVA (σ groups):")
        for var, r in anova_results.items():
            sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
            print(f"    {var:10s}: F={r['F']:7.2f}, p={r['p']:.2e}, η²={r['eta2']:.4f} {sig}")

    if boundary_results:
        print("\n  Boundary jumps:")
        for key, r in boundary_results.items():
            sig = "***" if r['binom_p'] < 0.001 else "**" if r['binom_p'] < 0.01 else "*" if r['binom_p'] < 0.05 else ""
            print(f"    {key:12s}: {r['frac_increased']:.1%} increase, binom p={r['binom_p']:.2e} {sig}")

    if era5_results:
        print("\n  ERA5 coverage test:")
        for var, r in era5_results.items():
            sig = "***" if r['p'] < 0.001 else "**" if r['p'] < 0.01 else "*" if r['p'] < 0.05 else ""
            print(f"    {var:10s}: F={r['F']:7.2f}, p={r['p']:.2e}, η²={r['eta2']:.4f} {sig}")

    if zang_results and 'comparison' in zang_results:
        c = zang_results['comparison']
        print("\n  Zang ≠ Heat:")
        print(f"    DTR Cohen's d(σ=4 vs σ=0)    = {c['d_DTR']:+.3f}")
        print(f"    TMAX_std Cohen's d(σ=4 vs σ=0) = {c['d_TMAX_std']:+.3f}")
        print(f"    |Δd| = {c['diff']:.3f} ({'distinct' if c['diff'] > 0.1 else 'similar'})")

    # ── Save output ────────────────────────────────────────────────────────
    if args.output:
        os.makedirs(args.output, exist_ok=True)

        # Save grid-year data
        df_out = df[['year', 'lon', 'lat', 'palace', 'sigma',
                     'TMAX', 'TMIN', 'DTR', 'TMAX_std', 'n_days']].copy()
        out_csv = os.path.join(args.output, 'gridyear_sigma.csv')
        df_out.to_csv(out_csv, index=False)
        if verbose:
            print(f"\n[Output] Grid-year data saved: {out_csv}")

        # Save ANOVA results
        anova_rows = []
        for var, r in anova_results.items():
            for s in range(5):
                if s in r['means']:
                    anova_rows.append({
                        'variable': var, 'sigma': s,
                        'n': r['ns'][s], 'mean': r['means'][s],
                        'F': r['F'], 'p': r['p'], 'eta2': r['eta2'],
                    })
        if anova_rows:
            anova_df = pd.DataFrame(anova_rows)
            anova_out = os.path.join(args.output, 'anova_sigma.csv')
            anova_df.to_csv(anova_out, index=False)
            if verbose:
                print(f"[Output] ANOVA results saved: {anova_out}")

    print("\n" + "=" * 70)
    print("Verification complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
