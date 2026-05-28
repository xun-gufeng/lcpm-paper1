#!/usr/bin/env python3
"""Generate 6 key figures for the LCPM paper. Output: PNG at 300 DPI."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

OUT = './figures/'
os.makedirs(OUT, exist_ok=True)

# Resolve CSV path: current dir → script dir → fallback
import sys
_CSV_CANDIDATES = [
    './cug_cma_gridyear_full.csv',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cug_cma_gridyear_full.csv'),
    './中医学习整理/气候数据/cug_cma_gridyear_full.csv',
]

def _find_csv():
    for p in _CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    print(f"WARNING: cug_cma_gridyear_full.csv not found. Searched: {_CSV_CANDIDATES}")
    sys.exit(1)

PALACE_WX = {0:4, 1:2, 2:0, 3:0, 4:2, 5:3, 6:3, 7:2, 8:1}
WX_COLORS = {0:'#2ecc71', 1:'#e74c3c', 2:'#f1c40f', 3:'#95a5a6', 4:'#3498db'}
WX_EN = {0:'Wood', 1:'Fire', 2:'Earth', 3:'Metal', 4:'Water'}

# Palace positions on 3x3 grid (col, row) row0=top
POS = {0:(1,0), 1:(0,2), 2:(2,1), 3:(2,2), 4:(1,1), 5:(0,0), 6:(0,1), 7:(2,0), 8:(1,2)}
P_NAMES = {0:'N 坎', 1:'SW 坤', 2:'E 震', 3:'SE 巽', 4:'C 中', 5:'NW 乾', 6:'W 兑', 7:'NE 艮', 8:'S 离'}

GEN = [(0,2),(0,3),(1,5),(1,6),(2,8),(3,8),(4,5),(4,6),(5,0),(6,0),(7,5),(7,6),(8,1),(8,4),(8,7)]
REST = [(0,8),(1,0),(4,0),(7,0),(2,1),(2,4),(2,7),(3,1),(3,4),(3,7),(5,2),(5,3),(6,2),(6,3),(8,5),(8,6)]


def draw_arrows(ax, links, color, style='-', lw=1.8, alpha=0.7, side=1):
    for i, j in links:
        xi, yi = POS[i]; xj, yj = POS[j]
        dx, dy = xj-xi, yj-yi
        L = max(np.sqrt(dx**2+dy**2), 0.01)
        ox, oy = -dy/L*0.07*side, dx/L*0.07*side
        ax.annotate('', xy=(xj+ox, yj+oy), xytext=(xi+ox, yi+oy),
                     arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha, linestyle=style))


def fig1():
    """Luoshu lattice topology."""
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_arrows(ax, GEN, '#27ae60', '-', 2.0, 0.75, side=1)
    draw_arrows(ax, REST, '#c0392b', '--', 1.6, 0.55, side=-1)
    for p in range(9):
        x, y = POS[p]; c = WX_COLORS[PALACE_WX[p]]
        ax.add_patch(plt.Circle((x,y), 0.24, color=c, ec='black', lw=2, zorder=5))
        ax.text(x, y, P_NAMES[p], ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    ax.legend(handles=[mpatches.Patch(color='#27ae60', label='Generation (15 bonds)'),
                       mpatches.Patch(color='#c0392b', label='Restriction (16 bonds)')],
              loc='upper right', fontsize=11)
    ax.set_xlim(-0.6,2.6); ax.set_ylim(-0.6,2.6); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Figure 1: Luoshu Lattice — Hetu Coupling Topology\n(15 Generation + 16 Restriction = 31 Directed Bonds)',
                 fontsize=12, fontweight='bold')
    plt.savefig(f'{OUT}fig1_luoshu_topology.png'); plt.close()
    print("Fig1 done")


def fig2():
    """Five USC ground states."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for k in range(5):
        ax = axes[k]
        for p in range(9):
            x, y = POS[p]; s = (PALACE_WX[p]+k)%5
            ax.add_patch(plt.Circle((x,y), 0.26, color=WX_COLORS[s], ec='black', lw=1.5, zorder=5))
            ax.text(x, y, str(s), ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)
        ax.set_xlim(-0.6,2.6); ax.set_ylim(-0.6,2.6); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'USC(k={k})\n31/31 bonds', fontsize=11, fontweight='bold')
    patches = [mpatches.Patch(color=WX_COLORS[w], label=f'{w}: {WX_EN[w]}') for w in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5,-0.02))
    fig.suptitle('Figure 2: Five USC Ground States — Z\u2085 Orbit (Degeneracy = 5)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(f'{OUT}fig2_usc_ground_states.png'); plt.close()
    print("Fig2 done")


def fig3():
    """YPM phase diagram."""
    h_y, h_d, T, n = 1.5, 0.3, 60, 12
    Jc = (h_y-h_d)*(T-n)/(T-1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Order parameter
    Jt = np.linspace(0, 2.0, 1000)
    phi = np.where(Jt < Jc, 0.0, np.where(Jt > Jc, 1.0, 0.5))
    a1.plot(Jt, phi, 'b-', lw=3)
    a1.axvline(Jc, color='red', ls='--', lw=2, label=f'$J_t^c = {Jc:.3f}$')
    a1.fill_between(Jt[Jt<=Jc], 0, 1, alpha=0.08, color='blue')
    a1.fill_between(Jt[Jt>=Jc], 0, 1, alpha=0.08, color='red')
    a1.text(Jc/2, 1.15, 'Responsive', ha='center', fontsize=12, color='blue', fontweight='bold')
    a1.text((Jc+2)/2, 1.15, 'Locked', ha='center', fontsize=12, color='red', fontweight='bold')
    a1.set_xlabel('$J_t$ (Temporal Coupling)'); a1.set_ylabel('$\\phi$')
    a1.set_title('(a) Order Parameter — First-Order Jump', fontweight='bold')
    a1.legend(fontsize=10); a1.set_xlim(0,2); a1.set_ylim(-0.1,1.35)

    # (b) Energy crossing
    EA = -9*h_y*T - 9*h_d*n
    EB = -9*Jt*(T-1) - 9*h_y*n - 9*h_d*T
    a2.plot(Jt, np.full_like(Jt, EA), 'b-', lw=3, label='Responsive $E_A$')
    a2.plot(Jt, EB, 'r-', lw=3, label='Locked $E_B$')
    a2.plot(Jt, np.minimum(EA, EB), 'k--', lw=2, alpha=0.5, label='Ground state')
    a2.axvline(Jc, color='gray', ls='--', lw=1.5, alpha=0.6)
    a2.annotate(f'$J_t^c={Jc:.3f}$', xy=(Jc, EA), xytext=(Jc+0.3, EA+120),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=11, fontweight='bold')
    a2.set_xlabel('$J_t$'); a2.set_ylabel('Energy'); a2.set_title('(b) Energy Crossing', fontweight='bold')
    a2.legend(fontsize=10)

    fig.suptitle('Figure 3: YPM Phase Diagram — First-Order Transition', fontsize=13, fontweight='bold')
    plt.savefig(f'{OUT}fig3_ypm_phase_diagram.png'); plt.close()
    print("Fig3 done")


def fig4():
    """Z₅ ANOVA effect sizes and σ-group climate profiles (grid-year level)."""
    import pandas as pd
    from scipy import stats as sp_stats

    csv_path = _find_csv()
    df = pd.read_csv(csv_path)

    vars_info = [('DTR', 'DTR'), ('TMIN', 'TMIN'), ('TMAX_std', 'TMAX_std')]
    var_labels = ['DTR', 'TMIN', r'TMAX$_{\rm std}$']
    var_colors = ['#e74c3c', '#3498db', '#2ecc71']

    # --- Compute ANOVA η² from raw data ---
    groups = [df[df['sigma'] == s] for s in range(5)]
    N = len(df)

    eta2_vals = []
    F_vals = []
    p_vals = []
    # For panel (b): group means and SE per variable per σ
    grp_means = []   # shape: (3 vars, 5 σ groups)
    grp_se = []      # shape: (3 vars, 5 σ groups)

    for col, _ in vars_info:
        ss_total = np.sum((df[col].values - df[col].mean())**2)
        ss_between = sum(len(g) * (g[col].mean() - df[col].mean())**2 for g in groups)
        eta2 = ss_between / ss_total if ss_total > 0 else 0
        eta2_vals.append(eta2)

        # F and p via one-way ANOVA
        F, p = sp_stats.f_oneway(*(g[col].dropna().values for g in groups))
        F_vals.append(F)
        p_vals.append(p)

        # Group means and SE
        means = np.array([g[col].mean() for g in groups])
        se = np.array([g[col].std(ddof=1) / np.sqrt(len(g)) for g in groups])
        grp_means.append(means)
        grp_se.append(se)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- (a) η² bar chart ---
    x = np.arange(len(vars_info))
    bars = a1.bar(x, eta2_vals, width=0.5, color=var_colors, edgecolor='black', lw=1.2)
    for i, (b, ev, fv, pv) in enumerate(zip(bars, eta2_vals, F_vals, p_vals)):
        a1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f'$\\eta^2={ev:.3f}$\n$F={fv:.0f}$',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Cohen's benchmarks
    a1.axhline(0.01, color='gray', ls='--', lw=1.2, alpha=0.7)
    a1.axhline(0.06, color='gray', ls='-.', lw=1.2, alpha=0.7)
    a1.text(len(vars_info)-0.5, 0.012, "Cohen's small (0.01)", fontsize=8, color='gray', ha='right')
    a1.text(len(vars_info)-0.5, 0.062, "Cohen's medium (0.06)", fontsize=8, color='gray', ha='right')
    a1.set_xticks(x)
    a1.set_xticklabels(var_labels)
    a1.set_ylabel(r'Effect size $\eta^2$')
    a1.set_title(r'(a) ANOVA Effect Sizes ($n = 19{,}022$)', fontweight='bold')
    a1.set_ylim(0, max(eta2_vals)*1.45)

    # --- (b) Standardized group means ± 2SE ---
    sigma_colors = [WX_COLORS[s] for s in range(5)]
    sigma_labels = [f'$\\sigma={s}$ ({WX_EN[s]})' for s in range(5)]

    # Standardize each variable across all groups
    for vi in range(len(vars_info)):
        means_all = grp_means[vi]
        grand_mean = means_all.mean()
        pooled_sd = np.sqrt(np.mean([(grp_means[vi][s] - grand_mean)**2 for s in range(5)]))
        if pooled_sd == 0:
            pooled_sd = 1
        z_means = (means_all - grand_mean) / pooled_sd
        z_se = grp_se[vi] / pooled_sd

        offset = (vi - 1) * 0.15
        for s in range(5):
            a2.errorbar(s + offset, z_means[s], yerr=2*z_se[s],
                        fmt='o', color=sigma_colors[s], markersize=7,
                        markeredgecolor='black', markeredgewidth=0.8,
                        capsize=3, capthick=1, elinewidth=1.2)

    # Legend for σ groups
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=sigma_colors[s], marker='o', linestyle='None',
                             markersize=7, markeredgecolor='black', markeredgewidth=0.8,
                             label=sigma_labels[s]) for s in range(5)]
    a2.legend(handles=handles, loc='upper right', fontsize=9, ncol=1)
    a2.set_xticks(range(5))
    a2.set_xticklabels([f'{s}' for s in range(5)])
    a2.set_xlabel(r'$\sigma$ group')
    a2.set_ylabel('Standardized mean (pooled $z$)')
    a2.set_title(r'(b) $\sigma$-Group Climate Profiles ($\pm$2 SE)', fontweight='bold')
    a2.axhline(0, color='gray', lw=0.5, alpha=0.5)

    fig.suptitle(r'Figure 4: $\mathbb{Z}_5$ ANOVA Effect Sizes and $\sigma$-Group Climate Profiles',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'{OUT}fig4_climate_anova.png'); plt.close()
    print("Fig4 done")


def fig5():
    """Da Si Tian boundary jumps — real CUG-CMA data."""
    import pandas as pd

    csv_path = _find_csv()
    df = pd.read_csv(csv_path)

    # Grid-year mean by year
    yearly = df.groupby('year')[['DTR', 'TMIN']].mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

    # 5-yr running mean helper
    def runmean(s, w=5):
        return s.rolling(w, center=True, min_periods=1).mean()

    # (a) 1924 boundary — DTR
    yrs_dtr = yearly.index.values
    dtr_raw = yearly['DTR'].values
    dtr_sm = runmean(yearly['DTR']).values
    a1.plot(yrs_dtr, dtr_raw, 'o', markersize=2.5, color='#2c3e50', alpha=0.5)
    a1.plot(yrs_dtr, dtr_sm, '-', lw=2.2, color='#e74c3c', label='5-yr running mean')
    a1.axvline(1924, color='red', ls='--', lw=2)
    a1.annotate('1924 Da Si Tian\nboundary', xy=(1924, dtr_sm.max()*0.7), fontsize=10, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round', fc='yellow', alpha=0.7))
    a1.set_xlabel('Year'); a1.set_ylabel('DTR Anomaly (°C)')
    a1.set_title('(a) 1924 Boundary — DTR\n89.2% grid points increasing, binomial $p=9.2×10^{-16}$', fontweight='bold')
    a1.legend(fontsize=9)

    # (b) 1984 boundary — TMIN
    yrs_tmin = yearly.index.values
    tmin_raw = yearly['TMIN'].values
    tmin_sm = runmean(yearly['TMIN']).values
    a2.plot(yrs_tmin, tmin_raw, 'o', markersize=2.5, color='#2c3e50', alpha=0.5)
    a2.plot(yrs_tmin, tmin_sm, '-', lw=2.2, color='#3498db', label='5-yr running mean')
    a2.axvline(1984, color='red', ls='--', lw=2)
    a2.annotate('1984 Da Si Tian\nboundary', xy=(1984, tmin_sm.max()*0.5), fontsize=10, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round', fc='yellow', alpha=0.7))
    a2.set_xlabel('Year'); a2.set_ylabel('TMIN Anomaly (°C)')
    a2.set_title('(b) 1984 Boundary — TMIN\n99.5% grid points increasing, binomial $p=1.9×10^{-54}$', fontweight='bold')
    a2.legend(fontsize=9)

    fig.suptitle('Figure 5: Da Si Tian Boundary Jumps — Phase Transition Signatures',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'{OUT}fig5_dst_boundary_jumps.png'); plt.close()
    print("Fig5 done")


def fig6():
    """Noise perturbation curves."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    rho = np.linspace(0, 1.0, 500)
    DA = 1 - 5*rho/4; DB = DA**2

    a1.plot(rho, DA, 'b-', lw=3, label='$\\Delta_A = 1 - 5\\rho/4$')
    a1.plot(rho, DB, 'r-', lw=3, label='$\\Delta_B = (1 - 5\\rho/4)^2$')
    a1.axvline(0.8, color='gray', ls='--', lw=2, alpha=0.7)
    a1.axhline(0, color='black', lw=0.5)
    a1.annotate('$\\rho_c = 0.80$', xy=(0.8,0), xytext=(0.85,0.3),
                arrowprops=dict(arrowstyle='->'), fontsize=12, fontweight='bold')
    a1.fill_between(rho, 0, DA, alpha=0.08, color='blue')
    a1.fill_between(rho, 0, DB, alpha=0.08, color='red')
    a1.set_xlabel('$\\rho$ (Noise Level)'); a1.set_ylabel('Contrast')
    a1.set_title('(a) Contrast Degradation vs Noise', fontweight='bold')
    a1.legend(fontsize=11); a1.set_xlim(0,1); a1.set_ylim(-0.2,1.1)

    DA_pos = DA[DA>=0]; DB_pos = DB[DA>=0]
    a2.plot(DA_pos, DB_pos, 'g-', lw=3, label='$\\Delta_B = \\Delta_A^2$')
    a2.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Linear reference')
    for rv in [0.0, 0.2, 0.4, 0.6]:
        da = 1-5*rv/4; db = da**2
        a2.scatter(da, db, s=80, c='green', edgecolors='black', zorder=5)
        a2.annotate(f'$\\rho={rv}$', (da,db), xytext=(8,-12), textcoords='offset points', fontsize=9)
    a2.set_xlabel('$\\Delta_A$ (External Field Contrast)')
    a2.set_ylabel('$\\Delta_B$ (Temporal Coupling Contrast)')
    a2.set_title('(b) Exact Identity: $\\Delta_B = \\Delta_A^2$', fontweight='bold')
    a2.legend(fontsize=11); a2.set_xlim(-0.1,1.1); a2.set_ylim(-0.1,1.1); a2.set_aspect('equal')

    fig.suptitle('Figure 6: Noise Perturbation — All-or-Nothing Degradation ($\\rho_c = 0.80$)',
                 fontsize=13, fontweight='bold')
    plt.savefig(f'{OUT}fig6_noise_perturbation.png'); plt.close()
    print("Fig6 done")


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print(f"\nAll 6 figures -> {OUT}")
