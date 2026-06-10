# -*- coding: utf-8 -*-
"""
pod_analysis.py
===============
Proper Orthogonal Decomposition (POD) of Human Airways snapshots.

Inspired by POD_builder.py (Prof. Marco E. Biancolini, Tor Vergata).
Adapted for the Human Airways Digital Twin dataset (DiTiDE / EuroHPC).

What this script does:
  1. Loads pre-computed regional statistics from results.csv and
     pressure_results.csv (geometry deformation in mm, pressure in Pa).
  2. Builds snapshot matrices  X_geom  (regions × snapshots) and
     X_press (regions × snapshots).
  3. Performs POD via SVD on both matrices.
  4. Plots the cumulated energy curve and singular-value decay.
  5. Reports how many modes are needed to capture 95 % and 99 % of
     the dataset variance.

Usage
-----
    python pod_analysis.py

Dependencies
------------
    pip install numpy pandas matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "results.csv"
PRESSURE_CSV = "pressure_results.csv"

# Anatomical regions to include in the analysis
GEOM_COLS = [
    "glotis_max", "larynx_max", "upper_trachea_bottom_max",
    "gl_max", "gr_max", "glr_max", "grr_max",
    "epiglotis_max", "mouth_region_max",
    "upper_trachea_top_max", "upper_trachea_middle_max",
]
PRES_COLS = [
    "glotis_mean_Pa", "larynx_mean_Pa", "upper_trachea_bottom_mean_Pa",
    "gl_mean_Pa", "gr_mean_Pa", "glr_mean_Pa", "grr_mean_Pa",
    "epiglotis_mean_Pa", "mouth_region_mean_Pa",
    "upper_trachea_top_mean_Pa", "upper_trachea_middle_mean_Pa",
]
REGION_LABELS = [
    "Glottis", "Larynx", "Trachea\n(bot)", "GL", "GR",
    "GLR", "GRR", "Epiglottis", "Mouth", "Trachea\n(top)", "Trachea\n(mid)",
]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_sorted(csv_path):
    """Load CSV and sort by snapshot number."""
    df = pd.read_csv(csv_path)
    df["num"] = df["snapshot"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("num").reset_index(drop=True)


print("[INFO] Loading data...")
results  = load_sorted(RESULTS_CSV)
pressure = load_sorted(PRESSURE_CSV)

# Build snapshot matrices  (regions × snapshots)
X_geom  = results[GEOM_COLS].values.T    # shape: (n_regions, n_snapshots)
X_press = pressure[PRES_COLS].values.T   # shape: (n_regions, n_snapshots)

print(f"[INFO] Geometry matrix : {X_geom.shape}  (regions × snapshots)")
print(f"[INFO] Pressure matrix : {X_press.shape}  (regions × snapshots)")

# ── POD FUNCTION ───────────────────────────────────────────────────────────────
def perform_pod(X, name=""):
    """
    Perform POD via SVD on snapshot matrix X.

    Steps:
      1. Centre: Xc = X - mean(X, axis=1)
      2. Economy SVD: Xc = U @ diag(S) @ Vt
      3. Compute cumulated energy = cumsum(S²) / sum(S²)

    Returns
    -------
    U     : POD modes  (n_features × n_snapshots)
    S     : singular values (descending)
    mean  : column-mean of X (used for reconstruction)
    energy: cumulated energy fraction per mode
    """
    mean = np.mean(X, axis=1, keepdims=True)
    Xc   = X - mean
    U, S, Vt = svd(Xc, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)

    k95 = np.searchsorted(energy, 0.95) + 1
    k99 = np.searchsorted(energy, 0.99) + 1
    print(f"[✓] {name}: modes for 95% energy = {k95},  99% energy = {k99}")
    print(f"    Singular values: {np.round(S[:8], 4)} ...")

    return U, S, mean, energy


print("\n" + "=" * 60)
print("POD — Proper Orthogonal Decomposition via SVD")
print("=" * 60)

U_geom,  S_geom,  mean_geom,  E_geom  = perform_pod(X_geom,  "Geometry (displacement mm)")
U_press, S_press, mean_press, E_press = perform_pod(X_press, "Pressure (Pa)")

# ── PLOT 1: ENERGY CONVERGENCE ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(S_geom)  + 1), E_geom,  "o-",
             color="#185FA5", linewidth=2, markersize=6, label="Geometry (displacement)")
axes[0].plot(range(1, len(S_press) + 1), E_press, "s-",
             color="#D85A30", linewidth=2, markersize=6, label="Pressure")
axes[0].axhline(0.99, color="gray", linestyle="--", linewidth=1.2, label="99% energy")
axes[0].axhline(0.95, color="gray", linestyle=":",  linewidth=1.0, label="95% energy")
axes[0].set_xlabel("Number of modes")
axes[0].set_ylabel("Cumulated Energy")
axes[0].set_title("Energy — POD (SVD)\nHow many modes are needed?", fontweight="bold")
axes[0].set_ylim(0, 1.05)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].semilogy(range(1, len(S_geom)  + 1), S_geom,  "o-",
                 color="#185FA5", markersize=6, linewidth=2, label="Geometry")
axes[1].semilogy(range(1, len(S_press) + 1), S_press, "s-",
                 color="#D85A30", markersize=6, linewidth=2, label="Pressure")
axes[1].set_xlabel("Mode index")
axes[1].set_ylabel("Singular value (log scale)")
axes[1].set_title("Singular Values Decay\n(faster = easier to compress)", fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("POD Analysis — Human Airways (100 patients)", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("pod_energy.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved → pod_energy.png")

# ── PLOT 2: POD MODE SHAPES ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = range(len(REGION_LABELS))

for i in range(min(3, U_geom.shape[1])):
    axes[0].plot(x, U_geom[:, i], "o-", linewidth=1.5,
                 markersize=5, label=f"Mode {i+1}")
axes[0].set_xticks(x)
axes[0].set_xticklabels(REGION_LABELS, fontsize=9)
axes[0].axhline(0, color="gray", linewidth=0.8)
axes[0].set_title("First 3 POD Modes — Geometry\n(contribution of each region)",
                  fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

for i in range(min(3, U_press.shape[1])):
    axes[1].plot(x, U_press[:, i], "s-", linewidth=1.5,
                 markersize=5, label=f"Mode {i+1}")
axes[1].set_xticks(x)
axes[1].set_xticklabels(REGION_LABELS, fontsize=9)
axes[1].axhline(0, color="gray", linewidth=0.8)
axes[1].set_title("First 3 POD Modes — Pressure\n(contribution of each region)",
                  fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("POD Mode Shapes — Human Airways", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("pod_modes.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved → pod_modes.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Geometry : {np.searchsorted(E_geom,  0.99)+1} modes for 99% energy  "
      f"(first SV = {S_geom[0]:.4f})")
print(f"  Pressure : {np.searchsorted(E_press, 0.99)+1} modes for 99% energy  "
      f"(first SV = {S_press[0]:.2f})")
print()
print("  Interpretation:")
print("  - Pressure is LOW-RANK: dominated by 1 main airflow gradient mode.")
print("  - Geometry is HIGHER-RANK: patient-specific deformation patterns")
print("    require more modes — reflects anatomical variability.")