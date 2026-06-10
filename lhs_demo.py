# -*- coding: utf-8 -*-
"""
lhs_demo.py
===========
Latin Hypercube Sampling (LHS) — demonstration for the Human Airways project.

Inspired by LHS_simple.py and LHS_numpy.py (Prof. Marco E. Biancolini, Tor Vergata).

What is LHS?
------------
Latin Hypercube Sampling is a statistical method for generating a
near-random sample of parameter values. It ensures that:
  - The parameter space is divided into N equal-probability strata.
  - Each stratum is sampled exactly once per dimension.
  - Result: much better space-filling than pure random sampling.

In our project, LHS was used to generate the 100 patient anatomies
stored in doe.csv — ensuring all combinations of anatomical parameters
(glottis area, trachea length, branch angles, etc.) are well covered.

Usage
-----
    python lhs_demo.py

Dependencies
------------
    pip install numpy scipy matplotlib pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_SAMPLES   = 100   # number of virtual patients (same as our dataset)
N_DIMS      = 3     # dimensions for visualisation (we show 3 of 26 params)
SEED        = 42

# ── PURE NUMPY LHS ────────────────────────────────────────────────────────────
def lhs_numpy(n: int, d: int, seed: int = 42) -> np.ndarray:
    """
    Pure-numpy Latin Hypercube Sampling.
    Returns (n, d) array in [0, 1]^d.

    Math:
      For each dimension j:
        strata k = 0 .. n-1
        point_k = (k + r_k) / n,  r_k ~ U(0,1)
        then apply random permutation across rows
    """
    rng = np.random.default_rng(seed)
    U = np.empty((n, d))
    for j in range(d):
        perm      = rng.permutation(n)
        offsets   = rng.random(n)
        U[:, j]   = (perm + offsets) / n
    return U


def lhs_scipy(n: int, d: int, seed: int = 42) -> np.ndarray:
    """
    LHS using SciPy QMC with 'random-cd' optimisation for better space-filling.
    Returns (n, d) array in [0, 1]^d.
    """
    sampler = qmc.LatinHypercube(d=d, seed=seed, optimization="random-cd")
    return sampler.random(n=n)


def map_to_ranges(U: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Affine map from [0,1]^d to [lows, highs]."""
    return lows + (highs - lows) * U


# ── GENERATE SAMPLES ───────────────────────────────────────────────────────────
print("=" * 60)
print("Latin Hypercube Sampling — Human Airways DOE")
print("=" * 60)

U_numpy  = lhs_numpy(N_SAMPLES, N_DIMS, seed=SEED)
U_scipy  = lhs_scipy(N_SAMPLES, N_DIMS, seed=SEED)
U_random = np.random.default_rng(SEED).random((N_SAMPLES, N_DIMS))

# Map to ranges of the 3 most important parameters from our doe.csv
param_names = ["A_glotis (mm²)", "l_trachea (mm)", "r_curvature (mm)"]
lows  = np.array([86.0,  80.0,  30.0])
highs = np.array([230.0, 150.0, 70.0])

X_lhs    = map_to_ranges(U_scipy, lows, highs)
X_random = map_to_ranges(U_random, lows, highs)

print(f"\nLHS sample — first 5 rows (3 parameters shown):")
for i in range(5):
    print(f"  Patient {i+1:3d}: "
          f"A_glotis={X_lhs[i,0]:.1f}  "
          f"l_trachea={X_lhs[i,1]:.1f}  "
          f"r_curvature={X_lhs[i,2]:.1f}")

# Verify stratification: each stratum used exactly once per dimension
bins = np.floor(U_scipy * N_SAMPLES).astype(int).clip(0, N_SAMPLES - 1)
print("\nStratification check (each bin used exactly once per dimension):")
for j in range(N_DIMS):
    ok = np.array_equal(np.sort(bins[:, j]), np.arange(N_SAMPLES))
    print(f"  Dimension {j+1} ({param_names[j]}): {'✓ OK' if ok else '✗ FAIL'}")

# ── PLOT: LHS vs PURE RANDOM ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(X_random[:, 0], X_random[:, 1],
                color="#D85A30", alpha=0.7, s=40, edgecolors="white", linewidths=0.5)
axes[0].set_xlabel(param_names[0])
axes[0].set_ylabel(param_names[1])
axes[0].set_title("Pure Random Sampling\n(clustering and gaps visible)", fontweight="bold")
axes[0].grid(alpha=0.3)

axes[1].scatter(X_lhs[:, 0], X_lhs[:, 1],
                color="#185FA5", alpha=0.7, s=40, edgecolors="white", linewidths=0.5)
axes[1].set_xlabel(param_names[0])
axes[1].set_ylabel(param_names[1])
axes[1].set_title("Latin Hypercube Sampling\n(uniform space coverage)", fontweight="bold")
axes[1].grid(alpha=0.3)

plt.suptitle(f"DOE Comparison — {N_SAMPLES} Virtual Patients\n"
             "LHS ensures every region of the parameter space is represented",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig("lhs_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[INFO] Saved → lhs_comparison.png")

# ── LOAD ACTUAL DOE AND COMPARE ───────────────────────────────────────────────
try:
    doe = pd.read_csv("doe.csv")
    X_doe = doe[["A_glotis", "l_trachea", "r_curvature"]].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_doe[:, 0], X_doe[:, 1],
               color="#0F6E56", alpha=0.8, s=50,
               edgecolors="white", linewidths=0.5, label="Actual DOE (doe.csv)")
    ax.set_xlabel("A_glotis (mm²)")
    ax.set_ylabel("l_trachea (mm)")
    ax.set_title("Actual DOE — 100 Patient Anatomies\n"
                 "Generated by RBF Morph using LHS", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("doe_actual.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] Saved → doe_actual.png")
except FileNotFoundError:
    print("[WARN] doe.csv not found — skipping actual DOE plot")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("  LHS guarantees that each parameter stratum is sampled exactly once.")
print("  With 100 patients and 26 parameters, LHS gives far better coverage")
print("  than random sampling — critical for training a reliable ROM.")