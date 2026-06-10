# -*- coding: utf-8 -*-
"""
rbf_inference.py
================
RBF-based surrogate model: predicts pressure POD coefficients
from geometry POD coefficients for new unseen patients.

Inspired by POD_builder.py (Prof. Marco E. Biancolini, Tor Vergata).
Adapted for the Human Airways Digital Twin dataset (DiTiDE / EuroHPC).

This is the core of the Digital Twin ROM pipeline:

  geometry (regional displacements, mm)
       ↓  project onto geometry POD (k=11 modes)
  11 geometry coefficients
       ↓  RBF interpolation  ← trained on 90 patients
  5 pressure coefficients
       ↓  reconstruct from pressure POD modes
  full regional pressure field prediction (11 regions)

NOTE: This script represents an intermediate development step.
      The final Digital Twin pipeline is implemented in digital_twin_final.py
      which adds: three-regime surrogates, DOE input, optimised kernels.

What this script does:
  1. Builds geometry and pressure POD bases on 90 training patients.
  2. Projects all 100 patients onto both bases → coefficient vectors.
  3. Trains RBF interpolator: geo_coeffs → press_coeffs.
  4. Predicts pressure for 10 held-out test patients.
  5. Plots predicted vs true: coefficients, peak pressure, full field.
  6. Saves results to rbf_inference.png.

Usage
-----
    python rbf_inference.py

Dependencies
------------
    pip install numpy pandas matplotlib scipy scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.interpolate import RBFInterpolator

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "results.csv"
PRESSURE_CSV = "pressure_results.csv"
DOE_CSV      = "doe.csv"

K_GEOM  = 5    # 5 modes: 99.46% geometry energy — matches final pipeline
K_PRESS = 3    # 3 modes: 99% pressure energy (confirmed on correct data)
N_TRAIN = 90
N_TEST  = 10

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
    df = pd.read_csv(csv_path)
    df["num"] = df["snapshot"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("num").reset_index(drop=True)

print("[INFO] Loading data...")
results  = load_sorted(RESULTS_CSV)
pressure = load_sorted(PRESSURE_CSV)
doe      = pd.read_csv(DOE_CSV)

X_geom  = results[GEOM_COLS].values.T    # (11, 100)
X_press = pressure[PRES_COLS].values.T   # (11, 100)
n_snap  = X_geom.shape[1]

# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
# Random split — avoids extrapolation bias from sequential ordering
np.random.seed(42)
idx       = np.random.permutation(n_snap)
train_idx = idx[:N_TRAIN]
test_idx  = idx[N_TRAIN:]
print(f"[INFO] Train: {len(train_idx)} patients  |  Test: {len(test_idx)} patients")

# ── BUILD POD BASES (training data only) ─────────────────────────────────────
def build_pod(X_train, k):
    mean = np.mean(X_train, axis=1, keepdims=True)
    Xc   = X_train - mean
    U, S, _ = svd(Xc, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    k = min(k, U.shape[1])
    print(f"    Energy with k={k}: {energy[k-1]*100:.2f}%")
    return U[:, :k], mean, S, energy

print("\n[INFO] Building Geometry POD on 90 patients...")
U_geom,  mean_geom,  S_geom,  E_geom  = build_pod(X_geom[:, train_idx],  K_GEOM)

print("[INFO] Building Pressure POD on 90 patients...")
U_press, mean_press, S_press, E_press = build_pod(X_press[:, train_idx], K_PRESS)

# ── PROJECT ALL 100 PATIENTS ONTO BOTH BASES ─────────────────────────────────
def get_coeffs(X, U, mean):
    """Returns (n_snap, k) coefficient matrix."""
    return ((U.T @ (X - mean)).T)

geo_coeffs_all  = get_coeffs(X_geom,  U_geom,  mean_geom)   # (100, 11)
pres_coeffs_all = get_coeffs(X_press, U_press, mean_press)  # (100,  5)

print(f"\n[INFO] Geometry coefficients  : {geo_coeffs_all.shape}")
print(f"[INFO] Pressure coefficients  : {pres_coeffs_all.shape}")

# ── TRAIN RBF ─────────────────────────────────────────────────────────────────
geo_train  = geo_coeffs_all[train_idx]   # (90, K_GEOM)
pres_train = pres_coeffs_all[train_idx]  # (90, K_PRESS)

# Normalise geometry coefficients — critical for RBF performance
# POD modes have very different scales: mode 1 ~±200, mode 3 ~±0.5
# Without normalisation RBF ignores small modes entirely
geo_std   = geo_train.std(axis=0) + 1e-10
geo_train_norm = geo_train / geo_std

print(f"\n[INFO] Training RBF: {geo_train_norm.shape} → {pres_train.shape}")
print(f"[INFO] Geometry coeff std (normalisation factors): "
      f"{np.round(geo_std, 2)}")
rbf = RBFInterpolator(
    geo_train_norm,
    pres_train,
    kernel="thin_plate_spline",
    smoothing=1e-3,
)
print("[✓] RBF trained (with normalised input).")

# ── PREDICT ON TEST SET ───────────────────────────────────────────────────────
geo_test       = geo_coeffs_all[test_idx]                    # (10, K_GEOM)
geo_test_norm  = geo_test / geo_std                          # normalise same scale
pres_test_true = pres_coeffs_all[test_idx]                  # (10, K_PRESS) ground truth
pres_test_pred = rbf(geo_test_norm)                         # (10, K_PRESS) predicted

# Reconstruct full pressure fields
def reconstruct_field(coeffs, U, mean):
    """coeffs: (n, k) → field: (n_regions, n)"""
    return U @ coeffs.T + mean

field_true = reconstruct_field(pres_test_true, U_press, mean_press)  # (11, 10)
field_pred = reconstruct_field(pres_test_pred, U_press, mean_press)  # (11, 10)

# ── METRICS ───────────────────────────────────────────────────────────────────
coeff_rmse  = np.sqrt(np.mean((pres_test_pred - pres_test_true)**2, axis=0))
field_rmse  = np.sqrt(np.mean((field_pred - field_true)**2))
peak_true   = np.max(np.abs(field_true), axis=0)
peak_pred   = np.max(np.abs(field_pred), axis=0)
peak_err    = np.abs(peak_true - peak_pred)
rel_err_pct = (peak_err / peak_true) * 100

print("\n" + "=" * 60)
print("RBF INFERENCE — TEST SET (10 unseen patients)")
print("=" * 60)
for k in range(K_PRESS):
    print(f"  Pressure mode {k+1}: coeff RMSE = {coeff_rmse[k]:.6f} Pa")
print(f"\n  Field RMSE         : {field_rmse:.6f} Pa")
print(f"  Mean peak error    : {peak_err.mean():.6f} Pa")
print(f"  Mean relative err  : {rel_err_pct.mean():.3f} %")
print(f"  Max  relative err  : {rel_err_pct.max():.3f} %")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1: Predicted vs true — pressure mode 1
ax = axes[0, 0]
ax.scatter(pres_test_true[:, 0], pres_test_pred[:, 0],
           color="#185FA5", s=80, zorder=3, label="Test patients")
lim = [min(pres_test_true[:, 0].min(), pres_test_pred[:, 0].min()),
       max(pres_test_true[:, 0].max(), pres_test_pred[:, 0].max())]
ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("True coefficient — Mode 1 (Pa)")
ax.set_ylabel("RBF predicted coefficient (Pa)")
ax.set_title("RBF: Pressure Mode 1\nTrue vs Predicted", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# 2: Predicted vs true — pressure mode 2
ax = axes[0, 1]
ax.scatter(pres_test_true[:, 1], pres_test_pred[:, 1],
           color="#0F6E56", s=80, zorder=3, label="Test patients")
lim = [min(pres_test_true[:, 1].min(), pres_test_pred[:, 1].min()),
       max(pres_test_true[:, 1].max(), pres_test_pred[:, 1].max())]
ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("True coefficient — Mode 2 (Pa)")
ax.set_ylabel("RBF predicted coefficient (Pa)")
ax.set_title("RBF: Pressure Mode 2\nTrue vs Predicted", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# 3: Peak pressure true vs predicted
ax = axes[0, 2]
ax.scatter(peak_true, peak_pred, color="#D85A30", s=80, zorder=3,
           label="Test patients")
lim = [min(peak_true.min(), peak_pred.min()),
       max(peak_true.max(), peak_pred.max())]
ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("True peak pressure (Pa)")
ax.set_ylabel("RBF predicted peak pressure (Pa)")
ax.set_title("Peak pressure: True vs Predicted\n(10 unseen patients)",
             fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# 4: Full field — one example patient
ax = axes[1, 0]
x = range(len(REGION_LABELS))
ax.plot(x, field_true[:, 0], "o-", color="#185FA5",
        linewidth=2, markersize=7, label="True (CFD)")
ax.plot(x, field_pred[:, 0], "s--", color="#D85A30",
        linewidth=2, markersize=7, label="RBF prediction")
ax.set_xticks(x)
ax.set_xticklabels(REGION_LABELS, fontsize=9)
ax.set_ylabel("Pressure (Pa)")
ax.set_title(f"Full pressure field — Patient {test_idx[0]+1}\nTrue vs RBF predicted",
             fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

# 5: Error per region (boxplot over 10 test patients)
ax = axes[1, 1]
err_per_region = np.abs(field_pred - field_true)  # (11, 10)
ax.boxplot(err_per_region.T,
           labels=[r.replace("\n", " ") for r in REGION_LABELS],
           patch_artist=True,
           boxprops=dict(facecolor="#CADCFC", color="#185FA5"),
           medianprops=dict(color="#D85A30", linewidth=2))
ax.set_ylabel("|Error| (Pa)")
ax.set_title("Prediction error per region\n(10 test patients)",
             fontweight="bold")
ax.tick_params(axis="x", labelsize=8, rotation=30)
ax.grid(alpha=0.3, axis="y")

# 6: Pipeline summary
ax = axes[1, 2]
ax.axis("off")
summary = (
    "RBF SURROGATE PIPELINE\n"
    "─────────────────────────────\n\n"
    "INPUT\n"
    "  New patient geometry\n"
    "  (11 regional displacements)\n\n"
    "  ↓  project on geometry POD\n\n"
    f"  {K_GEOM} geometry coefficients\n\n"
    "  ↓  RBF interpolation\n"
    f"     (trained on {N_TRAIN} patients)\n\n"
    f"  {K_PRESS} pressure coefficients\n\n"
    "  ↓  reconstruct via POD modes\n\n"
    "OUTPUT\n"
    "  Full pressure field\n"
    "  (11 anatomical regions)\n\n"
    "─────────────────────────────\n"
    f"  Field RMSE     : {field_rmse:.4e} Pa\n"
    f"  Mean peak err  : {peak_err.mean():.4e} Pa\n"
    f"  Mean rel. err  : {rel_err_pct.mean():.3f} %\n\n"
    "  Time: milliseconds\n"
    "  vs CFD: several hours"
)
ax.text(0.05, 0.97, summary, transform=ax.transAxes,
        fontsize=10.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#F0F4FF", alpha=0.8))

plt.suptitle("RBF Surrogate — Human Airways Digital Twin ROM",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("rbf_inference.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n[INFO] Saved → rbf_inference.png")

# ── DEMO: ONE NEW PATIENT ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"DEMO: Predicting pressure for patient {test_idx[4]+1} (never seen by model)")
print("=" * 60)

i = test_idx[4]
geo_new  = geo_coeffs_all[[i]]          # (1, 11)
pred_c   = rbf(geo_new)                 # (1,  5)
pred_f   = reconstruct_field(pred_c, U_press, mean_press)  # (11, 1)
true_f   = X_press[:, i]

print(f"\n  {'Region':<22} {'True (Pa)':>12}  {'Predicted (Pa)':>14}  {'|Error| (Pa)':>13}")
print(f"  {'-'*65}")
for j, label in enumerate(REGION_LABELS):
    t = true_f[j]
    p = float(pred_f[j, 0])
    print(f"  {label.replace(chr(10),' '):<22} {t:>12.6f}  {p:>14.6f}  {abs(t-p):>13.6f}")

tp = float(np.max(np.abs(true_f)))
pp = float(np.max(np.abs(pred_f)))
print(f"\n  True peak pressure      : {tp:.6f} Pa")
print(f"  Predicted peak pressure : {pp:.6f} Pa")
print(f"  Relative error          : {abs(tp-pp)/tp*100:.3f} %")
print("\n[✓] Digital Twin surrogate working.")
print("    New patient → geometry coefficients → RBF → pressure field.")
print("    Prediction in milliseconds vs hours for full CFD simulation.")