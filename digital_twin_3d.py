# -*- coding: utf-8 -*-
"""
digital_twin_3d_improved.py
==================
Human Airways Digital Twin — 3D Visualisation.

Usage
-----
    python digital_twin_3d_improved.py                       # interactive input
    python digital_twin_3d_improved.py --compare             # prediction vs nearest CFD + error map
    python digital_twin_3d_improved.py --no-deform           # show pressure on baseline mesh
    python digital_twin_3d_improved.py --deform-scale 0.25   # visual deformation scale

Dependencies
------------
    pip install numpy pandas scipy pyvista
"""

import sys
import struct
import json
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import svd
from scipy.interpolate import RBFInterpolator
import pyvista as pv

# ── PATHS ─────────────────────────────────────────────────────────────────────
POINTS_BIN    = "Points/points.bin"
SETTINGS_JSON = "Points/settings.json"
RESULTS_CSV   = "results.csv"
PRESSURE_CSV  = "pressure_results.csv"
DOE_CSV       = "doe.csv"

# ── CONFIG ────────────────────────────────────────────────────────────────────
K_GEOM         = 5
K_PRESS        = 3
REGIME_LOW_MAX = 30
REGIME_MID_MAX = 100
OUTLIER_IDX    = [53]

# Visual deformation is approximate: regional max displacement [mm] is mapped
# back to the point cloud as a small outward displacement for demonstration.
DEFAULT_DEFORM_SCALE = 0.20

TOP_DOE_FOR_GEOM = [
    "l_trachea", "r_curvature", "d_trachea", "A_epiglotis", "A_glotis",
    "l_rrr", "l_l", "l_rll", "teta_branch_r", "l_r"
]
GEOM_COLS = [
    "glotis_max", "larynx_max", "upper_trachea_bottom_max",
    "gl_max", "gr_max", "glr_max", "grr_max", "epiglotis_max",
    "mouth_region_max", "upper_trachea_top_max", "upper_trachea_middle_max",
]
PRES_COLS = [
    "glotis_mean_Pa", "larynx_mean_Pa", "upper_trachea_bottom_mean_Pa",
    "gl_mean_Pa", "gr_mean_Pa", "glr_mean_Pa", "grr_mean_Pa",
    "epiglotis_mean_Pa", "mouth_region_mean_Pa",
    "upper_trachea_top_mean_Pa", "upper_trachea_middle_mean_Pa",
]
REGION_NAMES = [
    "glotis", "larynx", "upper_trachea_bottom",
    "gl", "gr", "glr", "grr",
    "epiglotis", "mouth_region",
    "upper_trachea_top", "upper_trachea_middle",
]
REGION_LABELS = [
    "Glottis", "Larynx", "Trachea (bot)", "GL", "GR",
    "GLR", "GRR", "Epiglottis", "Mouth", "Trachea (top)", "Trachea (mid)",
]
REGIME_KERNELS = {
    "low":  {"kernel": "linear",  "smoothing": 1e-4},
    "mid":  {"kernel": "quintic", "smoothing": 1e-4},
    "high": {"kernel": "linear",  "smoothing": 0.1},
}

# ── BINARY READERS ────────────────────────────────────────────────────────────
def read_bin_vector(path):
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64).reshape(-1, 3)

# ── LOAD MESH ─────────────────────────────────────────────────────────────────
print("[INFO] Loading base mesh...")
pts   = read_bin_vector(POINTS_BIN)
n_pts = len(pts)
print(f"[INFO] Mesh: {n_pts:,} points")

with open(SETTINGS_JSON) as f:
    ns = json.load(f)["namedSelections"]

print("[INFO] Building region mask...")
region_mask = np.full(n_pts, -1, dtype=np.int32)
for reg_idx, reg_name in enumerate(REGION_NAMES):
    if reg_name in ns:
        start, _, end = ns[reg_name]
        region_mask[start:end+1] = reg_idx

n_assigned = (region_mask >= 0).sum()
print(f"[INFO] Points assigned to regions: {n_assigned:,} / {n_pts:,} "
      f"({n_assigned/n_pts*100:.1f}%)")

# ── LOAD TRAINING DATA ────────────────────────────────────────────────────────
def load_sorted(p):
    df = pd.read_csv(p)
    df["num"] = df["snapshot"].str.extract(r"(\d+)").astype(int)
    return df.sort_values("num").reset_index(drop=True)

print("[INFO] Loading training data...")
results  = load_sorted(RESULTS_CSV)
pressure = load_sorted(PRESSURE_CSV)
doe      = pd.read_csv(DOE_CSV)

doe_cols_all = [c for c in doe.columns
                if c not in ("snapshot", "Snapshot", "num", "index")
                and doe[c].dtype in (float, int, "float64", "int64")]
top_g_idx    = [doe_cols_all.index(c) for c in TOP_DOE_FOR_GEOM]

X_doe_geom = doe[doe_cols_all].values[:, top_g_idx]
X_geom     = results[GEOM_COLS].values
X_press    = pressure[PRES_COLS].values.T
mouth      = X_press[8]
n_snap     = 100

valid_idx    = np.array([i for i in range(n_snap) if i not in OUTLIER_IDX])
X_geom_v     = X_geom[valid_idx]
X_press_v    = X_press[:, valid_idx]
X_doe_geom_v = X_doe_geom[valid_idx]
mouth_v      = mouth[valid_idx]

low_idx  = np.where(mouth_v <  REGIME_LOW_MAX)[0]
mid_idx  = np.where((mouth_v >= REGIME_LOW_MAX) & (mouth_v <= REGIME_MID_MAX))[0]
high_idx = np.where(mouth_v >  REGIME_MID_MAX)[0]

# ── TRAIN SURROGATE ───────────────────────────────────────────────────────────
print("[INFO] Training surrogate...")

doe_geom_mean = X_doe_geom_v.mean(axis=0)
doe_geom_std  = X_doe_geom_v.std(axis=0) + 1e-10
geom_pop_mean = X_geom_v.mean(axis=0, keepdims=True)
geom_pop_std  = X_geom_v.std(axis=0,  keepdims=True) + 1e-10

rbf0 = RBFInterpolator(
    (X_doe_geom_v - doe_geom_mean) / doe_geom_std,
    (X_geom_v - geom_pop_mean) / geom_pop_std,
    kernel="thin_plate_spline", smoothing=1e-3
)

gm_g    = X_geom_v.mean(axis=0, keepdims=True)
_, S_g, Vt_g = svd(X_geom_v - gm_g, full_matrices=False)
Vt_g_k  = Vt_g[:K_GEOM, :]
G_all_v = (X_geom_v - gm_g) @ Vt_g_k.T

def train_sur(idx, name):
    G = G_all_v[idx]; Xp = X_press_v[:, idx]
    gm = G.mean(axis=0); gs = G.std(axis=0) + 1e-10
    rm = Xp.mean(axis=1, keepdims=True)
    rs = Xp.std(axis=1,  keepdims=True) + 1e-10
    Xpn = (Xp - rm) / rs
    pm  = Xpn.mean(axis=1, keepdims=True)
    Up, Sp, _ = svd(Xpn - pm, full_matrices=False)
    Up = Up[:, :min(K_PRESS, Up.shape[1])]
    pc  = (Up.T @ (Xpn - pm)).T
    cfg = REGIME_KERNELS[name.lower()]
    rbf = RBFInterpolator((G - gm) / gs, pc,
                          kernel=cfg["kernel"], smoothing=cfg["smoothing"])
    return {"gm": gm, "gs": gs, "rm": rm, "rs": rs,
            "Up": Up, "pm": pm, "rbf": rbf,
            "mouth_range": (float(mouth_v[idx].min()), float(mouth_v[idx].max()))}

surrogates = {
    "low":  train_sur(low_idx,  "low"),
    "mid":  train_sur(mid_idx,  "mid"),
    "high": train_sur(high_idx, "high"),
}
print("[✓] Surrogate ready.")

# ── PREDICT ───────────────────────────────────────────────────────────────────
def predict(doe_params_dict, regime=None):
    x  = np.array([doe_params_dict.get(c, doe[c].mean()) for c in TOP_DOE_FOR_GEOM])
    xn = (x - doe_geom_mean) / doe_geom_std
    geom_pred = (rbf0(xn.reshape(1, -1)) * geom_pop_std + geom_pop_mean).flatten()

    if regime is None:
        epi = geom_pred[7]; mth = geom_pred[8]
        regime = ("high" if (epi < 4.5 and mth < 6.0) else
                  "mid"  if (epi < 7.0 or mth < 7.5) else "low")

    s   = surrogates[regime]
    g_c = (geom_pred - gm_g.flatten()) @ Vt_g_k.T
    g_n = (g_c - s["gm"]) / s["gs"]
    pc  = s["rbf"](g_n.reshape(1, -1))
    fn  = s["Up"] @ pc.T + s["pm"]
    fp  = (fn * s["rs"] + s["rm"]).flatten()

    pressure_dict = {label: float(v) for label, v in zip(REGION_LABELS, fp)}
    return pressure_dict, geom_pred, regime

# ── BUILD MESH FIELD ──────────────────────────────────────────────────────────
def build_field(values_dict_or_series, use_pres_cols=False):
    """Assign per-region values to all mesh points."""
    if use_pres_cols:
        # pandas Series from pressure_results.csv
        mean_val = float(values_dict_or_series[PRES_COLS].mean())
        field = np.full(n_pts, mean_val, dtype=np.float32)
        for reg_idx, reg_name in enumerate(REGION_NAMES):
            col = reg_name + "_mean_Pa"
            if col in values_dict_or_series.index:
                field[region_mask == reg_idx] = float(values_dict_or_series[col])
    else:
        # dict {label: value}
        mean_val = float(np.mean(list(values_dict_or_series.values())))
        field = np.full(n_pts, mean_val, dtype=np.float32)
        for reg_idx, label in enumerate(REGION_LABELS):
            field[region_mask == reg_idx] = values_dict_or_series[label]
    return field


def values_from_pressure_row(row):
    """Return real CFD regional pressure values in the same order as REGION_LABELS."""
    return [float(row[c]) for c in PRES_COLS]


def values_from_pressure_dict(pressure_dict):
    """Return predicted regional pressure values in the same order as REGION_LABELS."""
    return [float(pressure_dict[label]) for label in REGION_LABELS]


def build_deformed_points(geom_pred, deform=True, scale=DEFAULT_DEFORM_SCALE):
    """
    Build an approximate deformed point cloud for visualisation.

    Important: this is NOT the original FEA displacement field. The surrogate predicts
    regional maximum displacement values [mm]. For visualisation, each anatomical
    region is displaced slightly outward from its local centroid. This makes the
    geometry variation visible in the 3D demo while keeping the result scientifically
    honest as a region-wise approximation.
    """
    if not deform:
        return pts.astype(np.float32)

    pts_def = pts.astype(np.float32).copy()

    for reg_idx, disp_mm in enumerate(geom_pred):
        mask = region_mask == reg_idx
        if not np.any(mask):
            continue

        region_pts = pts[mask]
        centroid = region_pts.mean(axis=0)
        directions = region_pts - centroid
        norms = np.linalg.norm(directions, axis=1)

        valid = norms > 1e-12
        unit = np.zeros_like(directions, dtype=np.float32)
        unit[valid] = (directions[valid] / norms[valid, None]).astype(np.float32)

        # geom_pred is in mm; points are in metres. Scale keeps demo deformation subtle.
        disp_m = float(disp_mm) * 1e-3 * float(scale)
        pts_def[mask] += unit * disp_m

    return pts_def


def build_geom_field(geom_pred):
    """Assign predicted regional displacement [mm] to mesh points for overlay/debug."""
    mean_val = float(np.mean(geom_pred))
    field = np.full(n_pts, mean_val, dtype=np.float32)
    for reg_idx, val in enumerate(geom_pred):
        field[region_mask == reg_idx] = float(val)
    return field


def build_abs_error_field(pred_dict, real_row):
    """Build region-wise absolute error field |predicted - real| in Pa."""
    pred_vals = values_from_pressure_dict(pred_dict)
    real_vals = values_from_pressure_row(real_row)
    err_vals = [abs(p - r) for p, r in zip(pred_vals, real_vals)]
    mean_val = float(np.mean(err_vals))
    field = np.full(n_pts, mean_val, dtype=np.float32)
    for reg_idx, err in enumerate(err_vals):
        field[region_mask == reg_idx] = float(err)
    return field, err_vals

# ── FIND NEAREST PATIENT IN SAME REGIME ──────────────────────────────────────
def find_nearest(params, regime):
    mouth_vals = pressure["mouth_region_mean_Pa"].values
    if regime == "low":
        candidates = np.where(mouth_vals < REGIME_LOW_MAX)[0]
    elif regime == "mid":
        candidates = np.where(
            (mouth_vals >= REGIME_LOW_MAX) & (mouth_vals <= REGIME_MID_MAX))[0]
    else:
        candidates = np.where(mouth_vals > REGIME_MID_MAX)[0]

    x      = np.array([params.get(c, doe[c].mean()) for c in doe_cols_all])
    X_cand = doe[doe_cols_all].values[candidates]
    m = X_cand.mean(axis=0); s = X_cand.std(axis=0) + 1e-10
    dists  = np.linalg.norm((X_cand - m) / s - (x - m) / s, axis=1)
    nearest = candidates[dists.argmin()]
    return nearest, pressure.iloc[nearest]

# ── VISUALISE ─────────────────────────────────────────────────────────────────
def clim_from_values(vals):
    vals = [float(v) for v in vals]
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1e-9)
    return [vmin - 0.08 * span, vmax + 0.08 * span]


def visualise(pressure_dict, geom_pred, regime, params,
              compare_snap=None, compare_pressure_row=None,
              deform=True, deform_scale=DEFAULT_DEFORM_SCALE):

    p_field = build_field(pressure_dict)
    pred_vals = values_from_pressure_dict(pressure_dict)
    pts_dt = build_deformed_points(geom_pred, deform=deform, scale=deform_scale)

    peak_region = max(pressure_dict, key=lambda k: abs(pressure_dict[k]))
    peak_val    = pressure_dict[peak_region]

    regime_desc = {"low": "free breathing", "mid": "moderate resistance",
                   "high": "obstructed"}

    compare = compare_snap is not None and compare_pressure_row is not None

    if compare:
        real_vals = values_from_pressure_row(compare_pressure_row)
        common_clim = clim_from_values(pred_vals + real_vals)
        err_field, err_vals = build_abs_error_field(pressure_dict, compare_pressure_row)
        err_clim = [0.0, max(max(err_vals), 1e-9)]
        mean_abs_err = float(np.mean(err_vals))
        max_abs_err = float(np.max(err_vals))
        n_panels = 3
        window_size = (2100, 850)
    else:
        common_clim = clim_from_values(pred_vals)
        err_vals = None
        n_panels = 1
        window_size = (950, 900)

    pl = pv.Plotter(
        shape=(1, n_panels),
        title="Human Airways Digital Twin — Improved 3D Prediction",
        window_size=window_size
    )
    pl.background_color = "white"

    # ── Panel 1: Digital Twin prediction ─────────────────────────────────────
    pl.subplot(0, 0)
    cloud_dt = pv.PolyData(pts_dt.astype(np.float32))
    cloud_dt["Predicted Pressure (Pa)"] = p_field
    cloud_dt["Predicted Deformation (mm)"] = build_geom_field(geom_pred)

    pl.add_mesh(cloud_dt,
                scalars="Predicted Pressure (Pa)",
                cmap="RdBu_r",
                clim=common_clim,
                point_size=1.4,
                render_points_as_spheres=True,
                scalar_bar_args={
                    "title": "Pressure (Pa)", "fmt": "%.1f",
                    "color": "black", "title_font_size": 15,
                    "label_font_size": 12, "position_x": 0.05, "width": 0.18,
                })

    deform_note = "approx. deformed mesh" if deform else "baseline mesh"
    info = (f"  DIGITAL TWIN PREDICTION\n"
            f"  ─────────────────────────\n"
            f"  Regime: {regime.upper()} ({regime_desc[regime]})\n"
            f"  Peak: {abs(peak_val):.1f} Pa ({peak_region})\n"
            f"  Geometry: {deform_note}\n"
            f"  ─────────────────────────\n")
    for label, val in pressure_dict.items():
        sign = "+" if val >= 0 else ""
        info += f"  {label:<14s}: {sign}{val:.1f} Pa\n"
    info += f"  ─────────────────────────\n  Predicted: milliseconds\n  vs CFD: hours"

    pl.add_text(info, position="upper_left", font_size=11,
                color="black", font="courier")
    pl.add_text("DIGITAL TWIN — predicted regional pressure",
                position="upper_edge", font_size=13,
                color="#185FA5", font="courier")

    # ── Panel 2: Nearest real patient ────────────────────────────────────────
    if compare:
        pl.subplot(0, 1)

        r_field = build_field(compare_pressure_row, use_pres_cols=True)
        cloud_real = pv.PolyData(pts.astype(np.float32))
        cloud_real["Real Pressure (Pa)"] = r_field

        pl.add_mesh(cloud_real,
                    scalars="Real Pressure (Pa)",
                    cmap="RdBu_r",
                    clim=common_clim,
                    point_size=1.4,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        "title": "Pressure (Pa)", "fmt": "%.1f",
                        "color": "black", "title_font_size": 15,
                        "label_font_size": 12, "position_x": 0.39, "width": 0.18,
                    })

        snap_num   = int(compare_pressure_row["num"]) + 1
        mouth_real = float(compare_pressure_row.get("mouth_region_mean_Pa", 0))

        real_info = (f"  NEAREST REAL CFD PATIENT\n"
                     f"  Snapshot {snap_num}\n"
                     f"  ─────────────────────────\n"
                     f"  Mouth: {mouth_real:.1f} Pa\n"
                     f"  Same colour scale as prediction\n"
                     f"  ─────────────────────────\n")
        for col, lab in zip(PRES_COLS, REGION_LABELS):
            v = float(compare_pressure_row.get(col, 0))
            sign = "+" if v >= 0 else ""
            real_info += f"  {lab:<14s}: {sign}{v:.1f} Pa\n"

        pl.add_text(real_info, position="upper_left", font_size=11,
                    color="black", font="courier")
        pl.add_text("REFERENCE — nearest CFD simulation",
                    position="upper_edge", font_size=13,
                    color="#D85A30", font="courier")

        # ── Panel 3: Absolute error map ──────────────────────────────────────
        pl.subplot(0, 2)
        cloud_err = pv.PolyData(pts.astype(np.float32))
        cloud_err["Absolute Error (Pa)"] = err_field

        pl.add_mesh(cloud_err,
                    scalars="Absolute Error (Pa)",
                    cmap="viridis",
                    clim=err_clim,
                    point_size=1.4,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        "title": "|Pred - CFD| (Pa)", "fmt": "%.1f",
                        "color": "black", "title_font_size": 15,
                        "label_font_size": 12, "position_x": 0.72, "width": 0.18,
                    })

        err_info = (f"  REGIONAL ERROR MAP\n"
                    f"  ─────────────────────────\n"
                    f"  Mean abs error: {mean_abs_err:.2f} Pa\n"
                    f"  Max  abs error: {max_abs_err:.2f} Pa\n"
                    f"  ─────────────────────────\n")
        for lab, err in zip(REGION_LABELS, err_vals):
            err_info += f"  {lab:<14s}: {err:.2f} Pa\n"

        pl.add_text(err_info, position="upper_left", font_size=11,
                    color="black", font="courier")
        pl.add_text("ERROR MAP — regional |prediction - CFD|",
                    position="upper_edge", font_size=13,
                    color="#5A2CA0", font="courier")

        pl.link_views()

    print("[INFO] 3D window open. Mouse to rotate/zoom. Q to quit.")
    pl.show()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compare_mode = "--compare" in sys.argv
    deform = "--no-deform" not in sys.argv
    deform_scale = DEFAULT_DEFORM_SCALE
    if "--deform-scale" in sys.argv:
        try:
            deform_scale = float(sys.argv[sys.argv.index("--deform-scale") + 1])
        except (IndexError, ValueError):
            print("[WARN] Invalid --deform-scale value. Using default.")
            deform_scale = DEFAULT_DEFORM_SCALE

    # Interactive input
    print("\n" + "=" * 65)
    print("HUMAN AIRWAYS DIGITAL TWIN — 3D Visualisation")
    print("Enter patient parameters. ENTER = population mean.")
    print("=" * 65 + "\n")

    params = {}
    for col in doe_cols_all:
        mv = doe[col].mean(); mn = doe[col].min(); mx = doe[col].max()
        try:
            raw = input(f"  {col:<22s} [mean={mv:.2f}, range {mn:.2f}–{mx:.2f}]: ")
            params[col] = float(raw) if raw.strip() else mv
        except (ValueError, EOFError):
            params[col] = mv

    print("\n[INFO] Running prediction...")
    pressure_dict, geom_pred, regime = predict(params)

    regime_desc = {"low": "free breathing", "mid": "moderate resistance",
                   "high": "obstructed (narrow epiglottis)"}
    print(f"[INFO] Regime: {regime.upper()} — {regime_desc[regime]}")

    for label, val in pressure_dict.items():
        sign = "+" if val >= 0 else ""
        print(f"  {label:<18s} {sign}{val:.2f} Pa")

    peak_region = max(pressure_dict, key=lambda k: abs(pressure_dict[k]))
    print(f"\n  Peak: {abs(pressure_dict[peak_region]):.2f} Pa ({peak_region})")

    snap_idx = None
    compare_pressure_row = None
    if compare_mode:
        print("\n[INFO] Finding nearest real patient in same regime...")
        snap_idx, compare_pressure_row = find_nearest(params, regime)
        compare_pressure_row = compare_pressure_row.copy()
        compare_pressure_row["num"] = snap_idx
        print(f"[INFO] Nearest patient: snapshot {snap_idx + 1}  "
              f"(mouth={float(compare_pressure_row['mouth_region_mean_Pa']):.1f} Pa)")

    print("\n[INFO] Opening improved 3D visualisation...")
    if deform:
        print(f"[INFO] Approximate regional deformation enabled (scale={deform_scale}).")
    else:
        print("[INFO] Deformation disabled: showing pressure on baseline mesh.")

    visualise(pressure_dict, geom_pred, regime, params,
              compare_snap=snap_idx,
              compare_pressure_row=compare_pressure_row,
              deform=deform,
              deform_scale=deform_scale)