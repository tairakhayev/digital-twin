#!/usr/bin/env python3
"""
Human Airways Digital Twin — Hybrid full-field CFD-like pressure prediction

Hybrid amplitude–shape formulation:

    DOE parameters (26 anatomical inputs)
        -> pressure amplitude surrogate (physics-informed)
        -> normalized full-field pressure-shape POD coefficients
        -> full pressure field = predicted amplitude × predicted normalized shape
        -> stabilized 3D visualization

Why this version is more physically stable than direct DOE -> full-field pressure:
    1. The pressure level/amplitude is predicted separately.
    2. The spatial pressure distribution is learned as a normalized shape.
    3. A small physics-informed correction uses airway-area resistance proxies
       such as 1/A_glotis^2 and 1/A_epiglotis^2.

Recommended command:
    python digital_twin_hybrid_fullfield.py --rebuild --clim -20 150

After cache is created:
    python digital_twin_hybrid_fullfield.py --clim -20 150
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError:
    print("[ERROR] pyvista is not installed. Install it with:")
    print("        pip install pyvista")
    sys.exit(1)

try:
    from scipy.interpolate import RBFInterpolator
except ImportError:
    print("[ERROR] scipy is not installed. Install it with:")
    print("        pip install scipy")
    sys.exit(1)


# ---------------------------------------------------------------------
# File and binary utilities
# ---------------------------------------------------------------------

def find_existing_path(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def snapshot_number(path: Path) -> int:
    m = re.search(r"snapshot[_-]?(\d+)", path.stem)
    if not m:
        return 10**9
    return int(m.group(1))


def read_points_bin(path: Path) -> np.ndarray:
    """
    Read 3D mesh points from DiTiDE binary file.

    In this dataset, points.bin usually has:
        8-byte header + float64 XYZ coordinates
    """
    size = path.stat().st_size
    print(f"[DEBUG] points.bin size: {size} bytes")

    if (size - 8) % (3 * 8) == 0:
        with open(path, "rb") as f:
            f.seek(8)
            arr = np.fromfile(f, dtype=np.float64)
        pts = arr.reshape(-1, 3).astype(np.float32)
        print(f"[INFO] Read points with 8-byte header: {pts.shape[0]:,} points")
        return pts

    if size % (3 * 8) == 0:
        arr = np.fromfile(path, dtype=np.float64)
        pts = arr.reshape(-1, 3).astype(np.float32)
        print(f"[INFO] Read points as raw float64 XYZ: {pts.shape[0]:,} points")
        return pts

    if (size - 8) % (3 * 4) == 0:
        with open(path, "rb") as f:
            f.seek(8)
            arr = np.fromfile(f, dtype=np.float32)
        pts = arr.reshape(-1, 3)
        print(f"[INFO] Read points with 8-byte header as float32: {pts.shape[0]:,} points")
        return pts

    raise ValueError(f"Cannot read points file: {path}\nFile size = {size} bytes")


def read_pressure_snapshot(path: Path, n_points: int) -> np.ndarray:
    """
    Read one CFD pressure snapshot.

    Possible formats:
        raw float64 / header+float64 / raw float32 / header+float32
    """
    size = path.stat().st_size

    if size == n_points * 8:
        return np.fromfile(path, dtype=np.float64).astype(np.float32)

    if size == 8 + n_points * 8:
        with open(path, "rb") as f:
            f.seek(8)
            return np.fromfile(f, dtype=np.float64).astype(np.float32)

    if size == n_points * 4:
        return np.fromfile(path, dtype=np.float32)

    if size == 8 + n_points * 4:
        with open(path, "rb") as f:
            f.seek(8)
            return np.fromfile(f, dtype=np.float32)

    raise ValueError(
        f"Snapshot size mismatch: {path}\n"
        f"File size: {size} bytes\n"
        f"Expected one of {n_points*8}, {8+n_points*8}, "
        f"{n_points*4}, {8+n_points*4} bytes"
    )


def locate_dataset(base_dir: Path):
    points_path = find_existing_path([
        base_dir / "Pressure" / "points.bin",
        base_dir / "pressure" / "points.bin",
        base_dir / "Points" / "points.bin",
        base_dir / "points" / "points.bin",
        base_dir / "points.bin",
    ])
    if points_path is None:
        raise FileNotFoundError("Could not find points.bin")

    pressure_snap_dir = find_existing_path([
        base_dir / "Pressure" / "snapshots",
        base_dir / "pressure" / "snapshots",
        base_dir / "Pressure",
        base_dir / "pressure",
    ])
    if pressure_snap_dir is None:
        raise FileNotFoundError("Could not find Pressure/snapshots directory")

    snapshots = sorted(
        [p for p in pressure_snap_dir.glob("snapshot*.bin") if "point" not in p.name.lower()],
        key=snapshot_number,
    )
    if not snapshots:
        raise FileNotFoundError(f"No pressure snapshots found in {pressure_snap_dir}")

    print(f"[INFO] Points file: {points_path}")
    print(f"[INFO] Pressure snapshots: {len(snapshots)} files")
    return points_path, snapshots


# ---------------------------------------------------------------------
# DOE and feature engineering
# ---------------------------------------------------------------------

def load_doe(base_dir: Path):
    doe_path = find_existing_path([
        base_dir / "doe.csv",
        base_dir / "Points" / "doe.csv",
        base_dir / "points" / "doe.csv",
        base_dir / "Pressure" / "doe.csv",
        base_dir / "pressure" / "doe.csv",
    ])
    if doe_path is None:
        raise FileNotFoundError("Could not find doe.csv")

    print(f"[INFO] Loading DOE: {doe_path}")
    doe = pd.read_csv(doe_path, sep=None, engine="python")

    exclude = {
        "points", "pressure", "snapshot", "file", "filename",
        "inlet_vel", "part_size", "part_inj_vel",
    }

    numeric_cols = []
    for col in doe.columns:
        col_lower = str(col).strip().lower()
        if col_lower in exclude:
            continue
        if pd.api.types.is_numeric_dtype(doe[col]):
            numeric_cols.append(col)

    X = doe[numeric_cols].astype(float).values.astype(np.float64)

    print(f"[INFO] DOE shape: {doe.shape}")
    print(f"[INFO] Anatomical input columns: {len(numeric_cols)}")
    if len(numeric_cols) != 26:
        print("[WARNING] Expected 26 anatomical DOE parameters.")
        print(f"[WARNING] Found {len(numeric_cols)} numeric input columns:")
        for c in numeric_cols:
            print(f"          - {c}")

    return doe, numeric_cols, X


def find_column(columns, candidates):
    lower_map = {str(c).lower(): i for i, c in enumerate(columns)}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in lower_map:
            return lower_map[cand_l]
    for i, c in enumerate(columns):
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return i
    return None


def resistance_proxy_from_X(X: np.ndarray, columns):
    """
    Physics-inspired proxy for airway resistance.

    Narrower glottis/epiglottis should increase pressure losses. A simple
    stable proxy is based on inverse squared areas relative to population mean.
    """
    idx_g = find_column(columns, ["A_glotis", "glotis"])
    idx_e = find_column(columns, ["A_epiglotis", "epiglotis"])

    proxy = np.ones(X.shape[0], dtype=np.float64)
    parts = []

    if idx_g is not None:
        Ag = np.maximum(X[:, idx_g].astype(np.float64), 1e-8)
        Ag_mean = np.mean(Ag)
        parts.append((Ag_mean / Ag) ** 2)

    if idx_e is not None:
        Ae = np.maximum(X[:, idx_e].astype(np.float64), 1e-8)
        Ae_mean = np.mean(Ae)
        parts.append((Ae_mean / Ae) ** 2)

    if parts:
        proxy = np.mean(np.vstack(parts), axis=0)

    return proxy


def build_enhanced_features(X: np.ndarray, columns):
    """
    Use original DOE plus physics-inspired nonlinear area features.
    The final array is standardized later.
    """
    features = [X.astype(np.float64)]

    for key in ["A_glotis", "A_epiglotis"]:
        idx = find_column(columns, [key, key.replace("A_", "")])
        if idx is not None:
            A = np.maximum(X[:, idx].astype(np.float64), 1e-8)
            A_mean = np.mean(A)
            features.append((A_mean / A)[:, None])
            features.append(((A_mean / A) ** 2)[:, None])
            features.append(np.log(A)[:, None])

    proxy = resistance_proxy_from_X(X, columns)
    features.append(proxy[:, None])

    return np.hstack(features)


def standardize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


# ---------------------------------------------------------------------
# Hybrid amplitude-shape POD model
# ---------------------------------------------------------------------

def pressure_statistics(P: np.ndarray):
    return {
        "min": float(np.min(P)),
        "p001": float(np.percentile(P, 0.1)),
        "p01": float(np.percentile(P, 1.0)),
        "p05": float(np.percentile(P, 5.0)),
        "p50": float(np.percentile(P, 50.0)),
        "p95": float(np.percentile(P, 95.0)),
        "p99": float(np.percentile(P, 99.0)),
        "p999": float(np.percentile(P, 99.9)),
        "max": float(np.max(P)),
    }


def compute_pod_snapshot_method(S, energy_target=0.99, max_modes=20):
    """
    POD using method of snapshots.
    S shape: n_patients x n_points
    """
    print("[INFO] Computing normalized-shape POD using snapshot method...")
    n_samples, _ = S.shape

    mean_shape = S.mean(axis=0).astype(np.float32)
    Xc = S - mean_shape[None, :]

    print("[INFO] Building snapshot correlation matrix...")
    C = Xc @ Xc.T

    print("[INFO] Solving eigenvalue problem...")
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    total_energy = eigvals.sum()
    if total_energy <= 0:
        raise RuntimeError("POD failed: total energy is zero.")

    cumulative_energy = np.cumsum(eigvals) / total_energy
    k_energy = int(np.searchsorted(cumulative_energy, energy_target) + 1)
    k = min(k_energy, max_modes, n_samples)

    print(f"[INFO] Modes needed for {energy_target*100:.1f}% energy: {k_energy}")
    print(f"[INFO] Using k={k} normalized-shape modes")
    print(f"[INFO] Captured energy: {cumulative_energy[k-1]*100:.2f}%")

    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]
    singular_values = np.sqrt(eigvals_k)
    singular_values_safe = singular_values.copy()
    singular_values_safe[singular_values_safe == 0] = 1e-12

    print("[INFO] Computing spatial POD modes...")
    modes = (eigvecs_k.T @ Xc) / singular_values_safe[:, None]
    modes = modes.astype(np.float32)

    coeffs = eigvecs_k * singular_values_safe[None, :]
    coeffs = coeffs.astype(np.float64)

    return mean_shape, modes, coeffs, eigvals, cumulative_energy, k


def compute_training_amplitudes(P: np.ndarray, amplitude_percentile: float):
    """
    Compute robust pressure amplitude for each training snapshot.
    Default p99 is more stable than global max.
    """
    amps = np.percentile(P, amplitude_percentile, axis=1).astype(np.float64)

    # Avoid zero or negative amplitudes; use robust positive fallback.
    positive = amps[amps > 1e-8]
    fallback = float(np.median(positive)) if positive.size else 1.0
    amps = np.where(amps > 1e-8, amps, fallback)
    return amps


def load_or_build_hybrid_model(base_dir, snapshots, n_points, args):
    cache_path = base_dir / "hybrid_fullfield_shape_pod_cache.npz"

    if cache_path.exists() and not args.rebuild:
        print(f"[INFO] Loading cached hybrid POD model: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        mean_shape = data["mean_shape"]
        modes = data["modes"]
        coeffs = data["coeffs"]
        eigvals = data["eigvals"]
        cumulative_energy = data["cumulative_energy"]
        k = int(data["k"])
        amplitudes = data["amplitudes"]
        pressure_stats = data["pressure_stats"].item()
        shape_stats = data["shape_stats"].item()
        coeff_mean = data["coeff_mean"]
        coeff_std = data["coeff_std"]
        coeff_min = data["coeff_min"]
        coeff_max = data["coeff_max"]

        print(f"[INFO] Loaded normalized-shape POD k={k}")
        print(f"[INFO] Captured energy: {cumulative_energy[k-1]*100:.2f}%")
        return (
            mean_shape, modes, coeffs, eigvals, cumulative_energy, k,
            amplitudes, pressure_stats, shape_stats,
            coeff_mean, coeff_std, coeff_min, coeff_max,
        )

    print("[INFO] Loading full pressure snapshots into memory...")
    P = np.empty((len(snapshots), n_points), dtype=np.float32)
    for i, snap in enumerate(snapshots):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  loading {i + 1}/{len(snapshots)}: {snap.name}")
        P[i, :] = read_pressure_snapshot(snap, n_points)

    pressure_stats = pressure_statistics(P)
    print("[INFO] Training full-field pressure statistics:")
    print(
        f"       p01={pressure_stats['p01']:.2f} Pa, "
        f"p99={pressure_stats['p99']:.2f} Pa, "
        f"p99.9={pressure_stats['p999']:.2f} Pa"
    )

    amplitudes = compute_training_amplitudes(P, args.amplitude_percentile)
    print("[INFO] Training amplitudes:")
    print(
        f"       percentile=p{args.amplitude_percentile:g}, "
        f"min={amplitudes.min():.2f}, median={np.median(amplitudes):.2f}, "
        f"max={amplitudes.max():.2f} Pa"
    )

    print("[INFO] Building normalized pressure shapes: shape = pressure / amplitude")
    S = P / amplitudes[:, None].astype(np.float32)

    shape_stats = pressure_statistics(S)
    print("[INFO] Normalized-shape statistics:")
    print(
        f"       p01={shape_stats['p01']:.3f}, "
        f"p99={shape_stats['p99']:.3f}, p99.9={shape_stats['p999']:.3f}"
    )

    mean_shape, modes, coeffs, eigvals, cumulative_energy, k = compute_pod_snapshot_method(
        S,
        energy_target=args.energy,
        max_modes=args.modes,
    )

    coeff_mean = coeffs.mean(axis=0)
    coeff_std = coeffs.std(axis=0)
    coeff_min = coeffs.min(axis=0)
    coeff_max = coeffs.max(axis=0)

    print(f"[INFO] Saving hybrid POD cache: {cache_path}")
    np.savez(
        cache_path,
        mean_shape=mean_shape,
        modes=modes,
        coeffs=coeffs,
        eigvals=eigvals,
        cumulative_energy=cumulative_energy,
        k=np.array(k),
        amplitudes=amplitudes,
        pressure_stats=np.array(pressure_stats, dtype=object),
        shape_stats=np.array(shape_stats, dtype=object),
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
    )

    return (
        mean_shape, modes, coeffs, eigvals, cumulative_energy, k,
        amplitudes, pressure_stats, shape_stats,
        coeff_mean, coeff_std, coeff_min, coeff_max,
    )


def train_rbf_model(Xs, Y, kernel="thin_plate_spline", smoothing=1e-4, label="RBF"):
    print(f"[INFO] Training {label}")
    if kernel in {"linear", "thin_plate_spline", "cubic", "quintic"}:
        model = RBFInterpolator(Xs, Y, kernel=kernel, smoothing=smoothing)
    else:
        model = RBFInterpolator(Xs, Y, kernel=kernel, smoothing=smoothing, epsilon=1.0)
    print(f"[✓] {label} trained")
    return model


def ask_patient_parameters(doe, columns):
    print()
    print("=" * 72)
    print("HUMAN AIRWAYS DIGITAL TWIN — HYBRID FULL-FIELD PREDICTION")
    print("Input : 26 anatomical DOE parameters")
    print("Output: pressure value on every mesh point")
    print("Model : pressure amplitude × normalized pressure shape")
    print("=" * 72)
    print("Enter patient parameters. ENTER = population mean.")
    print()

    values = []
    for col in columns:
        mean_val = float(doe[col].mean())
        min_val = float(doe[col].min())
        max_val = float(doe[col].max())

        while True:
            raw = input(f"  {col:<22} [mean={mean_val:.2f}, range {min_val:.2f}–{max_val:.2f}]: ").strip()
            if raw == "":
                val = mean_val
                break
            try:
                val = float(raw)
                break
            except ValueError:
                print("    Please enter a number or press ENTER.")

        if val < min_val or val > max_val:
            print(
                f"    [WARNING] {col}={val:.2f} is outside training range "
                f"({min_val:.2f}–{max_val:.2f}). Prediction is extrapolation."
            )
        values.append(val)

    return np.array(values, dtype=np.float64)


def predict_hybrid_pressure(
    patient_x,
    columns,
    feature_mean,
    feature_std,
    amp_model,
    shape_model,
    mean_shape,
    modes,
    amplitudes_train,
    pressure_stats,
    shape_stats,
    coeff_mean,
    coeff_std,
    coeff_min,
    coeff_max,
    X_train_original,
    args,
):
    """
    Predict a stabilized full-field pressure using the hybrid model.

    Model:
        pressure_field = amplitude × normalized_shape

    Stability controls:
        1. RBF amplitude is predicted in log-space.
        2. Predicted log-amplitude is clipped to the training log-amplitude range
           before exp(), preventing numerical explosion.
        3. Physics proxy controls amplitude sensitivity to airway area.
        4. Patient resistance proxy is computed relative to TRAINING means,
           not relative to the patient itself.
        5. Shape POD coefficients are constrained to the training coefficient range.
        6. Shape and final pressure are clipped to robust training percentiles.
    """

    # Build and standardize enhanced features for the new patient.
    Xp = patient_x[None, :]
    Xp_enh = build_enhanced_features(Xp, columns)
    Xp_scaled = (Xp_enh - feature_mean[None, :]) / feature_std[None, :]

    # ------------------------------------------------------------
    # 1) Predict pressure amplitude
    # ------------------------------------------------------------

    # RBF amplitude prediction in log-space.
    # IMPORTANT: clip log-amplitude before exp(), otherwise exp() can explode.
    log_amp_raw = float(np.ravel(amp_model(Xp_scaled))[0])

    log_amplitudes_train = np.log(np.maximum(amplitudes_train, 1e-8))
    log_amp_min = float(np.min(log_amplitudes_train))
    log_amp_max = float(np.max(log_amplitudes_train))
    log_margin = 0.10 * (log_amp_max - log_amp_min)

    log_amp_clipped = float(np.clip(
        log_amp_raw,
        log_amp_min - log_margin,
        log_amp_max + log_margin,
    ))

    amp_rbf = float(np.exp(log_amp_clipped))

    # ------------------------------------------------------------
    # 1A) Physics-informed amplitude correction
    # ------------------------------------------------------------
    # Smaller A_glotis / A_epiglotis -> larger resistance proxy -> larger amplitude.
    # IMPORTANT:
    # patient proxy must be computed relative to TRAINING reference means.
    # Do NOT use resistance_proxy_from_X(Xp, columns), because for one patient
    # its internal mean becomes the patient value itself and proxy becomes ~1.

    idx_g = find_column(columns, ["A_glotis", "glotis"])
    idx_e = find_column(columns, ["A_epiglotis", "epiglotis"])

    proxy_train_parts = []
    proxy_patient_parts = []

    if idx_g is not None:
        Ag_train = np.maximum(X_train_original[:, idx_g].astype(np.float64), 1e-8)
        Ag_mean = float(np.mean(Ag_train))
        Ag_patient = max(float(patient_x[idx_g]), 1e-8)

        proxy_train_parts.append((Ag_mean / Ag_train) ** 2)
        proxy_patient_parts.append((Ag_mean / Ag_patient) ** 2)

    if idx_e is not None:
        Ae_train = np.maximum(X_train_original[:, idx_e].astype(np.float64), 1e-8)
        Ae_mean = float(np.mean(Ae_train))
        Ae_patient = max(float(patient_x[idx_e]), 1e-8)

        proxy_train_parts.append((Ae_mean / Ae_train) ** 2)
        proxy_patient_parts.append((Ae_mean / Ae_patient) ** 2)

    if proxy_train_parts:
        proxy_train = np.mean(np.vstack(proxy_train_parts), axis=0)
        proxy_patient = float(np.mean(proxy_patient_parts))
    else:
        proxy_train = np.ones(X_train_original.shape[0], dtype=np.float64)
        proxy_patient = 1.0

    proxy_median = float(np.median(proxy_train))
    if proxy_median <= 0 or not np.isfinite(proxy_median):
        proxy_median = 1.0

    proxy_ratio = float(proxy_patient / max(proxy_median, 1e-12))
    proxy_ratio_limited = float(np.clip(proxy_ratio, 0.30, 3.00))

    amp_ref = float(np.median(amplitudes_train))
    amp_physics = amp_ref * proxy_ratio_limited

    # Blend data-driven amplitude and physics-informed amplitude.
    # Use log-space blending for positive and smooth amplitude.
    blend = float(np.clip(args.physics_blend, 0.0, 1.0))

    amplitude_blended = float(np.exp(
        (1.0 - blend) * np.log(max(amp_rbf, 1e-8))
        + blend * np.log(max(amp_physics, 1e-8))
    ))

    # Robust amplitude limits from training data.
    amp_low = float(np.percentile(amplitudes_train, 0.5))
    amp_high = float(np.percentile(amplitudes_train, 99.5))
    amplitude_final = float(np.clip(amplitude_blended, amp_low, amp_high))

    # ------------------------------------------------------------
    # 2) Predict normalized pressure shape
    # ------------------------------------------------------------

    coeff_raw = np.ravel(shape_model(Xp_scaled))
    coeff_pred = coeff_raw.copy()

    if not args.no_stabilize:
        # Sigma-based coefficient constraint
        lower_sigma = coeff_mean - args.coeff_sigma_limit * coeff_std
        upper_sigma = coeff_mean + args.coeff_sigma_limit * coeff_std
        coeff_pred = np.clip(coeff_pred, lower_sigma, upper_sigma)

        # Training-range coefficient constraint with small margin
        margin = 0.10 * (coeff_max - coeff_min)
        coeff_pred = np.clip(coeff_pred, coeff_min - margin, coeff_max + margin)

    shape_raw = mean_shape.astype(np.float64) + coeff_raw @ modes.astype(np.float64)
    shape = mean_shape.astype(np.float64) + coeff_pred @ modes.astype(np.float64)

    if not args.no_stabilize:
        shape = np.clip(shape, shape_stats["p001"], shape_stats["p999"])

    # ------------------------------------------------------------
    # 3) Reconstruct full pressure field
    # ------------------------------------------------------------

    pressure_raw = amp_rbf * shape_raw
    pressure = amplitude_final * shape

    if not args.no_stabilize:
        pressure = np.clip(pressure, pressure_stats["p001"], pressure_stats["p999"])

    diagnostics = {
        "log_amp_raw": log_amp_raw,
        "log_amp_clipped": log_amp_clipped,

        "amp_rbf": amp_rbf,

        # Two names for compatibility with different print blocks
        "amp_physics": amp_physics,
        "amp_proxy": amp_physics,

        "amp_blended": amplitude_blended,
        "amplitude_blended": amplitude_blended,

        "amp_pred": amplitude_final,
        "amplitude_final": amplitude_final,

        "amp_low": amp_low,
        "amp_high": amp_high,

        "proxy_patient": proxy_patient,
        "proxy_median": proxy_median,
        "proxy_ratio": proxy_ratio,
        "proxy_ratio_limited": proxy_ratio_limited,
    }

    return (
        pressure.astype(np.float32),
        pressure_raw.astype(np.float32),
        shape.astype(np.float32),
        diagnostics,
    )
# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def print_pressure_summary(label, pressure):
    vals = np.percentile(pressure, [0.1, 1, 5, 50, 95, 99, 99.9])
    print(f"[INFO] {label} pressure statistics:")
    print(f"       min={pressure.min():.2f}, max={pressure.max():.2f} Pa")
    print(
        f"       p0.1={vals[0]:.2f}, p1={vals[1]:.2f}, p5={vals[2]:.2f}, "
        f"p50={vals[3]:.2f}, p95={vals[4]:.2f}, p99={vals[5]:.2f}, p99.9={vals[6]:.2f}"
    )


def visualize_fullfield(points, pressure, diagnostics, args):
    print("[INFO] Creating PyVista point cloud...")
    cloud = pv.PolyData(points)
    cloud["Predicted pressure (Pa)"] = pressure

    pmin = float(np.nanpercentile(pressure, 1))
    pmax = float(np.nanpercentile(pressure, 99))
    if args.clim is not None:
        pmin, pmax = args.clim

    p99_text = float(np.nanpercentile(pressure, 99))
    p999_text = float(np.nanpercentile(pressure, 99.9))

    print(f"[INFO] Color scale: {pmin:.2f} / {pmax:.2f} Pa")

    plotter = pv.Plotter(window_size=(1100, 900))
    plotter.set_background("white")

    plotter.add_text(
        "DIGITAL TWIN — hybrid full-field CFD-like pressure",
        position="upper_left",
        font_size=13,
        color="black",
    )

    plotter.add_text(
        f"Points: {len(points):,}\n"
        f"Amplitude: {diagnostics['amp_pred']:.2f} Pa\n"
        f"Representative p99: {p99_text:.2f} Pa\n"
        f"High-pressure p99.9: {p999_text:.2f} Pa\n"
        f"Model: amplitude × normalized POD shape",
        position=(20, 745),
        font_size=10,
        color="black",
    )

    plotter.add_mesh(
        cloud,
        scalars="Predicted pressure (Pa)",
        cmap="coolwarm",
        clim=[pmin, pmax],
        point_size=args.point_size,
        render_points_as_spheres=False,
        show_scalar_bar=False,
    )

    plotter.add_scalar_bar(title="Pressure (Pa)", n_labels=6, fmt="%.1f")
    plotter.camera_position = "xz"
    plotter.camera.zoom(1.4)

    if args.screenshot:
        print(f"[INFO] Saving screenshot: {args.screenshot}")
        plotter.show(screenshot=args.screenshot)
    else:
        print("[INFO] Opening 3D full-field visualization...")
        print("[INFO] Mouse to rotate/zoom. Press Q to quit.")
        plotter.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid amplitude-shape Human Airways Digital Twin full-field prediction"
    )
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--energy", type=float, default=0.99)
    parser.add_argument("--kernel", type=str, default="thin_plate_spline")
    parser.add_argument("--smoothing", type=float, default=1e-4)
    parser.add_argument("--amplitude-smoothing", type=float, default=1e-3)
    parser.add_argument("--amplitude-percentile", type=float, default=99.0)
    parser.add_argument("--physics-blend", type=float, default=0.40,
                        help="Blend between data-driven amplitude and physics proxy. 0=data only, 1=physics proxy only.")
    parser.add_argument("--coeff-sigma-limit", type=float, default=3.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-stabilize", action="store_true")
    parser.add_argument("--screenshot", type=str, default="digital_twin_hybrid_fullfield_prediction.png")
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument("--clim", type=float, nargs=2, default=None)
    args = parser.parse_args()

    t0 = time.time()
    base_dir = Path(args.base_dir).resolve()
    print(f"[INFO] Base directory: {base_dir}")

    points_path, snapshots = locate_dataset(base_dir)

    print("[INFO] Loading mesh points...")
    points = read_points_bin(points_path)
    n_points = points.shape[0]
    print(f"[INFO] Mesh points: {n_points:,}")

    doe, doe_columns, X = load_doe(base_dir)
    if X.shape[0] != len(snapshots):
        print("[WARNING] Number of DOE rows and pressure snapshots differs.")
        n = min(X.shape[0], len(snapshots))
        print(f"          DOE rows: {X.shape[0]}, snapshots: {len(snapshots)}")
        print(f"          Using first {n} samples.")
        X = X[:n]
        doe = doe.iloc[:n].copy()
        snapshots = snapshots[:n]

    X_enh = build_enhanced_features(X, doe_columns)
    Xs, feature_mean, feature_std = standardize_features(X_enh)
    print(f"[INFO] Enhanced feature dimension: {X_enh.shape[1]}")

    (
        mean_shape, modes, shape_coeffs, eigvals, cumulative_energy, k,
        amplitudes, pressure_stats, shape_stats,
        coeff_mean, coeff_std, coeff_min, coeff_max,
    ) = load_or_build_hybrid_model(base_dir, snapshots, n_points, args)

    # Train two surrogates:
    #   1) amplitude in log-space
    #   2) normalized shape POD coefficients
    log_amp = np.log(np.maximum(amplitudes, 1e-8))[:, None]
    amp_model = train_rbf_model(
        Xs,
        log_amp,
        kernel=args.kernel,
        smoothing=args.amplitude_smoothing,
        label="amplitude RBF: DOE → log(amplitude)",
    )
    shape_model = train_rbf_model(
        Xs,
        shape_coeffs,
        kernel=args.kernel,
        smoothing=args.smoothing,
        label="shape RBF: DOE → normalized pressure-shape POD coefficients",
    )

    print()
    print("[✓] Hybrid full-field Digital Twin ready.")
    print(f"[INFO] Normalized-shape POD modes: {k}")
    print(f"[INFO] Captured shape energy: {cumulative_energy[k-1]*100:.2f}%")
    print(f"[INFO] Amplitude percentile: p{args.amplitude_percentile:g}")
    print(f"[INFO] Physics blend: {args.physics_blend:.2f}")
    print(f"[INFO] Setup time: {time.time() - t0:.1f} seconds")

    patient_x = ask_patient_parameters(doe, doe_columns)

    print()
    print("[INFO] Running hybrid full-field Digital Twin prediction...")
    tp = time.time()
    pressure, pressure_raw, shape, diagnostics = predict_hybrid_pressure(
        patient_x,
        doe_columns,
        feature_mean,
        feature_std,
        amp_model,
        shape_model,
        mean_shape,
        modes,
        amplitudes,
        pressure_stats,
        shape_stats,
        coeff_mean,
        coeff_std,
        coeff_min,
        coeff_max,
        X,
        args,
    )
    pred_time = time.time() - tp

    print("[✓] Prediction complete")
    print(f"[INFO] Prediction time: {pred_time*1000:.2f} ms")
    print(f"[INFO] Pressure field size: {pressure.shape[0]:,} points")
    print("[INFO] Amplitude diagnostics:")
    print(f"       RBF amplitude     : {diagnostics['amp_rbf']:.2f} Pa")
    print(f"       physics amplitude : {diagnostics['amp_physics']:.2f} Pa")
    print(f"       blended amplitude : {diagnostics['amp_blended']:.2f} Pa")
    print(f"       final amplitude   : {diagnostics['amp_pred']:.2f} Pa")
    print(f"       proxy patient/median: {diagnostics['proxy_patient']:.3f} / {diagnostics['proxy_median']:.3f}")

    print_pressure_summary("Raw", pressure_raw)
    if not args.no_stabilize:
        print("[INFO] Stabilization enabled:")
        print("       shape POD coefficients constrained to training range")
        print("       normalized shape clipped to robust training percentiles")
        print("       pressure clipped to robust training pressure percentiles")
    else:
        print("[WARNING] Stabilization disabled by --no-stabilize")
    print_pressure_summary("Final", pressure)

    out_bin = base_dir / "digital_twin_hybrid_fullfield_pressure.bin"
    pressure.astype(np.float32).tofile(out_bin)
    print(f"[INFO] Saved pressure field: {out_bin}")

    visualize_fullfield(points, pressure, diagnostics, args)


if __name__ == "__main__":
    main()
