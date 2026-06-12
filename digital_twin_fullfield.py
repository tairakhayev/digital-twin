#!/usr/bin/env python3
"""
Human Airways Digital Twin — Full-field CFD-like pressure prediction

This script builds a Reduced Order Model using full CFD pressure snapshots.

Pipeline:
    DOE parameters (26 anatomical inputs)
        -> RBF surrogate
        -> full pressure POD coefficients
        -> reconstruction of pressure on all mesh points
        -> stabilized full-field reconstruction of pressure on all mesh points
        -> 3D visualization

This is stronger than regional visualization because the output is not only
11 regional pressure values, but a pressure value for every mesh point.

Stabilization is included to prevent non-physical local spikes:
    - predicted POD coefficients are limited to the training coefficient range
    - reconstructed pressures are clipped to robust training percentiles
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
# Utility functions
# ---------------------------------------------------------------------

def find_existing_path(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def snapshot_number(path: Path) -> int:
    """
    Extract patient number from filenames like:
    snapshot_1.bin, snapshot1.bin, snapshot_001.bin
    """
    m = re.search(r"snapshot[_-]?(\d+)", path.stem)
    if not m:
        return 10**9
    return int(m.group(1))


def read_points_bin(path: Path) -> np.ndarray:
    """
    Read 3D mesh points from DiTiDE binary file.

    In this dataset, points.bin has:
        8-byte header + float64 XYZ coordinates

    Number of points:
        (file_size - 8) / (3 * 8)
    """
    size = path.stat().st_size
    print(f"[DEBUG] points.bin size: {size} bytes")

    # DiTiDE format: 8-byte header + float64 XYZ
    if (size - 8) % (3 * 8) == 0:
        with open(path, "rb") as f:
            f.seek(8)
            arr = np.fromfile(f, dtype=np.float64)

        pts = arr.reshape(-1, 3).astype(np.float32)
        print(f"[INFO] Read points with 8-byte header: {pts.shape[0]:,} points")
        return pts

    # Fallback: raw float64 XYZ without header
    if size % (3 * 8) == 0:
        arr = np.fromfile(path, dtype=np.float64)
        pts = arr.reshape(-1, 3).astype(np.float32)
        print(f"[INFO] Read points as raw float64 XYZ: {pts.shape[0]:,} points")
        return pts

    raise ValueError(
        f"Cannot read points file: {path}\n"
        f"File size = {size} bytes."
    )


def read_pressure_snapshot(path: Path, n_points: int) -> np.ndarray:
    """
    Read one CFD pressure snapshot.

    Possible formats:
        1) raw float64, length = n_points
        2) 8-byte header + float64, length = n_points
        3) raw float32, length = n_points
        4) 8-byte header + float32, length = n_points
    """
    size = path.stat().st_size

    # raw float64
    if size == n_points * 8:
        return np.fromfile(path, dtype=np.float64).astype(np.float32)

    # 8-byte header + float64
    if size == 8 + n_points * 8:
        with open(path, "rb") as f:
            f.seek(8)
            return np.fromfile(f, dtype=np.float64).astype(np.float32)

    # raw float32
    if size == n_points * 4:
        return np.fromfile(path, dtype=np.float32)

    # 8-byte header + float32
    if size == 8 + n_points * 4:
        with open(path, "rb") as f:
            f.seek(8)
            return np.fromfile(f, dtype=np.float32)

    raise ValueError(
        f"Snapshot size mismatch: {path}\n"
        f"File size: {size} bytes\n"
        f"Expected one of:\n"
        f"  {n_points * 8} bytes raw float64\n"
        f"  {8 + n_points * 8} bytes header + float64\n"
        f"  {n_points * 4} bytes raw float32\n"
        f"  {8 + n_points * 4} bytes header + float32"
    )

def load_doe(base_dir: Path):
    """
    Load DOE table and return only 26 anatomical numerical parameters.

    This avoids using extra simulation parameters such as:
        inlet_vel, part_size, part_inj_vel
    if they exist in another DOE file.
    """
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

    # sep=None can detect comma or semicolon
    doe = pd.read_csv(doe_path, sep=None, engine="python")

    # Remove columns that are not anatomical inputs
    exclude = {
        "points",
        "pressure",
        "snapshot",
        "file",
        "filename",
        "inlet_vel",
        "part_size",
        "part_inj_vel",
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


def locate_dataset(base_dir: Path):
    """
    Locate mesh points and pressure snapshots.
    """
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
        list(pressure_snap_dir.glob("snapshot*.bin")),
        key=snapshot_number
    )

    # Remove points.bin if it was included accidentally
    snapshots = [p for p in snapshots if "point" not in p.name.lower()]

    if len(snapshots) == 0:
        raise FileNotFoundError(f"No pressure snapshots found in {pressure_snap_dir}")

    print(f"[INFO] Points file: {points_path}")
    print(f"[INFO] Pressure snapshots: {len(snapshots)} files")

    return points_path, snapshots


def standardize_features(X):
    """
    Standardize DOE features manually.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


def compute_pod_snapshot_method(P, energy_target=0.99, max_modes=20):
    """
    Compute POD using the method of snapshots.

    P shape:
        n_patients x n_points

    This avoids calculating a huge SVD directly.
    """
    print("[INFO] Computing full-field pressure POD using snapshot method...")

    n_samples, n_points = P.shape

    mean_field = P.mean(axis=0).astype(np.float32)
    Xc = P - mean_field[None, :]

    print("[INFO] Building snapshot correlation matrix...")
    C = Xc @ Xc.T

    print("[INFO] Solving eigenvalue problem...")
    eigvals, eigvecs = np.linalg.eigh(C)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.maximum(eigvals, 0.0)

    total_energy = eigvals.sum()
    if total_energy <= 0:
        raise RuntimeError("POD failed: total energy is zero.")

    cumulative_energy = np.cumsum(eigvals) / total_energy

    k_energy = int(np.searchsorted(cumulative_energy, energy_target) + 1)
    k = min(k_energy, max_modes, n_samples)

    print(f"[INFO] Modes needed for {energy_target * 100:.1f}% energy: {k_energy}")
    print(f"[INFO] Using k={k} pressure modes")
    print(f"[INFO] Captured energy: {cumulative_energy[k - 1] * 100:.2f}%")

    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]

    singular_values = np.sqrt(eigvals_k)

    # Avoid division by zero
    singular_values_safe = singular_values.copy()
    singular_values_safe[singular_values_safe == 0] = 1e-12

    print("[INFO] Computing spatial POD modes...")
    modes = (eigvecs_k.T @ Xc) / singular_values_safe[:, None]
    modes = modes.astype(np.float32)

    # POD coefficients for training patients
    coeffs = eigvecs_k * singular_values_safe[None, :]
    coeffs = coeffs.astype(np.float64)

    # Training pressure statistics for stabilization.
    # We keep robust percentiles because absolute min/max may contain local CFD spikes.
    print("[INFO] Computing training pressure statistics...")
    pressure_stats = {
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

    coeff_mean = coeffs.mean(axis=0)
    coeff_std = coeffs.std(axis=0)
    coeff_min = coeffs.min(axis=0)
    coeff_max = coeffs.max(axis=0)

    print("[INFO] Training pressure statistics:")
    print(
        f"       p01={pressure_stats['p01']:.2f} Pa, "
        f"p99={pressure_stats['p99']:.2f} Pa, "
        f"min={pressure_stats['min']:.2f} Pa, "
        f"max={pressure_stats['max']:.2f} Pa"
    )

    return (
        mean_field,
        modes,
        coeffs,
        eigvals,
        cumulative_energy,
        k,
        pressure_stats,
        coeff_mean,
        coeff_std,
        coeff_min,
        coeff_max,
    )


def load_or_build_pod_model(base_dir, snapshots, n_points, args):
    """
    Load cached POD model if available, otherwise build it.
    Also stores pressure statistics and POD coefficient limits
    for stable prediction.
    """
    cache_path = base_dir / "fullfield_pressure_pod_cache.npz"

    if cache_path.exists() and not args.rebuild:
        print(f"[INFO] Loading cached POD model: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        mean_field = data["mean_field"]
        modes = data["modes"]
        coeffs = data["coeffs"]
        eigvals = data["eigvals"]
        cumulative_energy = data["cumulative_energy"]
        k = int(data["k"])

        # Backward compatibility: old cache may not contain stabilization data.
        if "pressure_stats" in data.files:
            pressure_stats = data["pressure_stats"].item()
            coeff_mean = data["coeff_mean"]
            coeff_std = data["coeff_std"]
            coeff_min = data["coeff_min"]
            coeff_max = data["coeff_max"]
        else:
            print("[WARNING] Old POD cache found without stabilization data.")
            print("[WARNING] Rebuild cache using:")
            print("          python digital_twin_fullfield.py --rebuild")
            pressure_stats = None
            coeff_mean = coeffs.mean(axis=0)
            coeff_std = coeffs.std(axis=0)
            coeff_min = coeffs.min(axis=0)
            coeff_max = coeffs.max(axis=0)

        print(f"[INFO] Loaded full-field POD k={k}")
        print(f"[INFO] Captured energy: {cumulative_energy[k - 1] * 100:.2f}%")

        return (
            mean_field,
            modes,
            coeffs,
            eigvals,
            cumulative_energy,
            k,
            pressure_stats,
            coeff_mean,
            coeff_std,
            coeff_min,
            coeff_max,
        )

    print("[INFO] Loading full pressure snapshots into memory...")
    P = np.empty((len(snapshots), n_points), dtype=np.float32)

    for i, snap in enumerate(snapshots):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  loading {i + 1}/{len(snapshots)}: {snap.name}")
        P[i, :] = read_pressure_snapshot(snap, n_points)

    (
        mean_field,
        modes,
        coeffs,
        eigvals,
        cumulative_energy,
        k,
        pressure_stats,
        coeff_mean,
        coeff_std,
        coeff_min,
        coeff_max,
    ) = compute_pod_snapshot_method(
        P,
        energy_target=args.energy,
        max_modes=args.modes,
    )

    print(f"[INFO] Saving POD cache: {cache_path}")
    np.savez(
        cache_path,
        mean_field=mean_field,
        modes=modes,
        coeffs=coeffs,
        eigvals=eigvals,
        cumulative_energy=cumulative_energy,
        k=np.array(k),
        pressure_stats=np.array(pressure_stats, dtype=object),
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
    )

    return (
        mean_field,
        modes,
        coeffs,
        eigvals,
        cumulative_energy,
        k,
        pressure_stats,
        coeff_mean,
        coeff_std,
        coeff_min,
        coeff_max,
    )


def train_rbf_surrogate(Xs, coeffs, kernel="thin_plate_spline", smoothing=1e-8):
    """
    Train RBF surrogate:
        standardized DOE -> full-field pressure POD coefficients
    """
    print("[INFO] Training RBF surrogate: DOE → full-field pressure POD coefficients")

    # RBFInterpolator supports multi-output targets.
    if kernel in {"linear", "thin_plate_spline", "cubic", "quintic"}:
        model = RBFInterpolator(
            Xs,
            coeffs,
            kernel=kernel,
            smoothing=smoothing,
        )
    else:
        # kernels such as gaussian/multiquadric need epsilon
        model = RBFInterpolator(
            Xs,
            coeffs,
            kernel=kernel,
            smoothing=smoothing,
            epsilon=1.0,
        )

    print("[✓] RBF surrogate trained")
    return model


def ask_patient_parameters(doe, columns):
    """
    Interactive input of 26 anatomical DOE parameters.
    ENTER = population mean.
    """
    print()
    print("=" * 65)
    print("HUMAN AIRWAYS DIGITAL TWIN — FULL-FIELD CFD-LIKE PREDICTION")
    print("Input : 26 anatomical DOE parameters")
    print("Output: pressure value on every mesh point")
    print("=" * 65)
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


def reconstruct_pressure_field(
    patient_x,
    feature_mean,
    feature_std,
    rbf_model,
    mean_field,
    modes,
    pressure_stats=None,
    coeff_mean=None,
    coeff_std=None,
    coeff_min=None,
    coeff_max=None,
    coeff_sigma_limit=3.0,
    use_stabilization=True,
):
    """
    Predict POD coefficients and reconstruct full pressure field.

    Stabilization:
    1. Predicted POD coefficients are limited to the training coefficient range.
    2. Final pressure field is clipped to robust training pressure percentiles.

    This prevents unrealistic local spikes caused by RBF extrapolation while
    preserving the global POD pressure pattern.
    """
    x_scaled = (patient_x[None, :] - feature_mean[None, :]) / feature_std[None, :]

    coeff_raw = rbf_model(x_scaled)[0]
    coeff_pred = coeff_raw.copy()

    if use_stabilization:
        # First safety layer: sigma-based coefficient limit.
        if coeff_mean is not None and coeff_std is not None:
            lower_sigma = coeff_mean - coeff_sigma_limit * coeff_std
            upper_sigma = coeff_mean + coeff_sigma_limit * coeff_std
            coeff_pred = np.clip(coeff_pred, lower_sigma, upper_sigma)

        # Second safety layer: training min/max coefficient range with 10% margin.
        if coeff_min is not None and coeff_max is not None:
            margin = 0.10 * (coeff_max - coeff_min)
            coeff_pred = np.clip(coeff_pred, coeff_min - margin, coeff_max + margin)

    # Raw reconstruction is kept for diagnostics.
    pressure_raw = mean_field.astype(np.float64) + coeff_raw @ modes.astype(np.float64)

    # Stabilized reconstruction is used for saving and visualization.
    pressure = mean_field.astype(np.float64) + coeff_pred @ modes.astype(np.float64)

    if use_stabilization and pressure_stats is not None:
        # Robust training range: avoids non-physical pointwise spikes.
        p_low = pressure_stats["p001"]
        p_high = pressure_stats["p999"]
        pressure = np.clip(pressure, p_low, p_high)

    return (
        pressure.astype(np.float32),
        coeff_pred,
        coeff_raw,
        pressure_raw.astype(np.float32),
    )


def visualize_fullfield(points, pressure, args):
    """
    Visualize pressure field on all mesh points.
    """
    print("[INFO] Creating PyVista point cloud...")

    cloud = pv.PolyData(points)
    cloud["Predicted pressure (Pa)"] = pressure

    pmin = float(np.nanpercentile(pressure, 1))
    pmax = float(np.nanpercentile(pressure, 99))

    if args.clim is not None:
        pmin, pmax = args.clim

    print(f"[INFO] Pressure min/max     : {pressure.min():.2f} / {pressure.max():.2f} Pa")
    print(f"[INFO] Color scale 1–99 pct: {pmin:.2f} / {pmax:.2f} Pa")

    plotter = pv.Plotter(window_size=(1100, 900))
    plotter.set_background("white")

    plotter.add_text(
        "DIGITAL TWIN — predicted full-field CFD-like pressure",
        position="upper_left",
        font_size=13,
        color="black",
    )
    p99 = float(np.nanpercentile(pressure, 99))
    p999 = float(np.nanpercentile(pressure, 99.9))
    plotter.add_text(
        f"Points: {len(points):,}\n"

        f"Representative p99: {p99:.2f} Pa\n"
        f"High-pressure p99.9: {p999:.2f} Pa\n"
        f"Prediction time: milliseconds after training\n"
        f"CFD simulation replaced by POD + RBF",
        position=(20, 760),
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

    plotter.add_scalar_bar(
        title="Pressure (Pa)",
        n_labels=6,
        fmt="%.1f",
    )

    plotter.camera_position = "xz"
    plotter.camera.zoom(1.4)

    if args.screenshot:
        print(f"[INFO] Saving screenshot: {args.screenshot}")
        plotter.show(screenshot=args.screenshot)
    else:
        print("[INFO] Opening 3D full-field visualization...")
        print("[INFO] Mouse to rotate/zoom. Press Q to quit.")
        plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Human Airways Digital Twin — full-field CFD-like pressure prediction"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project base directory. Default: current directory.",
    )

    parser.add_argument(
        "--modes",
        type=int,
        default=10,
        help="Maximum number of pressure POD modes. Default: 10.",
    )

    parser.add_argument(
        "--energy",
        type=float,
        default=0.99,
        help="Target POD energy. Default: 0.99.",
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default="thin_plate_spline",
        help="RBF kernel. Default: thin_plate_spline.",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=1e-8,
        help="RBF smoothing. Default: 1e-8.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild POD cache even if it already exists.",
    )

    parser.add_argument(
        "--screenshot",
        type=str,
        default="digital_twin_fullfield_prediction.png",
        help="Screenshot output file. Default: digital_twin_fullfield_prediction.png",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Point size for visualization. Default: 1.5.",
    )

    parser.add_argument(
        "--clim",
        type=float,
        nargs=2,
        default=None,
        help="Manual color limits, example: --clim -20 100",
    )

    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable POD coefficient and pressure clipping stabilization.",
    )

    parser.add_argument(
        "--coeff-sigma-limit",
        type=float,
        default=3.0,
        help="Sigma limit for POD coefficient stabilization. Default: 3.0.",
    )

    args = parser.parse_args()

    t0 = time.time()

    base_dir = Path(args.base_dir).resolve()
    print(f"[INFO] Base directory: {base_dir}")

    # 1. Locate and load mesh
    points_path, snapshots = locate_dataset(base_dir)

    print("[INFO] Loading mesh points...")
    points = read_points_bin(points_path)
    n_points = points.shape[0]

    print(f"[INFO] Mesh points: {n_points:,}")

    # 2. Load DOE
    doe, doe_columns, X = load_doe(base_dir)

    if X.shape[0] != len(snapshots):
        print("[WARNING] Number of DOE rows and pressure snapshots differs.")
        print(f"          DOE rows: {X.shape[0]}")
        print(f"          snapshots: {len(snapshots)}")
        n = min(X.shape[0], len(snapshots))
        print(f"          Using first {n} samples.")
        X = X[:n]
        doe = doe.iloc[:n].copy()
        snapshots = snapshots[:n]

    # 3. Standardize DOE features
    Xs, feature_mean, feature_std = standardize_features(X)

    # 4. Load or build full-field pressure POD
    (
        mean_field,
        modes,
        coeffs,
        eigvals,
        cumulative_energy,
        k,
        pressure_stats,
        coeff_mean,
        coeff_std,
        coeff_min,
        coeff_max,
    ) = load_or_build_pod_model(
        base_dir,
        snapshots,
        n_points,
        args,
    )

    # 5. Train RBF surrogate
    rbf_model = train_rbf_surrogate(
        Xs,
        coeffs,
        kernel=args.kernel,
        smoothing=args.smoothing,
    )

    print()
    print("[✓] Full-field Digital Twin ready.")
    print(f"[INFO] Pressure POD modes: {k}")
    print(f"[INFO] Captured energy: {cumulative_energy[k - 1] * 100:.2f}%")
    print(f"[INFO] Setup time: {time.time() - t0:.1f} seconds")

    # 6. Interactive patient input
    patient_x = ask_patient_parameters(doe, doe_columns)

    # 7. Predict full field
    print()
    print("[INFO] Running full-field Digital Twin prediction...")
    tp = time.time()

    pressure, coeff_pred, coeff_raw, pressure_raw = reconstruct_pressure_field(
        patient_x,
        feature_mean,
        feature_std,
        rbf_model,
        mean_field,
        modes,
        pressure_stats=pressure_stats,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
        coeff_sigma_limit=args.coeff_sigma_limit,
        use_stabilization=not args.no_stabilize,
    )

    pred_time = time.time() - tp

    print("[✓] Prediction complete")
    print(f"[INFO] Prediction time: {pred_time * 1000:.2f} ms")
    print(f"[INFO] Pressure field size: {pressure.shape[0]:,} points")

    print("[INFO] Raw pressure before stabilization:")
    print(f"       min={pressure_raw.min():.2f} Pa, max={pressure_raw.max():.2f} Pa")

    if args.no_stabilize:
        print("[INFO] Stabilization disabled by --no-stabilize")
    else:
        print("[INFO] Stabilization enabled:")
        print("       POD coefficients clipped to training coefficient range")
        if pressure_stats is not None:
            print(
                f"       Pressure clipped to training robust range: "
                f"{pressure_stats['p001']:.2f} to {pressure_stats['p999']:.2f} Pa"
            )

    p001, p01, p05, p50, p95, p99, p999 = np.percentile(
        pressure, [0.1, 1, 5, 50, 95, 99, 99.9]
    )

    print(f"[INFO] Pressure range: {pressure.min():.2f} to {pressure.max():.2f} Pa")
    print("[INFO] Pressure percentiles:")
    print(
        f"       p0.1={p001:.2f}, p1={p01:.2f}, p5={p05:.2f}, "
        f"p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}, p99.9={p999:.2f}"
    )
    print(f"[INFO] Peak pressure after stabilization: {pressure.max():.2f} Pa")

    # 8. Save numerical result
    out_bin = base_dir / "digital_twin_fullfield_pressure.bin"
    pressure.astype(np.float32).tofile(out_bin)
    print(f"[INFO] Saved pressure field: {out_bin}")

    # 9. Visualize
    visualize_fullfield(points, pressure, args)


if __name__ == "__main__":
    main()