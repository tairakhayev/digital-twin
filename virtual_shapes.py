#!/usr/bin/env python3
"""
Generate 1000 virtual airway shapes / virtual patients from DOE space.

Purpose
-------
This script closes the "generation of 1000 virtual shapes" requirement.

It does NOT run CFD or FEA. Instead, it uses the original doe.csv design space
and generates 1000 new virtual patient geometries as DOE parameter combinations
inside the training ranges.

Outputs
-------
1. virtual_shapes_1000_doe.csv
   - 1000 synthetic DOE rows with the same 26 anatomical parameters.

2. virtual_shapes_1000_summary.csv
   - same samples plus resistance proxy and qualitative LOW/MID/HIGH labels.

3. virtual_shapes_representative_cases.csv
   - one WIDE, one MEAN-like, and one NARROW case for visualization.

4. virtual_shapes_proxy_distribution.png
   - histogram of resistance proxy.

5. virtual_shapes_glottis_epiglottis.png
   - scatter plot of A_glotis vs A_epiglotis colored by proxy.

Recommended usage
-----------------
Run from the project root:

    python generate_1000_virtual_shapes.py

Optional:

    python generate_1000_virtual_shapes.py --n 1000 --seed 42

After generation, representative cases can be manually tested in:
    digital_twin_final.py
    digital_twin_hybrid.py

Interpretation
--------------
The 1000 virtual shapes are new virtual airway geometries in the DOE/anatomical
parameter space. They are used for population-level exploration of the trained
Digital Twin without creating new CFD/FEA simulations.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


EXCLUDE_COLUMNS = {
    "points",
    "pressure",
    "snapshot",
    "file",
    "filename",
    "inlet_vel",
    "part_size",
    "part_inj_vel",
}


def load_doe(base_dir: Path):
    candidates = [
        base_dir / "doe.csv",
        base_dir / "Points" / "doe.csv",
        base_dir / "points" / "doe.csv",
        base_dir / "Pressure" / "doe.csv",
        base_dir / "pressure" / "doe.csv",
    ]

    doe_path = None
    for p in candidates:
        if p.exists():
            doe_path = p
            break

    if doe_path is None:
        raise FileNotFoundError("Could not find doe.csv in project root or dataset folders.")

    print(f"[INFO] Loading DOE: {doe_path}")

    # sep=None detects comma or semicolon separated files
    doe = pd.read_csv(doe_path, sep=None, engine="python")

    numeric_cols = []
    for col in doe.columns:
        col_lower = str(col).strip().lower()
        if col_lower in EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(doe[col]):
            numeric_cols.append(col)

    if len(numeric_cols) != 26:
        print(f"[WARNING] Expected 26 anatomical parameters, found {len(numeric_cols)}.")
        print("[WARNING] Columns used:")
        for c in numeric_cols:
            print(f"          - {c}")

    return doe, numeric_cols


def latin_hypercube(n_samples: int, n_dim: int, rng: np.random.Generator):
    """
    Simple Latin Hypercube Sampling implementation without external dependencies.

    Returns values in [0, 1] with shape (n_samples, n_dim).
    """
    result = np.empty((n_samples, n_dim), dtype=np.float64)

    for j in range(n_dim):
        # one random point in each interval
        cut = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(cut)
        result[:, j] = cut

    return result


def generate_virtual_doe(doe: pd.DataFrame, columns, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)

    mins = doe[columns].min().astype(float).values
    maxs = doe[columns].max().astype(float).values

    lhs = latin_hypercube(n_samples, len(columns), rng)
    samples = mins[None, :] + lhs * (maxs - mins)[None, :]

    virtual = pd.DataFrame(samples, columns=columns)

    # Add virtual snapshot names for traceability
    virtual.insert(0, "virtual_id", [f"virtual_{i+1:04d}" for i in range(n_samples)])

    return virtual


def find_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in columns:
        cl = str(c).lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None


def add_physics_proxy(virtual: pd.DataFrame, original_doe: pd.DataFrame, columns):
    """
    Add a simple physically meaningful resistance proxy:
        proxy = mean( (A_mean / A_patient)^2 for glottis and epiglottis )

    Larger proxy means narrower airway and higher expected pressure.
    """
    g_col = find_col(columns, ["A_glotis", "glotis"])
    e_col = find_col(columns, ["A_epiglotis", "epiglotis"])

    parts = []
    used = []

    if g_col is not None:
        A_train = np.maximum(original_doe[g_col].astype(float).values, 1e-8)
        A_mean = float(np.mean(A_train))
        A_virtual = np.maximum(virtual[g_col].astype(float).values, 1e-8)
        parts.append((A_mean / A_virtual) ** 2)
        used.append(g_col)

    if e_col is not None:
        A_train = np.maximum(original_doe[e_col].astype(float).values, 1e-8)
        A_mean = float(np.mean(A_train))
        A_virtual = np.maximum(virtual[e_col].astype(float).values, 1e-8)
        parts.append((A_mean / A_virtual) ** 2)
        used.append(e_col)

    if parts:
        proxy = np.mean(np.vstack(parts), axis=0)
    else:
        proxy = np.ones(len(virtual), dtype=np.float64)

    virtual["resistance_proxy"] = proxy

    # Qualitative labels using quantiles of virtual population
    q_low = np.quantile(proxy, 1 / 3)
    q_high = np.quantile(proxy, 2 / 3)

    labels = np.where(
        proxy <= q_low,
        "LOW_RESISTANCE",
        np.where(proxy <= q_high, "MID_RESISTANCE", "HIGH_RESISTANCE")
    )
    virtual["virtual_regime"] = labels

    print(f"[INFO] Resistance proxy based on: {', '.join(used) if used else 'not available'}")
    print(f"[INFO] Proxy ranges: min={proxy.min():.3f}, median={np.median(proxy):.3f}, max={proxy.max():.3f}")

    return virtual


def select_representative_cases(summary: pd.DataFrame):
    """
    Pick three representative virtual cases:
        WIDE   = lowest resistance proxy
        MEAN   = closest to median proxy
        NARROW = highest resistance proxy
    """
    proxy = summary["resistance_proxy"].values
    median_proxy = np.median(proxy)

    idx_wide = int(np.argmin(proxy))
    idx_narrow = int(np.argmax(proxy))
    idx_mean = int(np.argmin(np.abs(proxy - median_proxy)))

    reps = summary.iloc[[idx_wide, idx_mean, idx_narrow]].copy()
    reps.insert(1, "case_type", ["WIDE_LOW_PRESSURE", "MEAN_LIKE", "NARROW_HIGH_PRESSURE"])

    return reps


def make_plots(summary: pd.DataFrame, columns, out_dir: Path):
    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(summary["resistance_proxy"], bins=35)
    plt.xlabel("Resistance proxy")
    plt.ylabel("Number of virtual shapes")
    plt.title("Distribution of 1000 virtual airway shapes")
    plt.tight_layout()
    hist_path = out_dir / "virtual_shapes_proxy_distribution.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # Scatter A_glotis vs A_epiglotis
    g_col = find_col(columns, ["A_glotis", "glotis"])
    e_col = find_col(columns, ["A_epiglotis", "epiglotis"])

    if g_col is not None and e_col is not None:
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(
            summary[g_col],
            summary[e_col],
            c=summary["resistance_proxy"],
            s=18,
        )
        plt.xlabel(g_col)
        plt.ylabel(e_col)
        plt.title("Virtual shapes in glottis/epiglottis design space")
        cbar = plt.colorbar(sc)
        cbar.set_label("Resistance proxy")
        plt.tight_layout()
        scatter_path = out_dir / "virtual_shapes_glottis_epiglottis.png"
        plt.savefig(scatter_path, dpi=200)
        plt.close()
    else:
        scatter_path = None

    print(f"[INFO] Saved plot: {hist_path}")
    if scatter_path:
        print(f"[INFO] Saved plot: {scatter_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 1000 virtual airway shapes from DOE design space.")
    parser.add_argument("--base-dir", type=str, default=".", help="Project base directory.")
    parser.add_argument("--n", type=int, default=1000, help="Number of virtual shapes to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GENERATION OF 1000 VIRTUAL AIRWAY SHAPES")
    print("Method: Latin Hypercube Sampling inside original DOE parameter ranges")
    print("=" * 72)

    doe, columns = load_doe(base_dir)

    print(f"[INFO] Original patients: {len(doe)}")
    print(f"[INFO] DOE parameters used: {len(columns)}")
    print(f"[INFO] Generating virtual shapes: {args.n}")

    virtual = generate_virtual_doe(doe, columns, args.n, args.seed)
    summary = add_physics_proxy(virtual.copy(), doe, columns)

    reps = select_representative_cases(summary)

    virtual_path = out_dir / "virtual_shapes_1000_doe.csv"
    summary_path = out_dir / "virtual_shapes_1000_summary.csv"
    reps_path = out_dir / "virtual_shapes_representative_cases.csv"

    virtual.to_csv(virtual_path, index=False)
    summary.to_csv(summary_path, index=False)
    reps.to_csv(reps_path, index=False)

    make_plots(summary, columns, out_dir)

    print()
    print("[✓] Virtual shape generation complete")
    print(f"[INFO] Saved: {virtual_path}")
    print(f"[INFO] Saved: {summary_path}")
    print(f"[INFO] Saved: {reps_path}")

    print()
    print("[INFO] Representative cases for manual Digital Twin visualization:")
    display_cols = ["virtual_id", "case_type", "resistance_proxy", "virtual_regime"]
    for c in ["A_glotis", "A_epiglotis"]:
        found = find_col(columns, [c, c.replace("A_", "")])
        if found is not None:
            display_cols.append(found)

    print(reps[display_cols].to_string(index=False))

    print()
    print("[NEXT STEP]")
    print("Use rows from virtual_shapes_representative_cases.csv as input for:")
    print("  python digital_twin_final.py")
    print("  python digital_twin_hybrid.py --clim -20 180 --physics-blend 1.0")
    print()
    print("Recommended presentation statement:")
    print("  1000 virtual airway geometries were generated using Latin Hypercube")
    print("  Sampling inside the original DOE design space. The trained Digital Twin")
    print("  can then evaluate these virtual cases instantly without new CFD/FEA runs.")


if __name__ == "__main__":
    main()
