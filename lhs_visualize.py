# -*- coding: utf-8 -*-
"""
lhs_visualize.py
================
3D visualisation of Latin Hypercube Sampling in parameter space.

Inspired by LHS_PyVista.py (Prof. Marco E. Biancolini, Tor Vergata).
Adapted for the Human Airways Digital Twin dataset.

Shows 100 patient anatomies as a 3D point cloud in the space of
the three most influential DOE parameters:
  - A_glotis   : glottis cross-sectional area (mm2)
  - l_trachea  : trachea length (mm)
  - r_curvature: trachea curvature radius (mm)

Controls
--------
  R  -> regenerate a new LHS cloud (demo mode)
  Q  -> quit

Usage
-----
    python lhs_visualize.py

Dependencies
------------
    pip install numpy scipy pyvista pandas
"""

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.stats import qmc

# ── CONFIG ────────────────────────────────────────────────────────────────────
N        = 100
SEED     = 42
DOE_CSV  = "doe.csv"

PARAM_NAMES = ["A_glotis (mm2)", "l_trachea (mm)", "r_curvature (mm)"]
LOWS        = np.array([86.0,  80.0, 30.0])
HIGHS       = np.array([230.0, 150.0, 70.0])


# ── LHS GENERATOR ─────────────────────────────────────────────────────────────
def generate_lhs(n=N, seed=SEED):
    sampler = qmc.LatinHypercube(d=3, seed=seed, optimization="random-cd")
    U = sampler.random(n=n)
    return LOWS + (HIGHS - LOWS) * U


# ── LOAD ACTUAL DOE ────────────────────────────────────────────────────────────
def load_doe():
    try:
        doe = pd.read_csv(DOE_CSV)
        return doe[["A_glotis", "l_trachea", "r_curvature"]].values.astype(np.float32)
    except FileNotFoundError:
        print(f"[WARN] {DOE_CSV} not found -- showing synthetic LHS only")
        return None


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    doe_pts = load_doe()

    pl = pv.Plotter(window_size=(1100, 800),
                    title="Human Airways -- LHS Parameter Space")
    pl.set_background("white")

    # Bounding box
    box = pv.Box(bounds=(LOWS[0], HIGHS[0],
                          LOWS[1], HIGHS[1],
                          LOWS[2], HIGHS[2])).extract_all_edges()
    pl.add_mesh(box, line_width=2, color="gray")

    # Actual DOE points
    if doe_pts is not None:
        cloud_doe = pv.PolyData(doe_pts)
        cloud_doe["patient"] = np.arange(N)
        pl.add_mesh(cloud_doe,
                    scalars="patient",
                    cmap="plasma",
                    render_points_as_spheres=True,
                    point_size=14,
                    scalar_bar_args={"title": "Patient index"})

    pl.add_text(
        f"Human Airways -- 100 Patients (DOE)\n"
        f"X: {PARAM_NAMES[0]}  Y: {PARAM_NAMES[1]}  Z: {PARAM_NAMES[2]}\n"
        f"Press R to overlay synthetic LHS  |  Q to quit",
        position="upper_left", font_size=9, color="black"
    )

    synthetic_cloud = [None]

    def regenerate():
        if synthetic_cloud[0] is not None:
            pl.remove_actor(synthetic_cloud[0])
        seed = int(np.random.default_rng().integers(0, 9999))
        X    = generate_lhs(seed=seed).astype(np.float32)
        c    = pv.PolyData(X)
        c["idx"] = np.arange(N)
        synthetic_cloud[0] = pl.add_mesh(
            c, scalars="idx", cmap="viridis",
            render_points_as_spheres=True, point_size=10,
        )
        pl.render()
        print(f"[INFO] New LHS generated (seed={seed})")

    pl.add_key_event("r", regenerate)
    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    main()