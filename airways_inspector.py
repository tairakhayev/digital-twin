# -*- coding: utf-8 -*-
"""
airways_inspector.py
====================
Interactive 3D inspector for Human Airways snapshots.

Inspired by hull_inspector.py (Prof. Marco E. Biancolini, Tor Vergata).
Adapted for the Human Airways Digital Twin dataset (DiTiDE / EuroHPC).

Controls
--------
  <- ->   navigate patients (snapshots)
  D       switch to Displacement view (mm)
  P       switch to Pressure view (Pa)
  Mouse   rotate / zoom
  Q       quit

Usage
-----
    python airways_inspector.py

Dependencies
------------
    pip install numpy pyvista pandas
"""

import numpy as np
import struct
import json
import pandas as pd
from pathlib import Path
import pyvista as pv

# ── CONFIG ────────────────────────────────────────────────────────────────────
POINTS_BIN    = "points.bin"
SETTINGS_JSON = "settings.json"
GEOM_DIR      = Path("snapshots")
PRES_DIR      = Path("Pressure/snapshots_pressure")
DOE_CSV       = "doe.csv"
RESULTS_CSV   = "results.csv"
PRESSURE_CSV  = "pressure_results.csv"

KEY_REGIONS = {
    "Glottis":       "glotis_max",
    "Larynx":        "larynx_max",
    "Trachea (bot)": "upper_trachea_bottom_max",
    "GL (left)":     "gl_max",
    "GR (right)":    "gr_max",
    "Epiglottis":    "epiglotis_max",
}
KEY_PRES = {
    "Glottis":       "glotis_mean_Pa",
    "Larynx":        "larynx_mean_Pa",
    "Trachea (bot)": "upper_trachea_bottom_mean_Pa",
    "GL (left)":     "gl_mean_Pa",
    "GR (right)":    "gr_mean_Pa",
    "Epiglottis":    "epiglotis_mean_Pa",
}

# ── I/O HELPERS ───────────────────────────────────────────────────────────────
def read_bin_vector(path):
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64).reshape(-1, 3)

def read_bin_scalar(path):
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("[INFO] Loading base mesh...")
pts = read_bin_vector(POINTS_BIN)

with open(SETTINGS_JSON) as f:
    ns = json.load(f)["namedSelections"]

geom_files = sorted(GEOM_DIR.glob("snapshot*.bin"),
                    key=lambda p: int(p.stem.replace("snapshot", "")))
pres_files = sorted(PRES_DIR.glob("snapshot*.bin"),
                    key=lambda p: int(p.stem.replace("snapshot", "")))

print(f"[INFO] Geometry snapshots : {len(geom_files)}")
print(f"[INFO] Pressure snapshots : {len(pres_files)}")
print("[INFO] Loading all snapshots (~30 sec)...")
geom_snaps = [read_bin_vector(p) for p in geom_files]
pres_snaps = [read_bin_scalar(p) for p in pres_files]
print("[INFO] Done!")

try:
    doe          = pd.read_csv(DOE_CSV)
    results      = pd.read_csv(RESULTS_CSV)
    results["num"] = results["snapshot"].str.extract(r"(\d+)").astype(int)
    results      = results.sort_values("num").reset_index(drop=True)
    pressure_res = pd.read_csv(PRESSURE_CSV)
    pressure_res["num"] = pressure_res["snapshot"].str.extract(r"(\d+)").astype(int)
    pressure_res = pressure_res.sort_values("num").reset_index(drop=True)
    has_metadata = True
    print("[INFO] Metadata loaded")
except Exception as e:
    has_metadata = False
    print(f"[WARN] Metadata not loaded: {e}")

# ── INFO TEXT ─────────────────────────────────────────────────────────────────
SEP = "================================"

def build_info(i):
    n = len(geom_snaps)
    is_disp = mode[0] == "displacement"

    lines = [
        SEP,
        f"  HUMAN AIRWAYS DIGITAL TWIN",
        f"  Patient  {i+1:3d}  /  {n}",
        SEP,
        f"  Mode: {'DISPLACEMENT  (FEA)' if is_disp else 'PRESSURE  (CFD)'}",
        SEP,
    ]

    if has_metadata and is_disp:
        row  = results.iloc[i]
        vals = {k: row[v] for k, v in KEY_REGIONS.items() if v in row}
        top  = max(vals, key=vals.get) if vals else "n/a"
        lines += [
            f"  Global max   :  {row['global_max']:5.1f} mm",
            f"  Global mean  :  {row['global_mean']:5.1f} mm",
            f"  Peak region  :  {top}",
            "",
            f"  Regional displacement (mm)",
            f"  ------------------------------",
        ]
        for region, col in KEY_REGIONS.items():
            if col in row:
                v   = row[col]
                bar = "|" * max(1, int(v / 2.0))
                lines.append(f"  {region:<14s}  {v:5.1f}  {bar}")
        doe_row = doe.iloc[i]
        lines += [
            "",
            SEP,
            f"  Patient anatomy (DOE)",
            f"  ------------------------------",
            f"  A_glotis     :  {doe_row.get('A_glotis',   0):.1f} mm2",
            f"  A_epiglotis  :  {doe_row.get('A_epiglotis',0):.1f} mm2",
            f"  l_trachea    :  {doe_row.get('l_trachea',  0):.1f} mm",
            f"  d_trachea    :  {doe_row.get('d_trachea',  0):.1f} mm",
        ]

    elif has_metadata and not is_disp:
        row = pressure_res.iloc[i]
        lines += [
            f"  Global max   :  {row['global_max_Pa']:.4f} Pa",
            f"  Global mean  :  {row['global_mean_Pa']:.4f} Pa",
            "",
            f"  Pressure drop (mouth -> trachea):",
            f"  {row.get('mouth_region_mean_Pa',0):.4f} Pa  ->  "
            f"{row.get('upper_trachea_bottom_mean_Pa',0):.4f} Pa",
            "",
            f"  Regional pressure (Pa)",
            f"  ------------------------------",
        ]
        for region, col in KEY_PRES.items():
            if col in row:
                lines.append(f"  {region:<14s}  {row[col]:8.4f}")

    lines += [
        "",
        SEP,
        f"  [D] Displacement   [P] Pressure",
        f"  [<-] prev          [->] next",
        f"  [Q]  quit",
        SEP,
    ]
    return "\n".join(lines)

# ── VIEWER ────────────────────────────────────────────────────────────────────
mode    = ["displacement"]
current = [0]

def get_points(i):
    return geom_snaps[i].astype(np.float32) if mode[0]=="displacement" else pts.astype(np.float32)

def get_scalars(i):
    if mode[0] == "displacement":
        mag = np.linalg.norm(geom_snaps[i] - pts, axis=1)
        return (mag * 1e3).astype(np.float32), "Displacement (mm)", [0, 25], "plasma"
    return pres_snaps[i].astype(np.float32), "Pressure (Pa)", [-150, 450], "RdBu_r"

def update(i):
    scalars, label, clim, cmap = get_scalars(i)
    cloud = pv.PolyData(get_points(i))
    cloud[label] = scalars
    pl.add_mesh(cloud, scalars=label, cmap=cmap, point_size=2,
                render_points_as_spheres=True, clim=clim,
                scalar_bar_args={
                    "title": label, "fmt": "%.2f",
                    "color": "black",
                    "title_font_size": 18,
                    "label_font_size": 15,
                    "position_x": 0.35,
                    "width": 0.30,
                },
                name="cloud")
    pl.remove_actor("info_text")
    pl.add_text(build_info(i),
                position="upper_left",
                font_size=14,
                color="black",
                font="courier",
                name="info_text")
    pl.render()

# Initial render
scalars0, label0, clim0, cmap0 = get_scalars(0)
cloud0 = pv.PolyData(get_points(0))
cloud0[label0] = scalars0

pl = pv.Plotter(title="Human Airways -- Displacement & Pressure Inspector",
                window_size=(1500, 950))
pl.background_color = "white"

pl.add_mesh(cloud0, scalars=label0, cmap=cmap0, point_size=2,
            render_points_as_spheres=True, clim=clim0,
            scalar_bar_args={
                "title": label0, "fmt": "%.2f",
                "color": "black",
                "title_font_size": 18,
                "label_font_size": 15,
                "position_x": 0.35,
                "width": 0.30,
            },
            name="cloud")

pl.add_text(build_info(0),
            position="upper_left",
            font_size=14,
            color="black",
            font="courier",
            name="info_text")

def next_snap():
    current[0] = min(current[0] + 1, len(geom_snaps) - 1)
    update(current[0])

def prev_snap():
    current[0] = max(current[0] - 1, 0)
    update(current[0])

def switch_displacement():
    mode[0] = "displacement"
    update(current[0])

def switch_pressure():
    mode[0] = "pressure"
    update(current[0])

pl.add_key_event("Right", next_snap)
pl.add_key_event("Left",  prev_snap)
pl.add_key_event("d",     switch_displacement)
pl.add_key_event("p",     switch_pressure)

print("\n3D window open!")
print("  <- ->   navigate patients")
print("  D       displacement view (FEA)")
print("  P       pressure view (CFD)")
print("  Mouse   rotate / zoom")
print("  Q       quit")

pl.show()