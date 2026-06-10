import numpy as np
import pandas as pd
import struct
import json
from pathlib import Path

def read_bin_scalar(path):
    """Read a scalar .bin file — one pressure value per mesh point."""
    with open(path, "rb") as f:
        count = struct.unpack("<q", f.read(8))[0]
        return np.frombuffer(f.read(count * 8), dtype=np.float64)

with open("pressure/settings.json") as f:
    ns = json.load(f)["namedSelections"]

snap_files = sorted(
    Path("pressure/snapshots").glob("snapshot*.bin"),
    key=lambda p: int(p.stem.replace("snapshot", ""))
)
print(f"Pressure snapshots found: {len(snap_files)}")

key_regions = [
    "epiglotis", "glotis", "larynx", "mouth_region",
    "upper_trachea_bottom", "upper_trachea_middle", "upper_trachea_top",
    "gl", "gr", "glr", "grr"
]

results = []
for snap_path in snap_files:
    pressure = read_bin_scalar(snap_path)
    row = {"snapshot": snap_path.stem}
    for region in key_regions:
        start, _, end = ns[region]
        p = pressure[start:end+1]
        row[region + "_mean_Pa"] = float(p.mean())
        row[region + "_max_Pa"]  = float(p.max())
        row[region + "_min_Pa"]  = float(p.min())
    row["global_mean_Pa"] = float(pressure.mean())
    row["global_max_Pa"]  = float(pressure.max())
    row["global_min_Pa"]  = float(pressure.min())
    results.append(row)
    print(f"  {snap_path.stem}: mean={row['global_mean_Pa']:.1f} Pa  "
          f"max={row['global_max_Pa']:.1f} Pa  min={row['global_min_Pa']:.1f} Pa")

df = pd.DataFrame(results)
df.to_csv("pressure_results.csv", index=False)
print("\nDone → pressure_results.csv")