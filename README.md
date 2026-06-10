# Human Airways Digital Twin

**Course:** Digital Twins Modeling and Applications  
**Professor:** Marco E. Biancolini — Università degli Studi di Roma Tor Vergata  
**Students:** Shugyla Assylbay & Tair Akhayev  
**Dataset:** DiTiDE / EuroHPC — RBF Morph / Ansys Twin Builder

---

## Project Overview

This project builds a **Digital Twin of human airways** for airflow simulation and drug delivery research. The dataset contains **100 patient-specific airway geometries**, each simulated with:

- **FEA** (Finite Element Analysis) → geometry deformation of airway walls
- **CFD** (Computational Fluid Dynamics) → static pressure field on airway walls

Each patient is defined by **26 anatomical parameters** (glottis area, trachea length, branch angles, etc.) stored in `doe.csv`, generated using **Latin Hypercube Sampling (LHS)** to ensure uniform coverage of the parameter space.

---

## Repository Structure

```
human_airways_project/
├── analysis.ipynb          # Main notebook — data loading, visualization, POD
├── airways_inspector.py    # Interactive 3D viewer (displacement & pressure)
├── pod_analysis.py         # POD/SVD analysis of snapshot matrices
├── lhs_demo.py             # Latin Hypercube Sampling demonstration
├── lhs_visualize.py        # 3D LHS visualization in PyVista
├── doe.csv                 # DOE parameters — 100 patients × 26 parameters
├── results.csv             # Geometry deformation results per region
├── pressure_results.csv    # CFD pressure results per region
├── correlation_results.csv # Pearson correlation: DOE parameters vs deformation
├── settings.json           # 37 named anatomical regions (index ranges)
├── points.bin              # Baseline mesh — 2,135,906 points × XYZ (float64)
├── snapshots/              # 100 geometry snapshots (one per patient)
└── Pressure/
    └── snapshots_pressure/ # 100 pressure snapshots (one per patient)
```

---

## Binary File Format

All `.bin` files share the same format:

```
[8 bytes]  int64  — number of values (count)
[count×8]  float64 array
```

- **Geometry snapshots**: `count = N_points × 3` → reshape to `(N, 3)` XYZ coordinates  
- **Pressure snapshots**: `count = N_points` → scalar pressure in Pa per point  
- **Displacement** = `snapshot - points.bin` (vector difference from baseline)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/11shugyla/human-airways-project.git
cd human-airways-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn pyvista scipy ipywidgets jupyter
```

---

## How to Run

### Main Notebook
```bash
jupyter notebook analysis.ipynb
```
Run all cells — loads data, plots displacement/pressure by region, runs POD analysis.

### Interactive 3D Viewer
```bash
python airways_inspector.py
```
Controls:
- `← →` — navigate between patients (snapshots)
- `D` — displacement colour map (FEA)
- `P` — pressure colour map (CFD)
- `Mouse` — rotate / zoom
- `Q` — quit

### POD Analysis
```bash
python pod_analysis.py
```
Runs SVD on geometry and pressure snapshot matrices, plots energy convergence curve.

### LHS Demonstration
```bash
python lhs_demo.py       # 2D comparison: LHS vs random sampling
python lhs_visualize.py  # 3D point cloud of DOE patients in parameter space
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Mesh points | 2,135,906 |
| Patients (snapshots) | 100 |
| Anatomical regions | 37 |
| DOE parameters | 26 |
| Max displacement | 29.2 mm (snapshot 76) |
| Mean displacement | ~10 mm |
| Pressure drop (mouth → trachea) | 0.047 Pa |
| POD modes for 99% geometry energy | 14 |
| POD modes for 99% pressure energy | 5 |

**Key finding:** Pressure in the GL/GR glottis branches correlates strongly with displacement (r = −0.87) — higher air pressure stiffens the walls and reduces deformation.

**POD finding:** Pressure is low-rank (5 modes) — dominated by one main airflow gradient. Geometry is higher-rank (14 modes) — reflects patient-specific anatomical variability.

---

## Dataset

Dataset provided by **RBF Morph** (Emanuele Di Meo) as part of the **DiTiDE EuroHPC** project. Simulations run with **Ansys Twin Builder** on the upper human airways (mouth → glottis → trachea → bronchial tree).
