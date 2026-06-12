# Human Airways Digital Twin

**Course:** Digital Twins Modeling and Applications
**Professor:** Marco E. Biancolini — Università degli Studi di Roma Tor Vergata
**Students:** Shugyla Assylbay & Tair Akhayev
**Dataset:** DiTiDE / EuroHPC — RBF Morph / Ansys Twin Builder

---

## Project Overview

This project develops a **Reduced Order Model (ROM)-based Digital Twin prototype** for human airway simulations.

The original dataset contains 100 patient-specific airway cases. Each patient is described by:

* **26 anatomical DOE parameters** stored in `doe.csv`
* **FEA geometry deformation results**
* **CFD pressure field results**

The goal is to replace expensive full FEA/CFD simulations with a fast surrogate model that can predict regional airway pressure for a new patient in milliseconds.

The final Digital Twin pipeline is:

```text
New patient anatomical parameters
        ↓
RBF-0: DOE parameters → geometry deformation
        ↓
Geometry POD coefficients
        ↓
Regime-based RBF-1: geometry → pressure POD coefficients
        ↓
Predicted regional pressure field
        ↓
3D visualisation on airway mesh
```

---

## Main Contributions

The project includes:

1. Loading and inspecting airway FEA/CFD data
2. Interactive 3D visualisation of real snapshots
3. Regional displacement and pressure analysis
4. POD/SVD dimensionality reduction
5. K-Fold and Leave-One-Out validation
6. RBF surrogate modelling
7. Final regime-based Digital Twin model
8. 3D visualisation of predicted pressure with comparison to CFD reference data

---

## Repository Structure

```text
digital-twin/
├── README.md
├── .gitignore
│
├── analysis.ipynb
├── airways_inspector.py
├── pod_analysis.py
├── lhs_demo.py
├── lhs_visualize.py
│
├── analyze_pressure.py
├── kfold_validation.py
├── rbf_inference.py
├── digital_twin_final.py
├── digital_twin_3d.py
│
├── doe.csv
├── results.csv
├── pressure_results.csv
├── correlation_results.csv
│
├── pod_energy.png
├── pod_modes.png
├── kfold_validation.png
├── kfold_rbf_validation.png
├── rbf_inference.png
├── digital_twin_final_validation.png
├── digital_twin_final_demo.png
├── digital_twin_final_prediction.png
├── digital_twin_prediction.png
└── digital_twin_unseen_demo.png
```

---

## Important Note About Raw Data

The original raw binary folders are **not included** in this repository because of file size:

```text
Points/
Pressure/
```

These folders contain the full mesh and simulation snapshots:

```text
Points/
├── points.bin
├── snapshots/
├── doe.csv
├── settings.json
└── outputDefinition.json

Pressure/
├── points.bin
├── snapshots/
├── doe.csv
├── settings.json
└── outputDefinition.json
```

To run scripts that require the full mesh, such as:

```text
airways_inspector.py
digital_twin_3d.py
```

place the `Points/` and `Pressure/` folders in the project root.

The repository includes the processed regional CSV files:

```text
results.csv
pressure_results.csv
doe.csv
correlation_results.csv
```

These files are sufficient for POD analysis, validation, RBF inference, and the final regional Digital Twin model.

---

## Binary File Format

The original `.bin` files use the following format:

```text
[8 bytes]  int64    number of values
[count×8] float64  data values
```

For geometry:

```text
points.bin or geometry snapshot → reshape to (N, 3)
```

For pressure:

```text
pressure snapshot → one scalar pressure value per mesh point
```

The relation between files is:

```text
Patient 1 = row 1 in doe.csv + geometry snapshot_1 + pressure snapshot_1
Patient 2 = row 2 in doe.csv + geometry snapshot_2 + pressure snapshot_2
...
Patient 100 = row 100 in doe.csv + geometry snapshot_100 + pressure snapshot_100
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/tairakhayev/digital-twin.git
cd digital-twin
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn pyvista jupyter
```

Optional, if using notebooks:

```bash
pip install ipywidgets
```

---

## How to Run

### 1. Exploratory Analysis

```bash
jupyter notebook analysis.ipynb
```

This notebook performs data loading, regional displacement analysis, pressure analysis, correlation analysis, and POD exploration.

---

### 2. Interactive 3D Viewer for Original Snapshots

```bash
python airways_inspector.py
```

This script requires the original `Points/` and `Pressure/` folders.

Controls:

```text
← / →   navigate between patients
D       displacement mode
P       pressure mode
Mouse   rotate / zoom
Q       quit
```

---

### 3. POD/SVD Analysis

```bash
python pod_analysis.py
```

This script performs POD/SVD analysis and generates:

```text
pod_energy.png
pod_modes.png
```

---

### 4. POD Validation

```bash
python kfold_validation.py
```

This script performs:

* 5-Fold Cross-Validation for geometry POD
* 5-Fold Cross-Validation for pressure POD
* Leave-One-Out validation
* RBF full pipeline validation

Outputs:

```text
kfold_validation.png
kfold_rbf_validation.png
```

---

### 5. RBF Surrogate Inference

```bash
python rbf_inference.py
```

This script trains an RBF surrogate model that maps geometry POD coefficients to pressure POD coefficients.

Output:

```text
rbf_inference.png
```

---

### 6. Final Digital Twin Model

Validation mode:

```bash
python digital_twin_final.py --validate
```

Demo mode:

```bash
python digital_twin_final.py --demo
```

Interactive mode:

```bash
python digital_twin_final.py
```

In interactive mode, the user enters 26 anatomical DOE parameters. Pressing `ENTER` uses the population mean for each parameter.

The final model uses a regime-based surrogate strategy:

```text
LOW   pressure regime: mouth pressure < 30 Pa
MID   pressure regime: 30–100 Pa
HIGH  pressure regime: mouth pressure > 100 Pa
```

This improves the physical consistency of the surrogate model because airway pressure response is nonlinear and depends strongly on obstruction effects.

---

### 7. 3D Digital Twin Visualisation

```bash
python digital_twin_3d.py --compare --deform-scale 0.25
```

This script shows:

```text
1. Predicted Digital Twin pressure field
2. Nearest real CFD patient
3. Regional absolute error map
```

The visualisation maps predicted regional pressure values back onto the full airway point cloud using anatomical region definitions from `settings.json`.

Important limitation:

```text
The model predicts regional pressure values, not independent pressure values at all 2.1 million mesh points.
```

The 3D deformation is an approximate region-wise visualisation based on predicted regional displacement values.

---

## Key Results

| Result                           |            Value | Meaning                                       |
| -------------------------------- | ---------------: | --------------------------------------------- |
| Number of patients               |              100 | FEA/CFD simulation cases                      |
| DOE parameters                   |               26 | Anatomical input variables                    |
| Mesh size                        | 2,135,906 points | Full airway point cloud                       |
| Geometry POD energy              | 5 modes ≈ 99.46% | Geometry is low-dimensional at regional level |
| Pressure POD                     | 5 modes ≈ 99.95% | Pressure field is strongly low-rank           |
| POD LOO pressure error           |         0.029 Pa | Reliable POD reconstruction                   |
| Previous global RBF CV error     |             441% | Limited by nonlinear pressure response        |
| Final regime-based full pipeline |   median ≈ 30.2% | Improved Digital Twin generalisation          |
| DOE → geometry stage             |        r ≈ 0.917 | Strong geometry prediction stage              |
| In-sample Digital Twin demo      |  median ≈ 16.99% | Confirms pipeline implementation              |

---

## Scientific Interpretation

The POD analysis shows that the airway pressure field has a low-dimensional structure. This means that the pressure variation across 100 patients can be represented using only a few dominant pressure patterns.

The main physical pressure pattern is the pressure gradient from the mouth region toward the trachea and branches. Additional POD modes capture patient-specific effects such as glottis narrowing, epiglottis variation, and branch asymmetry.

The RBF surrogate is more difficult because the relationship between anatomy, geometry deformation, and pressure is nonlinear. The largest errors occur in anatomically complex regions such as:

```text
glottis
epiglottis
larynx
mouth region
```

These regions are sensitive to small geometric changes and can produce nonlinear pressure jumps.

---

## Main Conclusion

This repository presents a working **ROM-based Digital Twin prototype** for human airway simulations.

The project demonstrates that:

1. Human airway pressure fields can be represented using a small number of POD modes.
2. POD reconstruction is reliable for unseen patients.
3. RBF surrogate modelling can predict regional pressure fields quickly.
4. A regime-based approach improves the prediction compared with one global RBF model.
5. The final model can produce physically meaningful pressure predictions in milliseconds.

However, the model should be understood as a prototype, not a complete clinical Digital Twin. Further improvements would require:

* More CFD/FEA snapshots
* Full-mesh POD instead of regional statistics
* More advanced nonlinear surrogate models
* Real-time sensor or patient-specific clinical data integration

---

## Final Statement

The project successfully extends the original dataset analysis into a complete Digital Twin workflow:

```text
Data inspection
→ POD/SVD reduction
→ validation
→ RBF surrogate modelling
→ interactive prediction
→ 3D visualisation
```

It provides a scientifically honest and technically functional prototype for fast airway pressure prediction.
