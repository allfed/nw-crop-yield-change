# 🌾❄️ NuclearWinterCropYieldAnalysis

Code and scripts for calculating percentage changes in crop and grass yields under different soot emission scenarios associated with varying intensities of nuclear winters.

This project processes crop and grass yield data from NetCDF datasets provided by [Xia et al. (2022)](https://www.nature.com/articles/s43016-022-00573-0), calculates percentage changes in yields for various crops and grasses across countries for the first 10 years post‑nuclear winter, and outputs the results to CSV files. It performs spatial data processing, aggregation, and analysis, and generates visualisations to evaluate the impact of different nuclear‑winter scenarios on global agriculture.

The code can be run locally using Python. Setup instructions are below.

> **Heads‑up**: You do **not** need to run the code to see the results. All model outputs are in the **`results/`** folder now (see the directory structure below).

---

## 1 · Setup

### 1.1 Dependencies (Poetry)

1. Install [Poetry](https://python-poetry.org/docs/).
2. From the repository root, install dependencies and activate the virtual‑env:

   ```bash
   poetry install
   poetry shell
   ```
3. Exit the environment with `exit`.

### 1.2 Input‑data layout

Raw data lives under **`data/raw/`** and is already included in the repo.

* **Crop & grass yields**: NetCDF files from [Xia et al. (2022)](https://osf.io/yrbse/)
* **Country borders**: Generalised ISO country shapefile from the ALLFED project, shipped under **`data/external/World_Countries__Generalized_ISO/`**

---

## 2 · Running the analysis

After activating the Poetry environment run:

```bash
python src/1_yield_change_calculation.py
```

The script will

1. load configuration from **`config/config.yaml`**,
2. process crop & grass yields for each soot‑emission scenario, and
3. write the results to CSV **in `results/`** (separate files for rain‑fed and irrigated outputs).

Logs are written to **`results/logs/`** while the script runs.

---

## 3 · Project structure

```text
.
├── config
│   └── config.yaml
├── data
│   ├── external
│   │   └── World_Countries__Generalized_ISO
│   │       ├── World_Countries__Generalized_ISO.{cpg,dbf,prj,shp,shx}
│   └── raw
│       ├── Crop Yield/                     # NetCDF crop‑yield files
│       ├── Grass Production/               # NetCDF grass‑yield files
│       └── rutgers_nw_production_raw.csv   # Reference dataset
│
├── docs
│   └── sources/README.md                   # Rendered copy of this file for Sphinx/Read‑the‑Docs
│
├── results
│   ├── logs/                               # *.log files produced during runs
│   └── output_<scenario>_*.csv             # Model outputs (one file per run / aggregate)
│
├── scripts                                 # Exploratory notebooks & helper images
│   ├── 1_Country shapefile debugging.ipynb
│   ├── 2_Reference file peculiarities.ipynb
│   ├── 3_Geospatial merging calculations.ipynb
│   └── *.png
│
├── src
│   └── 1_yield_change_calculation.py       # Main processing script
│
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

---

## 4 · Directory cheat‑sheet

| Folder | Key contents & purpose |
| ------ | ---------------------- |
| **`config/`** | `config.yaml` – centralises file paths, EPSG codes, crop mappings, country‑name mappings, etc. |
| **`data/raw/`** | Source NetCDF yield datasets and Rutgers reference CSV. |
| **`data/external/`** | Country shapefile used for spatial clipping. |
| **`results/`** | All CSV outputs from the script. Sub‑folder `logs/` captures run‑time logs. |
| **`scripts/`** | Jupyter notebooks & diagnostic plots – useful for deep‑dives but *not* needed for the main pipeline. |
| **`src/`** | `1_yield_change_calculation.py` – orchestrates the full pipeline. |

---

## 5 · Questions / support

* **Twitter**: [@thicknavyrain](https://twitter.com/thicknavyrain)
* **LinkedIn**: [Ricky Nathvani](https://www.linkedin.com/in/ricky-nathvani/)

---

## 6 · License

```text
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
```

This project is licensed under the Apache License 2.0.
