# 🌾❄️ NuclearWinterCropYieldAnalysis

Code and scripts for calculating percentage changes in crop and grass yields under different soot emission scenarios associated with varying intensities of nuclear winters.

This project processes crop and grass yield data from NetCDF datasets provided by [Xia et al. (2022)](https://www.nature.com/articles/s43016-022-00573-0), calculates percentage changes in yields for various crops and grasses across countries for the first 10 years post-nuclear winter, and outputs the results to CSV files. It performs spatial data processing, aggregation, and analysis, and generates visualizations to evaluate the impact of different nuclear winter scenarios on global agriculture.

The code can be run locally using Python. Setup instructions are below.

**Note**: The code does not have to be run to view the results. All outputs of the code can be found in the reports and data/processed folders. The reports folder also contains detailed documentation on the results.

## Setup

### Dependencies: Setup in Local Development Environment

#### Dependency Management with Poetry

See [https://python-poetry.org/docs/](https://python-poetry.org/docs/) for installation instructions.

Once Poetry is installed, navigate to the root folder of the repository and run:

```bash
poetry install
```

This command will install all the required dependencies listed in the pyproject.toml file.

To activate the virtual environment for this project using Poetry, run:

```bash
poetry shell
```

To exit the virtual environment, simply type:

```bash
exit
```

### Input Data Management

The raw data is included in the repository under the `data/raw` directory.

- Crop yield and grass production data are sourced from [Xia et al. (2022)](https://osf.io/yrbse/).
- Country shapefile for spatial analysis is sourced from the ALLFED [LosingIndustryCropYields](https://github.com/allfed/LosingIndustryCropYields/) project and is located in `data/external/World_Countries__Generalized_`.

Ensure that the data files are in the correct directories as specified in the project structure below.

## How to Run the Code

After setting up the Poetry environment as described above, you can run the analysis scripts.

### Calculate Yield Changes:

To process the yield data and calculate the percentage changes for different nuclear winter scenarios, run:

```bash
python src/1_yield_change_calculation.py
```

This script processes the yield data for various crops and grasses under different soot emission scenarios and outputs CSV files containing the percentage changes in yields for each country and year. The outputs are saved in the `data/processed` directory.

### Model Evaluation:

To evaluate the model's correspondence with the reference data (provided in `data/raw/rutgers_nw_production_raw.csv`), run:

```bash
python src/2_model_evaluation.py
```

This script compares the calculated yield changes with the reference data, computes evaluation metrics, and generates plots saved in the `reports/figures` directory.

## Files

### Project Structure

```lua
.
├── config
│   └── config.yaml
├── data
│   ├── external
│   │   └── World_Countries__Generalized_
│   │       ├── World_Countries__Generalized_.cpg
│   │       ├── World_Countries__Generalized_.dbf
│   │       ├── World_Countries__Generalized_.prj
│   │       ├── World_Countries__Generalized_.shp
│   │       └── World_Countries__Generalized_.shx
│   ├── processed
│   │   ├── output_150Tg_crops_and_grasses_1.csv
│   │   ├── output_16Tg_crops_and_grasses_1.csv
│   │   ├── output_27Tg_crops_and_grasses_1.csv
│   │   ├── output_37Tg_crops_and_grasses_1.csv
│   │   ├── output_47Tg_crops_and_grasses_1.csv
│   │   ├── output_5Tg_crops_and_grasses_1.csv
│   │   ├── output_5Tg_crops_and_grasses_2.csv
│   │   ├── output_5Tg_crops_and_grasses_3.csv
│   │   └── output_5Tg_crops_and_grasses_aggregated.csv
│   └── raw
│       ├── Crop Yield
│       │   ├── clm5_crop_2deg_cpl_cntrl_03_0005-0019_yield_latlon_CLM5mask_6crops.nc
│       │   ├── [Other NetCDF files for crop yield data]
│       ├── Grass Production
│       │   ├── clm5_crop_2deg_cpl_cntrl_03_0005-0018_yield_latlon_CLM5mask_C3-C4-LEAFC.nc
│       │   ├── [Other NetCDF files for grass production data]
│       └── rutgers_nw_production_raw.csv
├── logs
│   ├── model_evaluation.log
│   └── yield_processing.log
├── notebooks
│   ├── 1_Country shapefile debugging.ipynb
│   ├── 2_Reference file peculiarities.ipynb
│   ├── 3_Geospatial merging calculations.ipynb
│   ├── cornrain_yield_points_overlay.png
│   ├── United Arab Emirates_Corn_yield_year1_comparison.png
│   └── United Arab Emirates_Corn_yield_year1.png
├── poetry.lock
├── pyproject.toml
├── README.md
├── reports
│   ├── figures
│   │   ├── country_mad_values.png
│   │   ├── country_r2_values.png
│   │   ├── country_r2_values_zoomed.png
│   │   ├── output_150Tg_crops_and_grasses_1_evaluation.png
│   │   └── [Other evaluation plots]
│   ├── fraction_of_countries_metrics.csv
│   └── model_evaluation_metrics.csv
├── src
│   ├── 1_yield_change_calculation.py
│   └── 2_model_evaluation.py
└── tests
```

### Which File Does What?

- **config**
  - `config.yaml`: Configuration file containing file paths, EPSG codes, crop aggregation mappings, country name mappings, and other parameters used in the scripts.

- **data**
  - **external**: Contains external data files, such as the country shapefile used for spatial analysis.
    - `World_Countries__Generalized_`: Directory containing the shapefile components for country boundaries.

  - **processed**: Contains the processed output files generated by the analysis scripts.
    - `output_<scenario>_crops_and_grasses_<index>.csv`: CSV files containing the percentage changes in yields for each country and year under different nuclear winter scenarios.

  - **raw**: Contains the raw input data required for the analysis.
    - `Crop Yield`: Directory containing NetCDF files with crop yield data under various nuclear winter scenarios.
    - `Grass Production`: Directory containing NetCDF files with grass production data under various nuclear winter scenarios.
    - `rutgers_nw_production_raw.csv`: Reference data file used for model evaluation.

- **logs**
  - `model_evaluation.log`: Log file capturing the output of the model evaluation script.
  - `yield_processing.log`: Log file capturing the output of the yield change calculation script.

- **notebooks**
  - `1_Country shapefile debugging.ipynb`: Notebook exploring the country shapefile data, addressing inconsistencies in country naming, and explaining the approach taken in the analysis code based on the raster and shapefile data.
  - `2_Reference file peculiarities.ipynb`: Notebook examining peculiarities in the reference data file (`rutgers_nw_production_raw.csv`) used for model evaluation, explaining the metrics used and the handling of spurious data.
  - `3_Geospatial merging calculations.ipynb`: Notebook describing how the raster data is clipped and processed in the analysis code, including the assumptions made.
  - **Additional PNG files**: Visual outputs and overlays generated during exploratory analysis.

- **reports**
  - **figures**: Contains plots and figures generated by the model evaluation script.
    - `country_mad_values.png`: Plot of Mean Absolute Difference (MAD) values for each country.
    - `country_r2_values.png`: Plot of R² values for each country.
    - `country_r2_values_zoomed.png`: Zoomed-in plot of R² values for better visualization.
    - `output_150Tg_crops_and_grasses_1_evaluation.png`: Evaluation plot for the 150 Tg scenario.
    - **[Other evaluation plots]**: Additional plots generated during model evaluation.

  - `fraction_of_countries_metrics.csv`: CSV file containing metrics on the fraction of countries meeting specified R² and MAD thresholds.
  - `model_evaluation_metrics.csv`: CSV file containing the calculated evaluation metrics for each country and crop.

- **src**
  - `1_yield_change_calculation.py`: Main script for processing yield data, calculating percentage changes, and outputting results to CSV files.
  - `2_model_evaluation.py`: Script for evaluating the model's outputs against the reference data, calculating metrics, and generating plots.

- **tests**
  - Directory reserved for test scripts (currently empty).

- **poetry.lock & pyproject.toml**
    - `poetry.lock`: File generated by Poetry, locking the specific versions of dependencies used in the project.
    - `pyproject.toml`: Configuration file for Poetry, listing the project's dependencies and other metadata.

- **Additional Information**
    - ***Contact***
    For questions or support, please contact:

    Twitter: @thicknavyrain
    LinkedIn: Ricky Nathvani

- **License**

arduino

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

This project is licensed under the Apache License 2.0.

