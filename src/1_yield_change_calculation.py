"""
CROP AND GRASS YIELD PERCENTAGE CHANGE CALCULATION SCRIPT

This code processes crop and grass yield data from NetCDF datasets, calculates percentage changes in yields
for various crops and grasses across countries for the first 10 years post‑nuclear winter, and outputs the 
results to CSV files:

* **Rain‑fed components** (per crop + summed grasses) → `output_<scenario>_crops_and_grasses_rainfed_*.csv`
* **Irrigated components** (per crop) → `output_<scenario>_crops_irrigated_*.csv`
* **TOTAL (rain‑fed + irrigated) crop yield** (with grasses included once) → `output_<scenario>_crops_and_grasses_*.csv`

Aggregate files are also produced when a scenario consists of multiple realisations.

It performs the following steps:

1. Align longitudes in the datasets from [0, 360] to [‑180, 180] for spatial merging.
2. Ensures consistent coordinate reference systems (CRS) across datasets.
3. Assigns CRS and reprojects yield data to the appropriate projection before aggregation.
4. Processes rain‑fed, irrigated **and total (rain‑fed + irrigated)** yields for specified crops.
5. Aggregates yields for grasses (C3grass and C4grass) when processing rain‑fed and total components.
6. Clips yield data to country geometries provided in a shapefile.
7. Calculates total yield per country and percentage changes compared to averaged control datasets.
8. Applies country name mappings for consistency.
9. Outputs the results to separate CSV files, sorted alphabetically by country name.

AUTHOR: Ricky Nathvani
DATE: 2025‑05‑04
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import logging
import yaml


def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("results/logs/yield_processing.log"),
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    """
    Loads the configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# --------------------------------------------------------------------------------------
# Helper functions (longitude fix, CRS assignment, clipping, maths)
# --------------------------------------------------------------------------------------

def fix_longitude(ds):
    """
    Correct longitude from [0, 360] to [‑180, 180] and sort the dataset.
    """
    ds = ds.assign_coords(lon=(((ds['lon'] + 180) % 360) - 180)).sortby('lon')
    return ds


def assign_crs(data_array, epsg_code):
    """
    Assign a coordinate reference system (CRS) to a DataArray after validating latitude values.
    """
    if 'lat' not in data_array.coords:
        logging.error("DataArray does not contain 'lat' coordinate.")
        raise KeyError("Missing 'lat' coordinate.")

    lat = data_array['lat']
    min_lat = lat.min().item()
    max_lat = lat.max().item()
    logging.info(f"Latitude range before reprojection: min={min_lat}, max={max_lat}")

    if min_lat <= -90 or max_lat >= 90:
        logging.warning(
            f"Invalid latitude values detected: min={min_lat}, max={max_lat}. Clipping to valid range.")
        data_array = data_array.where((lat > -90) & (lat < 90), drop=True)
        lat = data_array['lat']
        min_lat = lat.min().item()
        max_lat = lat.max().item()
        logging.info(f"Latitude range after clipping: min={min_lat}, max={max_lat}")
        if min_lat <= -90 or max_lat >= 90:
            logging.error("Clipping did not resolve invalid latitude values.")
            raise ValueError("Data contains invalid latitude values even after clipping.")

    if 'lat' not in data_array.dims or 'lon' not in data_array.dims:
        logging.error("'lat' and/or 'lon' are not dimensions of the DataArray.")
        raise ValueError("'lat' and/or 'lon' are not dimensions of the DataArray.")

    try:
        data_array = (
            data_array.rio.write_crs("EPSG:4326")
            .rio.set_spatial_dims(x_dim='lon', y_dim='lat')
            .rio.reproject(f"EPSG:{epsg_code}")
        )
    except Exception as e:
        logging.error(f"Reprojection failed: {e}")
        raise
    return data_array


def get_component_yield(ds, component_idx, epsg_code, year):
    """Return yield for a single component for the chosen year (after CRS)."""
    component_yield = ds['yield'].isel(crops=component_idx, time=year).fillna(0).astype(np.float32)
    return assign_crs(component_yield, epsg_code)


def aggregate_grass_yields(ds, epsg_code, year):
    """Sum C3 and C4 grass yields for the selected year (after CRS)."""
    grass_yields = []
    for grass_idx in range(len(ds['grass'])):
        grass_yield = ds['yield'].isel(grass=grass_idx, time=year).fillna(0).astype(np.float32)
        grass_yields.append(assign_crs(grass_yield, epsg_code))
    return sum(grass_yields)


def clip_yield_to_country(yield_data, country_geometry, epsg_code):
    """Clip DataArray to a single country's geometry."""
    return yield_data.rio.clip(country_geometry, crs=f"EPSG:{epsg_code}", drop=False, all_touched=True)


def calculate_percentage_change(yield_value, reference_yield_value):
    """Compute percentage change, guarding against divide‑by‑zero."""
    return 100 * (yield_value - reference_yield_value) / reference_yield_value if reference_yield_value > 0 else np.nan


# --------------------------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------------------------

def process_yields(
    ds,
    control_datasets,
    gdf,
    aggregation_info,
    names_list,
    years,
    epsg_code,
    country_mapping,
    component_type,
    include_grasses=False,
    ds_grass=None,
    grass_control_datasets=None,
):
    """
    Compute % yield change for either rain‑fed, irrigated **or total** components.
    """
    percentage_changes = {}

    for group_name, sub_components in tqdm(aggregation_info.items(), desc=f'Processing crops ({component_type})'):
        for year in range(years):
            # Determine component indices and labels
            if component_type == 'rainfed':
                component_indices = [names_list.index(sub_components[0])]
                component_suffix = 'rainfed'
            elif component_type == 'irrigated':
                component_indices = [names_list.index(sub_components[1])]
                component_suffix = 'irrigated'
            elif component_type == 'total':
                component_indices = [names_list.index(c) for c in sub_components]
                component_suffix = ''
            else:
                logging.error(f"Invalid component_type: {component_type}")
                continue

            # Sum selected components for target dataset
            yield_components = [get_component_yield(ds, idx, epsg_code, year) for idx in component_indices]
            yield_year = sum(yield_components)

            # Build equivalent sum for each control dataset
            reference_yield_years = []
            for ctrl_ds in control_datasets:
                ctrl_components = [get_component_yield(ctrl_ds, idx, epsg_code, year) for idx in component_indices]
                reference_yield_years.append(sum(ctrl_components))
            reference_yield_year_avg = xr.concat(xr.align(*reference_yield_years, join='exact'), dim='dataset').mean(dim='dataset')

            # Per‑country calculations
            for _, country in gdf.iterrows():
                cname = country_mapping.get(country['COUNTRY'], country['COUNTRY'])
                iso = country['ISO']
                geom = [country['geometry']]
                try:
                    yld = clip_yield_to_country(yield_year, geom, epsg_code).sum(skipna=True).item()
                    ref = clip_yield_to_country(reference_yield_year_avg, geom, epsg_code).sum(skipna=True).item()
                    pct = calculate_percentage_change(yld, ref)

                    if cname not in percentage_changes:
                        percentage_changes[cname] = {'ISO3 Country Code': iso, 'Country': cname}

                    col_name = 'spring_wheat' if group_name == 'Wheat' else group_name.lower()
                    if component_type == 'total':
                        percentage_changes[cname][f'{col_name}_year{year + 1}'] = pct
                    else:
                        percentage_changes[cname][f'{col_name}_{component_suffix}_year{year + 1}'] = pct
                except Exception as e:
                    logging.error(f"Could not process {cname} (year {year + 1}, group {group_name}, {component_type}): {e}")

    # Grasses (only once, when rain‑fed or total)
    if include_grasses and component_type in {'rainfed', 'total'}:
        for year in range(years):
            yield_year = aggregate_grass_yields(ds_grass, epsg_code, year)
            reference_yield_years = [aggregate_grass_yields(ctrl, epsg_code, year) for ctrl in grass_control_datasets]
            reference_yield_year_avg = xr.concat(xr.align(*reference_yield_years, join='exact'), dim='dataset').mean(dim='dataset')
            for _, country in gdf.iterrows():
                cname = country_mapping.get(country['COUNTRY'], country['COUNTRY'])
                iso = country['ISO']
                geom = [country['geometry']]
                try:
                    yld = clip_yield_to_country(yield_year, geom, epsg_code).sum(skipna=True).item()
                    ref = clip_yield_to_country(reference_yield_year_avg, geom, epsg_code).sum(skipna=True).item()
                    pct = calculate_percentage_change(yld, ref)
                    if cname not in percentage_changes:
                        percentage_changes[cname] = {'ISO3 Country Code': iso, 'Country': cname}
                    percentage_changes[cname][f'grasses_year{year + 1}'] = pct
                except Exception as e:
                    logging.error(f"Could not process {cname} (year {year + 1}, grasses): {e}")

    result_df = pd.DataFrame.from_dict(percentage_changes, orient='index').reset_index(drop=True)
    return result_df.sort_values(by='Country').reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Scenario driver
# --------------------------------------------------------------------------------------

def process_scenario(
    scenario_name,
    crop_file_paths,
    grass_file_paths,
    control_crop_files,
    control_grass_files,
    gdf,
    crop_aggregation,
    country_mapping,
    EPSG=4326,
    years=10,
):
    """Process all realisations for a single soot‑emission scenario."""
    control_datasets = [fix_longitude(xr.open_dataset(fp)) for fp in control_crop_files]
    grass_control_datasets = [fix_longitude(xr.open_dataset(fp)) for fp in control_grass_files]

    combined_results = {key: [] for key in ('rainfed', 'irrigated', 'total')}

    for idx, (crop_fp, grass_fp) in enumerate(zip(crop_file_paths, grass_file_paths)):
        ds_crop = fix_longitude(xr.open_dataset(crop_fp))
        ds_grass = fix_longitude(xr.open_dataset(grass_fp))
        crop_names = ds_crop['crops'].attrs['long_name'].split(',')

        # ----- Rain‑fed -----
        rainfed_df = process_yields(
            ds=ds_crop,
            control_datasets=control_datasets,
            gdf=gdf,
            aggregation_info=crop_aggregation,
            names_list=crop_names,
            years=years,
            epsg_code=EPSG,
            country_mapping=country_mapping,
            component_type='rainfed',
            include_grasses=True,
            ds_grass=ds_grass,
            grass_control_datasets=grass_control_datasets,
        )
        rf_name = f'results/output_{scenario_name}_crops_and_grasses_rainfed_{idx + 1}.csv'
        rainfed_df.to_csv(rf_name, index=False)
        logging.info(f"Saved rain‑fed results to {rf_name}")
        combined_results['rainfed'].append(rainfed_df)

        # ----- Irrigated -----
        irrigated_df = process_yields(
            ds=ds_crop,
            control_datasets=control_datasets,
            gdf=gdf,
            aggregation_info=crop_aggregation,
            names_list=crop_names,
            years=years,
            epsg_code=EPSG,
            country_mapping=country_mapping,
            component_type='irrigated',
        )
        ir_name = f'results/output_{scenario_name}_crops_irrigated_{idx + 1}.csv'
        irrigated_df.to_csv(ir_name, index=False)
        logging.info(f"Saved irrigated results to {ir_name}")
        combined_results['irrigated'].append(irrigated_df)

        # ----- TOTAL (rain‑fed + irrigated) -----
        total_df = process_yields(
            ds=ds_crop,
            control_datasets=control_datasets,
            gdf=gdf,
            aggregation_info=crop_aggregation,
            names_list=crop_names,
            years=years,
            epsg_code=EPSG,
            country_mapping=country_mapping,
            component_type='total',
            include_grasses=True,
            ds_grass=ds_grass,
            grass_control_datasets=grass_control_datasets,
        )
        tot_name = f'results/output_{scenario_name}_crops_and_grasses_{idx + 1}.csv'
        total_df.to_csv(tot_name, index=False)
        logging.info(f"Saved total results to {tot_name}")
        combined_results['total'].append(total_df)

    # Aggregate across realisations (if >1)
    for key, dfs in combined_results.items():
        if len(dfs) > 1:
            avg_df = pd.concat(dfs).groupby(['ISO3 Country Code', 'Country']).mean().reset_index()
            agg_name = 'crops_irrigated' if key == 'irrigated' else 'crops_and_grasses'
            out_file = f'results/output_{scenario_name}_{agg_name}_{"_" if key != "total" else ""}{key if key != "total" else ""}aggregated.csv'
            # Clean up double underscores if they appear
            out_file = out_file.replace("__", "_")
            avg_df.to_csv(out_file, index=False)
            logging.info(f"Saved aggregated {key} results to {out_file}")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------

def main():
    setup_logging()
    config = load_config('config/config.yaml')
    control_crop_files = config['control_crop_files']
    control_grass_files = config['control_grass_files']
    scenarios = config['scenarios']
    gdf = gpd.read_file(config['shapefile_path']).to_crs(epsg=config['EPSG'])
    crop_aggregation = config['crop_aggregation']
    country_mapping = config['country_mapping']

    for scenario in scenarios:
        logging.info(f"Processing scenario: {scenario['name']}")
        process_scenario(
            scenario_name=scenario['name'],
            crop_file_paths=scenario['crop_files'],
            grass_file_paths=scenario['grass_files'],
            control_crop_files=control_crop_files,
            control_grass_files=control_grass_files,
            gdf=gdf,
            crop_aggregation=crop_aggregation,
            country_mapping=country_mapping,
            EPSG=config['EPSG'],
            years=config['years'],
        )


if __name__ == '__main__':
    main()
