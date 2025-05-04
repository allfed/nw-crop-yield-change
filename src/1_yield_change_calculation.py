"""
CROP AND GRASS YIELD PERCENTAGE CHANGE CALCULATION SCRIPT

This code processes crop and grass yield data from NetCDF datasets, calculates percentage changes in yields
for various crops and grasses across countries for the first 10 years post-nuclear winter, and outputs the 
results to CSV files separately for rainfed and irrigated components.
It performs the following steps:

1. Aligns longitudes in the datasets from [0, 360] to [-180, 180] for spatial merging.
2. Ensures consistent coordinate reference systems (CRS) across datasets.
3. Assigns CRS and reprojects yield data to the appropriate projection before aggregation.
4. Processes rainfed and irrigated yields separately for specified crops.
5. Aggregates yields for grasses (C3grass and C4grass) when processing rainfed components.
6. Clips yield data to country geometries provided in a shapefile.
7. Calculates total yield per country and percentage changes compared to averaged control datasets.
8. Applies country name mappings for consistency.
9. Outputs the results to separate CSV files for rainfed and irrigated components, sorted alphabetically by country name.

AUTHOR: Ricky Nathvani
DATE: 2024-11-30
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


def fix_longitude(ds):
    """
    Correct longitude from [0, 360] to [-180, 180] and sort the dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset with longitude coordinates to be corrected.

    Returns:
    - xarray.Dataset: The dataset with corrected and sorted longitude coordinates.
    """
    ds = ds.assign_coords(lon=(((ds['lon'] + 180) % 360) - 180)).sortby('lon')
    return ds


def assign_crs(data_array, epsg_code):
    """
    Assign a coordinate reference system (CRS) to a DataArray after validating latitude values.

    Parameters:
    - data_array (xarray.DataArray): The DataArray to assign CRS to.
    - epsg_code (int): The EPSG code for the coordinate reference system.

    Returns:
    - xarray.DataArray: The DataArray with assigned CRS.
    """
    # Check if 'lat' coordinate exists
    if 'lat' not in data_array.coords:
        logging.error("DataArray does not contain 'lat' coordinate.")
        raise KeyError("Missing 'lat' coordinate.")

    # Extract latitude values
    lat = data_array['lat']
    min_lat = lat.min().item()
    max_lat = lat.max().item()
    logging.info(f"Latitude range before reprojection: min={min_lat}, max={max_lat}")

    # Validate latitude values, including exact poles
    if min_lat <= -90 or max_lat >= 90:
        logging.warning(f"Invalid latitude values detected: min={min_lat}, max={max_lat}. Clipping to valid range.")
        # Clip latitude values to the valid range, excluding exact poles
        data_array = data_array.where((lat > -90) & (lat < 90), drop=True)
        lat = data_array['lat']
        min_lat = lat.min().item()
        max_lat = lat.max().item()
        logging.info(f"Latitude range after clipping: min={min_lat}, max={max_lat}")

        # Re-check latitude after clipping
        if min_lat <= -90 or max_lat >= 90:
            logging.error("Clipping did not resolve invalid latitude values.")
            raise ValueError("Data contains invalid latitude values even after clipping.")

    # Verify spatial dimensions
    dims = data_array.dims
    coords = list(data_array.coords)
    logging.info(f"DataArray dimensions: {dims}")
    logging.info(f"DataArray coordinates: {coords}")

    # Ensure 'lat' and 'lon' are dimensions
    if 'lat' not in dims or 'lon' not in dims:
        logging.error("'lat' and/or 'lon' are not dimensions of the DataArray.")
        raise ValueError("'lat' and/or 'lon' are not dimensions of the DataArray.")

    # Proceed with CRS assignment
    try:
        # Log existing CRS if available
        existing_crs = data_array.rio.crs
        if existing_crs:
            logging.info(f"Existing CRS: {existing_crs}")
        else:
            logging.info("No existing CRS found. Assigning EPSG:4326 as the source CRS.")

        data_array = (
            data_array.rio.write_crs("EPSG:4326")  # Assuming original CRS is WGS84
            .rio.set_spatial_dims(x_dim='lon', y_dim='lat')
            .rio.reproject(f"EPSG:{epsg_code}")
        )
        logging.info(f"Reprojection to EPSG:{epsg_code} successful.")
    except Exception as e:
        logging.error(f"Reprojection failed: {e}")
        raise e

    return data_array


def get_component_yield(ds, component_idx, epsg_code, year):
    """
    Extracts the yield for a single component (rainfed or irrigated) after assigning CRS and reprojecting.

    Parameters:
    - ds (xarray.Dataset): The dataset containing yield data.
    - component_idx (int): The index of the component in the 'crops' dimension.
    - epsg_code (int): The EPSG code for the coordinate reference system.
    - year (int): The year index to extract.

    Returns:
    - xarray.DataArray: The yield for the component for the specified year.
    """
    # Extract yield for the given component and year
    component_yield = ds['yield'].isel(crops=component_idx, time=year).fillna(0).astype(np.float32)
    # Assign CRS and reproject
    component_yield = assign_crs(component_yield, epsg_code)
    return component_yield


def aggregate_grass_yields(ds, epsg_code, year):
    """
    Sum the yields of C3grass and C4grass across the 'grass' dimension after assigning CRS and reprojecting.

    Parameters:
    - ds (xarray.Dataset): The dataset containing grass yield data.
    - epsg_code (int): The EPSG code for the coordinate reference system.
    - year (int): The year index to extract.

    Returns:
    - xarray.DataArray: The total yield for grasses for the specified year.
    """
    grass_yields = []
    for grass_idx in range(len(ds['grass'])):
        # Extract yield for this grass type and the given year
        grass_yield = ds['yield'].isel(grass=grass_idx, time=year).fillna(0).astype(np.float32)
        # Assign CRS and reproject
        grass_yield = assign_crs(grass_yield, epsg_code)
        grass_yields.append(grass_yield)
    # Sum the reprojected yields
    total_yield = sum(grass_yields)
    return total_yield


def clip_yield_to_country(yield_data, country_geometry, epsg_code):
    """
    Clip yield data to a country geometry.

    Parameters:
    - yield_data (xarray.DataArray): The yield data to be clipped.
    - country_geometry (list): A list containing the country's geometry.
    - epsg_code (int): The EPSG code for the coordinate reference system.

    Returns:
    - xarray.DataArray: The clipped yield data for the country.
    """
    clipped_yield = yield_data.rio.clip(
        country_geometry, crs=f"EPSG:{epsg_code}", drop=False, all_touched=True
    )
    return clipped_yield


def calculate_percentage_change(yield_value, reference_yield_value):
    """
    Calculate the percentage change between yield and reference yield.

    Parameters:
    - yield_value (float): The yield value for the target dataset.
    - reference_yield_value (float): The yield value for the reference dataset.

    Returns:
    - float: The percentage change in yield.
    """
    if reference_yield_value > 0:
        percentage_change = 100 * (yield_value - reference_yield_value) / reference_yield_value
    else:
        percentage_change = np.nan
    return percentage_change


def process_yields(ds, control_datasets, gdf, aggregation_info, names_list, years, epsg_code, country_mapping, component_type, include_grasses=False, ds_grass=None, grass_control_datasets=None, grass_names=None):
    """
    Unified function to process yield data for crops and optionally grasses, processing either rainfed or irrigated components.

    Parameters:
    - ds (xarray.Dataset): The target dataset containing yield data for crops.
    - control_datasets (list of xarray.Dataset): List of control datasets for crops.
    - gdf (geopandas.GeoDataFrame): GeoDataFrame containing country geometries.
    - aggregation_info (dict): Dictionary mapping main crops to their components.
    - names_list (list): List of names from the crop datasets.
    - years (int): Number of years to process.
    - epsg_code (int): The EPSG code for the coordinate reference system.
    - country_mapping (dict): Dictionary mapping old country names to new ones.
    - component_type (str): 'rainfed' or 'irrigated'.
    - include_grasses (bool): Whether to process grasses as well.
    - ds_grass (xarray.Dataset): The target dataset containing yield data for grasses.
    - grass_control_datasets (list of xarray.Dataset): List of control datasets for grasses.
    - grass_names (list): List of names from the grass datasets.

    Returns:
    - pandas.DataFrame: DataFrame containing percentage changes in yield for each country and group.
    """
    # Initialize a dictionary to store percentage changes for each country
    percentage_changes = {}

    # Process crops
    groups = aggregation_info.items()

    # Loop over each group
    for group_name, sub_components in tqdm(groups, desc=f'Processing crops ({component_type})'):
        # Iterate over each year
        for year in range(years):
            # For crops, process specified component
            if component_type == 'rainfed':
                component = sub_components[0]
            elif component_type == 'irrigated':
                component = sub_components[1]
            else:
                logging.error(f"Invalid component_type: {component_type}")
                continue

            component_idx = names_list.index(component)
            # Get the yield data for this component
            yield_year = get_component_yield(ds, component_idx, epsg_code, year)
            # Process control datasets
            reference_yield_years = []
            for ctrl_ds in control_datasets:
                reference_yield_year = get_component_yield(ctrl_ds, component_idx, epsg_code, year)
                reference_yield_years.append(reference_yield_year)
            # Align and average the reference yield datasets
            reference_yield_years_aligned = xr.align(*reference_yield_years, join='exact')
            reference_yield_year_avg = xr.concat(reference_yield_years_aligned, dim='dataset').mean(dim='dataset')

            # Now process each country
            for idx, country in gdf.iterrows():
                country_name = country['COUNTRY']
                country_iso = country['ISO']

                # Apply country name mapping if necessary
                country_name = country_mapping.get(country_name, country_name)
                country_geometry = [country['geometry']]

                try:
                    # Clip the yield data to the country geometry
                    clipped_yield = clip_yield_to_country(
                        yield_year, country_geometry, epsg_code
                    )
                    clipped_reference_yield = clip_yield_to_country(
                        reference_yield_year_avg, country_geometry, epsg_code
                    )

                    # Calculate the yield for the country
                    yield_value = clipped_yield.sum(skipna=True).item()
                    reference_yield_value = clipped_reference_yield.sum(skipna=True).item()

                    # Calculate the percentage change
                    percentage_change = calculate_percentage_change(
                        yield_value, reference_yield_value
                    )

                    # Initialize the dictionary entry for the country if it doesn't exist
                    if country_name not in percentage_changes:
                        percentage_changes[country_name] = {
                            'ISO3 Country Code': country_iso,
                            'Country': country_name
                        }

                    # Adjust the group name for special cases
                    if group_name == "Wheat":
                        group_col_name = "spring_wheat"
                    else:
                        group_col_name = group_name.lower()

                    # Store the percentage change
                    # Include component_type in the column name
                    percentage_changes[country_name][f'{group_col_name}_{component_type}_year{year + 1}'] = percentage_change

                except Exception as e:
                    logging.error(f"Could not process {country_name} for year {year + 1}, group {group_name}, component {component_type}: {e}")

    # If include_grasses is True and component_type is 'rainfed', process grasses
    if include_grasses and component_type == 'rainfed':
        # Process grasses
        group_name = 'grasses'
        for year in range(years):
            yield_year = aggregate_grass_yields(ds_grass, epsg_code, year)
            # Aggregate yields for each control dataset
            reference_yield_years = []
            for ctrl_ds in grass_control_datasets:
                reference_yield_year = aggregate_grass_yields(ctrl_ds, epsg_code, year)
                reference_yield_years.append(reference_yield_year)
            # Align and average the reference yield datasets
            reference_yield_years_aligned = xr.align(*reference_yield_years, join='exact')
            reference_yield_year_avg = xr.concat(reference_yield_years_aligned, dim='dataset').mean(dim='dataset')

            # Now process each country
            for idx, country in gdf.iterrows():
                country_name = country['COUNTRY']
                country_iso = country['ISO']

                # Apply country name mapping if necessary
                country_name = country_mapping.get(country_name, country_name)
                country_geometry = [country['geometry']]

                try:
                    # Clip the yield data to the country geometry
                    clipped_yield = clip_yield_to_country(
                        yield_year, country_geometry, epsg_code
                    )
                    clipped_reference_yield = clip_yield_to_country(
                        reference_yield_year_avg, country_geometry, epsg_code
                    )

                    # Calculate the total yield for the country
                    yield_value = clipped_yield.sum(skipna=True).item()
                    reference_yield_value = clipped_reference_yield.sum(skipna=True).item()

                    # Calculate the percentage change
                    percentage_change = calculate_percentage_change(
                        yield_value, reference_yield_value
                    )

                    # Initialize the dictionary entry for the country if it doesn't exist
                    if country_name not in percentage_changes:
                        percentage_changes[country_name] = {
                            'ISO3 Country Code': country_iso,
                            'Country': country_name
                        }

                    # Store the percentage change
                    percentage_changes[country_name][f'{group_name}_year{year + 1}'] = percentage_change

                except Exception as e:
                    logging.error(f"Could not process {country_name} for year {year + 1}, group {group_name}: {e}")

    result_df = pd.DataFrame.from_dict(percentage_changes, orient='index').reset_index(drop=True)

    # Sort by country name
    result_df = result_df.sort_values(by='Country').reset_index(drop=True)

    return result_df


def process_scenario(scenario_name, crop_file_paths, grass_file_paths, control_crop_files, control_grass_files, gdf, crop_aggregation, country_mapping, EPSG=4326, years=10):
    """
    Process a single scenario and output the results to CSV files.

    Parameters:
    - scenario_name (str): Name of the scenario (e.g., '150Tg', '5Tg', etc.).
    - crop_file_paths (list): List of file paths for the crop datasets for this scenario.
    - grass_file_paths (list): List of file paths for the grass datasets for this scenario.
    - control_crop_files (list): List of control crop dataset file paths.
    - control_grass_files (list): List of control grass dataset file paths.
    - gdf (geopandas.GeoDataFrame): GeoDataFrame containing country geometries.
    - crop_aggregation (dict): Dictionary mapping main crops to their rainfed and irrigated components.
    - country_mapping (dict): Dictionary mapping old country names to new ones.
    - EPSG (int): The EPSG code for the coordinate reference system.
    - years (int): Number of years to process.

    Returns:
    - None
    """
    # Load control datasets
    control_datasets = [fix_longitude(xr.open_dataset(fp)) for fp in control_crop_files]
    grass_control_datasets = [fix_longitude(xr.open_dataset(fp)) for fp in control_grass_files]

    # Lists to store results
    combined_results_rainfed = []
    combined_results_irrigated = []

    # Process each pair of crop and grass files
    for idx, (crop_file, grass_file) in enumerate(zip(crop_file_paths, grass_file_paths)):
        ds_crop = fix_longitude(xr.open_dataset(crop_file))
        ds_grass = fix_longitude(xr.open_dataset(grass_file))
        crop_names = ds_crop['crops'].attrs['long_name'].split(',')
        grass_names = ds_grass['grass'].attrs['long_name'].split(', ')

        # Process rainfed components (including grasses)
        if 'yield' in ds_crop.data_vars and all('yield' in ctrl_ds.data_vars for ctrl_ds in control_datasets):
            result_rainfed_df = process_yields(
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
                grass_names=grass_names
            )
        else:
            logging.warning(f"Crop dataset does not contain a 'yield' variable for scenario {scenario_name}.")
            continue

        # Process irrigated components
        if 'yield' in ds_crop.data_vars and all('yield' in ctrl_ds.data_vars for ctrl_ds in control_datasets):
            result_irrigated_df = process_yields(
                ds=ds_crop,
                control_datasets=control_datasets,
                gdf=gdf,
                aggregation_info=crop_aggregation,
                names_list=crop_names,
                years=years,
                epsg_code=EPSG,
                country_mapping=country_mapping,
                component_type='irrigated',
                include_grasses=False
            )
        else:
            logging.warning(f"Crop dataset does not contain a 'yield' variable for scenario {scenario_name}.")
            continue

        # Save the DataFrames
        output_filename_rainfed = f'results/output_{scenario_name}_crops_and_grasses_rainfed_{idx + 1}.csv'
        result_rainfed_df.to_csv(output_filename_rainfed, index=False)
        logging.info(f"Saved rainfed results to {output_filename_rainfed}")
        combined_results_rainfed.append(result_rainfed_df)

        output_filename_irrigated = f'results/output_{scenario_name}_crops_irrigated_{idx + 1}.csv'
        result_irrigated_df.to_csv(output_filename_irrigated, index=False)
        logging.info(f"Saved irrigated results to {output_filename_irrigated}")
        combined_results_irrigated.append(result_irrigated_df)

    # For scenarios with multiple files (e.g., 5Tg), compute aggregated results
    if len(combined_results_rainfed) > 1:
        # Average combined results for rainfed
        avg_combined_rainfed_df = pd.concat(combined_results_rainfed).groupby(['ISO3 Country Code', 'Country']).mean().reset_index()
        # Save the aggregated result
        output_filename_rainfed = f'results/output_{scenario_name}_crops_and_grasses_rainfed_aggregated.csv'
        avg_combined_rainfed_df.to_csv(output_filename_rainfed, index=False)
        logging.info(f"Saved aggregated rainfed results to {output_filename_rainfed}")

    if len(combined_results_irrigated) > 1:
        # Average combined results for irrigated
        avg_combined_irrigated_df = pd.concat(combined_results_irrigated).groupby(['ISO3 Country Code', 'Country']).mean().reset_index()
        # Save the aggregated result
        output_filename_irrigated = f'results/output_{scenario_name}_crops_irrigated_aggregated.csv'
        avg_combined_irrigated_df.to_csv(output_filename_irrigated, index=False)
        logging.info(f"Saved aggregated irrigated results to {output_filename_irrigated}")


def main():
    """
    Main function to execute the crop and grass yield percentage change calculation for multiple scenarios.
    """
    setup_logging()

    # Load configuration
    config = load_config('config/config.yaml')

    # Control files (common across nuclear winter soot level scenarios)
    control_crop_files = config['control_crop_files']
    control_grass_files = config['control_grass_files']

    # Scenarios to process
    scenarios = config['scenarios']

    # Path to shapefile with the country boundaries
    shapefile_path = config['shapefile_path']

    gdf = gpd.read_file(shapefile_path)

    # Coordinate Reference System: Currently 4326 for consistency with input files
    EPSG = config['EPSG']

    # Ensure the CRS is consistent
    gdf = gdf.to_crs(epsg=EPSG)

    # Crop aggregation dictionary: maps crop categories to their rainfed and irrigated components
    crop_aggregation = config['crop_aggregation']

    # Country name mapping for consistency
    country_mapping = config['country_mapping']

    # Process each scenario of nuclear winter (different soot levels)
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
            EPSG=EPSG,
            years=config['years']
        )


if __name__ == '__main__':
    main()
