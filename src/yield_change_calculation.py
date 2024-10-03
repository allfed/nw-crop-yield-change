import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import logging
import yaml
from pathlib import Path


def setup_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("yield_processing.log"),
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


def aggregate_yields(ds, rain_idx, irr_idx):
    """
    Extract and sum rainfed and irrigated yields from a dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset containing yield data.
    - rain_idx (int): The index of the rainfed crop in the 'crops' dimension.
    - irr_idx (int): The index of the irrigated crop in the 'crops' dimension.

    Returns:
    - xarray.DataArray: The total yield (rainfed + irrigated) for the crop.
    """
    rain_yield = ds['yield'].isel(crops=rain_idx).fillna(0).astype(np.float32)
    irr_yield = ds['yield'].isel(crops=irr_idx).fillna(0).astype(np.float32)
    total_yield = (rain_yield + irr_yield).astype(np.float32)
    return total_yield


def aggregate_grass_yields(ds):
    """
    Sum the yields of C3grass and C4grass across the 'grass' dimension.

    Parameters:
    - ds (xarray.Dataset): The dataset containing grass yield data.

    Returns:
    - xarray.DataArray: The total yield for grasses.
    """
    total_yield = ds['yield'].fillna(0).astype(np.float32).sum(dim='grass')
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
        country_geometry, crs=f"EPSG:{epsg_code}", drop=False, all_touched=False
    )
    return clipped_yield


def calculate_percentage_change(total_yield_value, reference_yield_value):
    """
    Calculate the percentage change between total yield and reference yield.

    Parameters:
    - total_yield_value (float): The total yield value for the target dataset.
    - reference_yield_value (float): The total yield value for the reference dataset.

    Returns:
    - float: The percentage change in yield.
    """
    if reference_yield_value > 0:
        percentage_change = 100 * (total_yield_value - reference_yield_value) / reference_yield_value
    else:
        percentage_change = np.nan
    return percentage_change


def process_yields(
    ds,
    control_datasets,
    gdf,
    aggregation_dict,
    names,
    years,
    epsg_code,
    country_mapping,
    yield_type
):
    """
    Process yields (crops or grasses), calculate percentage changes, and return the result DataFrame.

    Parameters:
    - ds (xarray.Dataset): The target dataset containing yield data.
    - control_datasets (list of xarray.Dataset): List of control datasets.
    - gdf (geopandas.GeoDataFrame): GeoDataFrame containing country geometries.
    - aggregation_dict (dict): Dictionary mapping main categories to their components.
    - names (list): List of names from the datasets (crops or grasses).
    - years (int): Number of years to process.
    - epsg_code (int): The EPSG code for the coordinate reference system.
    - country_mapping (dict): Dictionary mapping old country names to new ones.
    - yield_type (str): Type of yield being processed ('crop' or 'grass').

    Returns:
    - pandas.DataFrame: DataFrame containing percentage changes in yields for each country and category.
    """
    # Initialize a dictionary to store percentage changes for each country
    percentage_changes = {}

    # Determine the aggregation function based on yield_type
    if yield_type == 'crop':
        aggregate_func = aggregate_yields
    elif yield_type == 'grass':
        aggregate_func = aggregate_grass_yields
    else:
        logging.error(f"Invalid yield_type: {yield_type}. Must be 'crop' or 'grass'.")
        raise ValueError("Invalid yield_type. Must be 'crop' or 'grass'.")

    # Loop over each category (crop or grass group), aggregate yields, and calculate percentage changes
    for main_category, sub_components in tqdm(aggregation_dict.items(), desc=f'Processing {yield_type}s'):
        if yield_type == 'crop':
            # For crops, sub_components are [Rainfed, Irrigated]
            rain_component, irr_component = sub_components
            try:
                rain_idx = names.index(rain_component)
                irr_idx = names.index(irr_component)
            except ValueError as e:
                logging.error(f"Component {e} not found in names list for category {main_category}.")
                continue

            # Aggregate yields for target dataset
            total_yield = aggregate_func(ds, rain_idx, irr_idx)
        elif yield_type == 'grass':
            # For grasses, sub_components might be different; assuming they are not used similarly
            # If grasses have multiple components, adjust accordingly
            total_yield = aggregate_func(ds)

        # Iterate over each year
        for year in range(years):
            # Extract the yield data for the given year and assign CRS
            total_yield_year = total_yield.isel(time=year)
            total_yield_year = assign_crs(total_yield_year, epsg_code)

            # Initialize a list to store the percentage changes for each reference dataset
            year_percentage_changes = []

            # For each control dataset
            for ctrl_ds in control_datasets:
                if yield_type == 'crop':
                    # Aggregate yields for control dataset
                    reference_yield = aggregate_func(ctrl_ds, rain_idx, irr_idx)
                elif yield_type == 'grass':
                    # Aggregate yields for control dataset
                    reference_yield = aggregate_func(ctrl_ds)

                # Extract the yield data for the given year in the reference dataset
                reference_yield_year = reference_yield.isel(time=year)
                # Assign CRS
                reference_yield_year = assign_crs(reference_yield_year, epsg_code)

                # Iterate over each country polygon in the GeoDataFrame
                for idx, country in gdf.iterrows():
                    country_name = country['COUNTRY']
                    country_iso = country['ISO']

                    # Apply country name mapping if necessary
                    if country_name in country_mapping:
                        country_name = country_mapping[country_name]

                    country_geometry = [country['geometry']]

                    try:
                        # Clip the yield data to the country geometry
                        clipped_total_yield = clip_yield_to_country(
                            total_yield_year, country_geometry, epsg_code
                        )
                        clipped_reference_yield = clip_yield_to_country(
                            reference_yield_year, country_geometry, epsg_code
                        )

                        # Calculate the total yield for the country
                        total_yield_value = clipped_total_yield.sum(skipna=True).item()
                        reference_yield_value = clipped_reference_yield.sum(skipna=True).item()

                        # Calculate the percentage change
                        percentage_change = calculate_percentage_change(
                            total_yield_value, reference_yield_value
                        )

                        # Append the percentage change to the list for this year
                        year_percentage_changes.append({
                            'country': country_name,
                            'iso': country_iso,
                            'percentage_change': percentage_change
                        })

                    except Exception as e:
                        logging.error(f"Could not process {country_name} for year {year + 1} and {yield_type} {main_category}: {e}")

            # Now average the percentage changes across all reference datasets
            if year_percentage_changes:
                year_df = pd.DataFrame(year_percentage_changes)

                # Average the percentage changes for each country
                average_changes = year_df.groupby(['country', 'iso'])['percentage_change'].mean().reset_index()

                # Store the results
                for _, row in average_changes.iterrows():
                    country_name = row['country']
                    country_iso = row['iso']
                    avg_percentage_change = row['percentage_change']

                    # Initialize the dictionary entry for the country if it doesn't exist
                    if country_name not in percentage_changes:
                        percentage_changes[country_name] = {
                            'ISO3 Country Code': country_iso,
                            'Country': country_name
                        }

                    # Store the average percentage change
                    if yield_type == "crop":
                        if main_category == "Wheat":
                            category_name = "spring_wheat"
                        else:
                            category_name = main_category.lower()
                        key = f'{category_name}_year{year + 1}'
                    elif yield_type == "grass":
                        key = f'{yield_type}s_year{year + 1}'

                    percentage_changes[country_name][key] = avg_percentage_change

    result_df = pd.DataFrame.from_dict(percentage_changes, orient='index').reset_index(drop=True)

    # Sort by country name
    result_df = result_df.sort_values(by='Country').reset_index(drop=True)

    return result_df


def process_scenario(
    scenario_name,
    crop_file_paths,
    grass_file_paths,
    control_crop_files,
    control_grass_files,
    gdf,
    crop_aggregation,
    country_mapping,
    EPSG=6933,
    years=10
):
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

    # Process crops
    crop_results = []
    for idx, crop_file in enumerate(crop_file_paths):
        ds = fix_longitude(xr.open_dataset(crop_file))
        crop_names = ds['crops'].attrs['long_name'].split(',')

        # Process crops using the unified function
        if 'yield' in ds.data_vars and all('yield' in ctrl_ds.data_vars for ctrl_ds in control_datasets):
            result_df = process_yields(
                ds=ds,
                control_datasets=control_datasets,
                gdf=gdf,
                aggregation_dict=crop_aggregation,
                names=crop_names,
                years=years,
                epsg_code=EPSG,
                country_mapping=country_mapping,
                yield_type='crop'
            )
            crop_results.append(result_df)
        else:
            logging.warning(f"One or more crop datasets do not contain a 'yield' variable for scenario {scenario_name}.")
            continue

    # Process grasses
    grass_results = []
    for idx, grass_file in enumerate(grass_file_paths):
        grass_ds = fix_longitude(xr.open_dataset(grass_file))
        grass_names = grass_ds['grass'].attrs['long_name'].split(', ')

        # Process grasses using the unified function
        if 'yield' in grass_ds.data_vars and all('yield' in ctrl_ds.data_vars for ctrl_ds in grass_control_datasets):
            grass_result_df = process_yields(
                ds=grass_ds,
                control_datasets=grass_control_datasets,
                gdf=gdf,
                aggregation_dict={},  # Assuming no sub-components for grasses
                names=grass_names,
                years=years,
                epsg_code=EPSG,
                country_mapping=country_mapping,
                yield_type='grass'
            )
            grass_results.append(grass_result_df)
        else:
            logging.warning(f"One or more grass datasets do not contain a 'yield' variable for scenario {scenario_name}.")
            continue

    # Merge crop and grass results
    for idx, (crop_df, grass_df) in enumerate(zip(crop_results, grass_results)):
        final_df = pd.merge(crop_df, grass_df, on=['ISO3 Country Code', 'Country'], how='outer')
        final_df = final_df.sort_values(by='Country').reset_index(drop=True)
        output_filename = f'data/processed/output_{scenario_name}_crops_and_grasses_{idx + 1}.csv'
        final_df.to_csv(output_filename, index=False)
        logging.info(f"Saved combined results to {output_filename}")

    # For scenarios with multiple files (e.g., 5Tg), compute aggregated results
    if len(crop_results) > 1 and len(grass_results) > 1:
        # Average crop results
        avg_crop_df = pd.concat(crop_results).groupby(['ISO3 Country Code', 'Country']).mean().reset_index()
        # Average grass results
        avg_grass_df = pd.concat(grass_results).groupby(['ISO3 Country Code', 'Country']).mean().reset_index()
        # Merge averaged results
        final_avg_df = pd.merge(avg_crop_df, avg_grass_df, on=['ISO3 Country Code', 'Country'], how='outer')
        final_avg_df = final_avg_df.sort_values(by='Country').reset_index(drop=True)
        # Save the aggregated result
        output_filename = f'data/processed/output_{scenario_name}_crops_and_grasses_aggregated.csv'
        final_avg_df.to_csv(output_filename, index=False)
        logging.info(f"Saved aggregated results to {output_filename}")


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

    # Coordinate Reference System: MUST BE 6933 for valid results unless testing
    EPSG = config['EPSG']

    # Ensure the CRS is consistent
    gdf = gdf.to_crs(epsg=EPSG)

    # Crop aggregation dictionary: maps crop categories to their rain and irrigated components
    crop_aggregation = config['crop_aggregation']

    # Country name mapping for consistency with rutgers_nw_production_raw.csv
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
