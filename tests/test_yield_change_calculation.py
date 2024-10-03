# test_yield_change_calculation.py

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import Polygon
from pathlib import Path
import logging

# Import the functions to test
from src.yield_change_calculation import (
    fix_longitude,
    assign_crs,
    aggregate_yields,
    aggregate_grass_yields,
    clip_yield_to_country,
    calculate_percentage_change,
    process_yields
)

# Disable logging during tests to avoid clutter
logging.disable(logging.CRITICAL)


def test_fix_longitude():
    # Create a test dataset with lon coordinates from 0 to 360
    lon = np.array([0, 90, 180, 270, 360])
    lat = np.array([-90, 0, 90])
    data = np.random.rand(len(lat), len(lon))
    ds = xr.Dataset({'data': (('lat', 'lon'), data)}, coords={'lat': lat, 'lon': lon})

    # Apply fix_longitude
    ds_fixed = fix_longitude(ds)

    # Expected lon values after fixing
    expected_lon = np.array([-180, -90, 0, 90, 180])

    # Check that lon coordinates are now in [-180, 180]
    np.testing.assert_array_equal(ds_fixed['lon'].values, expected_lon)

    # Since we sorted by 'lon', we need to rearrange the original data accordingly
    # Create mapping from original lon indices to fixed lon indices
    # Original lon indices: [0,1,2,3,4] corresponds to lon=[0,90,180,270,360]
    # Fixed lon indices after sortby('lon'): [0,1,2,3,4] corresponds to lon=[-180,-90,0,90,180]
    # Mapping: [4, 1, 0, 1, 2] (we wrap around at 360 degrees)
    rearranged_indices = [4, 0, 1, 2, 3]  # Adjusted mapping
    expected_data = data[:, rearranged_indices]

    np.testing.assert_array_equal(ds_fixed['data'].values, expected_data)


def test_assign_crs():
    # Create a test DataArray
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-90, 90, 180)
    data = np.random.rand(len(lat), len(lon))
    da = xr.DataArray(data, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

    # Apply assign_crs
    epsg_code = 4326  # WGS84
    da_with_crs = assign_crs(da, epsg_code)

    # Check that CRS is assigned
    assert da_with_crs.rio.crs.to_epsg() == epsg_code

    # Check that spatial dimensions are set correctly
    assert da_with_crs.rio.x_dim == 'lon'
    assert da_with_crs.rio.y_dim == 'lat'


def test_assign_crs_missing_lat():
    # Create a DataArray without 'lat' coordinate
    lon = np.linspace(-180, 180, 10)
    data = np.random.rand(10)
    da = xr.DataArray(data, coords={'lon': lon}, dims=['lon'])

    with pytest.raises(KeyError) as excinfo:
        assign_crs(da, 4326)
    assert "Missing 'lat' coordinate." in str(excinfo.value)


def test_assign_crs_invalid_lat():
    # Create a DataArray with invalid latitudes
    lon = np.linspace(-180, 180, 10)
    lat = np.array([-91, -45, 0, 45, 91])
    data = np.random.rand(len(lat), len(lon))
    da = xr.DataArray(data, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])

    with pytest.raises(ValueError) as excinfo:
        assign_crs(da, 4326)
    assert "Data contains invalid latitude values even after clipping." in str(excinfo.value)


def test_assign_crs_lat_lon_not_dimensions():
    # Create a DataArray where 'lat' and 'lon' are coordinates but not dimensions
    lon = np.linspace(-180, 180, 10)
    lat = np.linspace(-90, 90, 5)
    time = pd.date_range('2000-01-01', periods=3)
    data = np.random.rand(len(time), len(lat), len(lon))
    da = xr.DataArray(
        data,
        coords={'time': time, 'lat': ('space', lat), 'lon': ('space', lon)},
        dims=['time', 'space', 'space']
    )

    with pytest.raises(ValueError) as excinfo:
        assign_crs(da, 4326)
    assert "'lat' and/or 'lon' are not dimensions of the DataArray." in str(excinfo.value)


def test_aggregate_yields():
    # Create a test dataset
    crops = ['CornRain', 'CornIrr', 'WheatRain', 'WheatIrr']
    lon = np.linspace(-180, 180, 10)
    lat = np.linspace(-90, 90, 5)
    data = np.random.rand(len(crops), len(lat), len(lon))
    ds = xr.Dataset({'yield': (('crops', 'lat', 'lon'), data)}, coords={'crops': crops, 'lat': lat, 'lon': lon})

    # Indices for CornRain and CornIrr
    rain_idx = 0
    irr_idx = 1
    total_yield = aggregate_yields(ds, rain_idx, irr_idx)

    # Expected total yield is the sum of rainfed and irrigated yields
    expected_total_yield = (ds['yield'].isel(crops=rain_idx).fillna(0).astype(np.float32) +
                            ds['yield'].isel(crops=irr_idx).fillna(0).astype(np.float32))

    xr.testing.assert_allclose(total_yield, expected_total_yield)


def test_aggregate_grass_yields():
    # Create a test dataset
    grasses = ['C3grass', 'C4grass']
    lon = np.linspace(-180, 180, 10)
    lat = np.linspace(-90, 90, 5)
    data = np.random.rand(len(grasses), len(lat), len(lon))
    ds = xr.Dataset({'yield': (('grass', 'lat', 'lon'), data)}, coords={'grass': grasses, 'lat': lat, 'lon': lon})

    total_yield = aggregate_grass_yields(ds)

    # Expected total yield is the sum over 'grass' dimension
    expected_total_yield = ds['yield'].fillna(0).astype(np.float32).sum(dim='grass')

    xr.testing.assert_allclose(total_yield, expected_total_yield)


def test_clip_yield_to_country():
    # Create a test DataArray
    lon = np.linspace(-10, 10, 20)
    lat = np.linspace(-10, 10, 20)
    data = np.random.rand(len(lat), len(lon))
    da = xr.DataArray(data, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])
    da = da.rio.write_crs("EPSG:4326")

    # Define a country geometry as a simple square polygon
    country_geom = [Polygon([(-5, -5), (-5, 5), (5, 5), (5, -5), (-5, -5)])]
    epsg_code = 4326

    # Clip
    clipped_da = clip_yield_to_country(da, country_geom, epsg_code)

    # Check that clipped_da only contains data within the polygon
    assert clipped_da['lon'].min() >= -5
    assert clipped_da['lon'].max() <= 5
    assert clipped_da['lat'].min() >= -5
    assert clipped_da['lat'].max() <= 5

    # Ensure that the clipped data is not empty
    assert not clipped_da.isnull().all()


def test_calculate_percentage_change():
    # Case where reference_yield_value > 0
    total_yield_value = 120
    reference_yield_value = 100
    expected_percentage_change = 20  # (120 - 100)/100 * 100
    percentage_change = calculate_percentage_change(total_yield_value, reference_yield_value)
    assert percentage_change == expected_percentage_change

    # Case where reference_yield_value = 0
    total_yield_value = 50
    reference_yield_value = 0
    percentage_change = calculate_percentage_change(total_yield_value, reference_yield_value)
    assert np.isnan(percentage_change)

    # Case where total_yield_value < reference_yield_value
    total_yield_value = 80
    reference_yield_value = 100
    expected_percentage_change = -20  # (80 - 100)/100 * 100
    percentage_change = calculate_percentage_change(total_yield_value, reference_yield_value)
    assert percentage_change == expected_percentage_change


def test_process_yields_invalid_yield_type():
    with pytest.raises(ValueError) as excinfo:
        process_yields(
            ds=None,
            control_datasets=None,
            gdf=None,
            aggregation_dict=None,
            names=None,
            years=None,
            epsg_code=None,
            country_mapping=None,
            yield_type='invalid_type'
        )
    assert "Invalid yield_type. Must be 'crop' or 'grass'." in str(excinfo.value)
