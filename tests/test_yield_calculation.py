import pathlib
import importlib.util
import sys

import numpy as np
import pytest
import xarray as xr

# -----------------------------------------------------------------------------
# Dynamically load the main module (file is named with a leading digit)
# -----------------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SRC_FILE = ROOT_DIR / "src" / "1_yield_change_calculation.py"

spec = importlib.util.spec_from_file_location("yield_change", SRC_FILE)
yc = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(yc)  # type: ignore


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_dataset():
    """Return a minimal synthetic crop‑yield xarray.Dataset for tests."""
    lon = np.array([0, 90, 180])
    lat = np.array([-45, 45])

    # dims: crops (2), time (2), lat (2), lon (3)
    data = np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(np.float32)
    ds = xr.Dataset(
        {
            "yield": (("crops", "time", "lat", "lon"), data),
        },
        coords={
            "crops": [0, 1],
            "time": [0, 1],
            "lat": lat,
            "lon": lon,
        },
    )
    # Pretend long names for crops
    ds["crops"].attrs["long_name"] = "maize_rainfed,maize_irrigated"
    return ds


# -----------------------------------------------------------------------------
# Tests for independent / simple logic
# -----------------------------------------------------------------------------

def test_calculate_percentage_change_normal():
    """Positive reference – returns expected percentage."""
    result = yc.calculate_percentage_change(120, 100)
    assert pytest.approx(result) == 20.0


def test_calculate_percentage_change_zero_reference():
    """Zero reference – returns NaN (avoid divide‑by‑zero)."""
    result = yc.calculate_percentage_change(50, 0)
    assert np.isnan(result)


# -----------------------------------------------------------------------------
# Tests for longitude wrapping / sorting
# -----------------------------------------------------------------------------

def test_fix_longitude(simple_dataset):
    wrapped = yc.fix_longitude(simple_dataset.copy())
    # Longitudes should now be in (‑180, 180] and sorted ascending
    assert np.all((wrapped["lon"] >= -180) & (wrapped["lon"] <= 180))
    assert np.all(np.diff(wrapped["lon"]) > 0)


# -----------------------------------------------------------------------------
# Tests for data‑selection helpers (stub out heavy reprojection)
# -----------------------------------------------------------------------------

def test_get_component_yield(monkeypatch, simple_dataset):
    """Ensure getter selects correct crop/time slice and keeps dimensions."""

    # Stub assign_crs to identity to sidestep rioxarray
    monkeypatch.setattr(yc, "assign_crs", lambda da, epsg: da)

    comp = yc.get_component_yield(simple_dataset, component_idx=1, epsg_code=4326, year=1)
    # Should be DataArray with lat/lon dims
    assert isinstance(comp, xr.DataArray)
    assert set(comp.dims) == {"lat", "lon"}
    # Values should equal the correct slice from the original data
    expected = simple_dataset["yield"].isel(crops=1, time=1)
    assert comp.equals(expected)


# -----------------------------------------------------------------------------
# Tests for grass aggregation helper (stub out reprojection again)
# -----------------------------------------------------------------------------

def test_aggregate_grass_yields(monkeypatch):
    """Sum across grass dimension and verify aggregation logic."""
    # Build tiny grass dataset with dims grass, time, lat, lon (sizes: 2,1,1,1)
    data = np.array([[[[1]]], [[[2]]]], dtype=np.float32)  # grass types 1 & 2
    ds_grass = xr.Dataset(
        {
            "yield": (("grass", "time", "lat", "lon"), data),
        },
        coords={
            "grass": [0, 1],
            "time": [0],
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    # Stub assign_crs
    monkeypatch.setattr(yc, "assign_crs", lambda da, epsg: da)

    total = yc.aggregate_grass_yields(ds_grass, epsg_code=4326, year=0)
    assert isinstance(total, xr.DataArray)
    # Should equal 3 everywhere
    assert float(total) == 3.0
