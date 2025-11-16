"""
Configuration for FMI Open Data API
"""

import os
from typing import Optional

# FMI Open Data WFS endpoint
FMI_WFS_BASE_URL = "https://opendata.fmi.fi/wfs"

# Default timeout for API requests
DEFAULT_TIMEOUT = 60

# Maximum retries for failed requests
MAX_RETRIES = 3

# Stored queries available in FMI Open Data
# Updated based on actual API capabilities (Nov 2024)
STORED_QUERIES = {
    # Observations - Real-time/Instantaneous
    'observations_realtime': 'fmi::observations::weather::timevaluepair',
    'observations_multipointcoverage': 'fmi::observations::weather::multipointcoverage',
    'observations_simple': 'fmi::observations::weather::simple',
    
    # Observations - Time-aggregated
    'observations_hourly': 'fmi::observations::weather::hourly::timevaluepair',
    'observations_daily': 'fmi::observations::weather::daily::timevaluepair',
    
    # Forecasts - Official edited forecast (best for general use)
    'forecast_edited': 'fmi::forecast::edited::weather::scandinavia::point::timevaluepair',
    'forecast_edited_simple': 'fmi::forecast::edited::weather::scandinavia::point::simple',
    
    # Forecasts - HARMONIE model (high resolution)
    'forecast_harmonie': 'fmi::forecast::harmonie::surface::point::timevaluepair',
    'forecast_harmonie_simple': 'fmi::forecast::harmonie::surface::point::simple',
    
    # Forecasts - MEPS (MetCoOp Ensemble Prediction System)
    'forecast_meps': 'fmi::forecast::meps::surface::point::timevaluepair',
    
    # Forecasts - ECMWF (European model, longer range)
    'forecast_ecmwf': 'ecmwf::forecast::surface::point::timevaluepair',
}

# Common weather parameters
WEATHER_PARAMETERS = {
    'temperature': 't2m',              # Air temperature at 2m (°C)
    'humidity': 'rh',                  # Relative humidity (%)
    'wind_speed': 'ws_10min',          # Wind speed 10min avg (m/s)
    'wind_direction': 'wd_10min',      # Wind direction 10min avg (°)
    'wind_gust': 'wg_10min',           # Wind gust 10min (m/s)
    'precipitation': 'r_1h',           # Precipitation amount 1h (mm)
    'pressure': 'p_sea',               # Pressure at sea level (hPa)
    'visibility': 'vis',               # Horizontal visibility (m)
    'cloud_amount': 'n_man',           # Cloud amount (1/8)
    'snow_depth': 'snow_aws',          # Snow depth (cm)
}

# Forecast parameters (slightly different naming)
FORECAST_PARAMETERS = {
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'wind_speed': 'WindSpeedMS',
    'wind_direction': 'WindDirection',
    'wind_gust': 'WindGust',
    'precipitation_1h': 'Precipitation1h',
    'pressure': 'Pressure',
    'cloud_cover': 'TotalCloudCover',
    'wind_u': 'WindUMS',              # Wind U component
    'wind_v': 'WindVMS',              # Wind V component
}


def get_fmi_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Get FMI API credentials from environment.
    
    Note: FMI Open Data API does not require authentication,
    but we keep this function for consistency with other APIs.
    
    Returns:
        Tuple of (api_key, api_secret) - both None for FMI
    """
    return None, None


