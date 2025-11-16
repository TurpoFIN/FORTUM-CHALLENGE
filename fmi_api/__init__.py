"""
FMI Open Data API Integration

This package provides a Python interface to the Finnish Meteorological Institute's
Open Data WFS API for accessing weather observations and forecasts.

Basic usage:
    >>> from fmi_api import get_observations, get_forecast
    >>> 
    >>> # Get historical observations
    >>> obs_df = get_observations(
    ...     place='Helsinki',
    ...     start_date='2023-01-01T00:00:00Z',
    ...     end_date='2023-01-31T23:59:59Z'
    ... )
    >>> 
    >>> # Get weather forecast
    >>> forecast_df = get_forecast(
    ...     place='Helsinki',
    ...     start_time='2024-01-01T00:00:00Z',
    ...     end_time='2024-01-03T00:00:00Z'
    ... )
"""

from .fmi_client import FMIClient

# Observation functions
from .observations import (
    get_observations,
    get_temperature_observations,
    get_wind_observations,
    get_precipitation_observations,
    get_comprehensive_weather_observations,
    pivot_observations,
    aggregate_to_daily
)

# Forecast functions
from .forecasts import (
    get_forecast,
    get_temperature_forecast,
    get_wind_forecast,
    get_precipitation_forecast,
    get_comprehensive_weather_forecast,
    pivot_forecast
)

# Station functions
from .weather_stations import (
    get_all_stations,
    get_nearby_stations,
    find_station_by_name,
    find_stations_for_power_plants,
    get_station_info
)

# Utility functions
from .utils import (
    save_to_parquet,
    load_from_parquet,
    save_to_csv,
    load_from_csv
)

# Configuration
from .config import (
    WEATHER_PARAMETERS,
    FORECAST_PARAMETERS
)

__version__ = '0.1.0'

__all__ = [
    # Client
    'FMIClient',
    
    # Observations
    'get_observations',
    'get_temperature_observations',
    'get_wind_observations',
    'get_precipitation_observations',
    'get_comprehensive_weather_observations',
    'pivot_observations',
    'aggregate_to_daily',
    
    # Forecasts
    'get_forecast',
    'get_temperature_forecast',
    'get_wind_forecast',
    'get_precipitation_forecast',
    'get_comprehensive_weather_forecast',
    'pivot_forecast',
    
    # Stations
    'get_all_stations',
    'get_nearby_stations',
    'find_station_by_name',
    'find_stations_for_power_plants',
    'get_station_info',
    
    # Utils
    'save_to_parquet',
    'load_from_parquet',
    'save_to_csv',
    'load_from_csv',
    
    # Config
    'WEATHER_PARAMETERS',
    'FORECAST_PARAMETERS',
]


