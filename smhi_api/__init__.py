"""
SMHI API Integration Module
Provides functions to fetch meteorological observation data from opendata.smhi.se
"""

from .api_client import SMHIClient
from .observations import (
    get_temperature_data,
    get_wind_speed_data,
    get_humidity_data,
    get_precipitation_data
)
from .bulk_fetcher import (
    fetch_powerplant_weather_data,
    PowerPlantWeatherFetcher
)

__all__ = [
    'SMHIClient',
    'get_temperature_data',
    'get_wind_speed_data',
    'get_humidity_data',
    'get_precipitation_data',
    'fetch_powerplant_weather_data',
    'PowerPlantWeatherFetcher'
]


