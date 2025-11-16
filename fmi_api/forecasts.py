"""
Functions for retrieving weather forecasts from FMI Open Data
"""

from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

from .fmi_client import FMIClient
from .config import FORECAST_PARAMETERS
from .observations import NAMESPACES, _parse_time_value_pairs


def get_forecast(
    place: Optional[str] = None,
    latlon: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get weather forecast from FMI.
    
    Args:
        place: Place name (e.g., 'Helsinki', 'Tampere')
        latlon: Latitude,longitude as 'lat,lon' (e.g., '60.17,24.94')
        start_time: Start time in ISO format (e.g., '2024-01-01T00:00:00Z')
        end_time: End time in ISO format
        parameters: List of parameters to retrieve (use config.FORECAST_PARAMETERS)
        timestep: Time step in minutes (default: 60)
        
    Returns:
        DataFrame with forecast data
        
    Example:
        >>> df = get_forecast(
        ...     place='Helsinki',
        ...     start_time='2024-01-01T00:00:00Z',
        ...     end_time='2024-01-03T00:00:00Z',
        ...     parameters=['Temperature', 'WindSpeedMS', 'Precipitation1h']
        ... )
    """
    # Default to next 48 hours if no times specified
    if start_time is None:
        start_dt = datetime.now(timezone.utc)
        end_dt = start_dt + timedelta(hours=48)
        start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    with FMIClient() as client:
        root = client.get_forecast(
            place=place,
            latlon=latlon,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters,
            timestep=timestep
        )
        
        # Use the same parser as observations (format is similar)
        df = _parse_time_value_pairs(root)
        
        # Rename timestamp to forecast_time for clarity
        if not df.empty and 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'forecast_time'})
        
        return df


def get_temperature_forecast(
    place: Optional[str] = None,
    latlon: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get temperature forecast.
    
    Args:
        place: Place name
        latlon: Latitude,longitude
        start_time: Start time in ISO format
        end_time: End time in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with temperature forecast in Celsius
    """
    return get_forecast(
        place=place,
        latlon=latlon,
        start_time=start_time,
        end_time=end_time,
        parameters=[FORECAST_PARAMETERS['temperature']],
        timestep=timestep
    )


def get_wind_forecast(
    place: Optional[str] = None,
    latlon: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get wind forecast (speed, direction, gust).
    
    Args:
        place: Place name
        latlon: Latitude,longitude
        start_time: Start time in ISO format
        end_time: End time in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with wind forecast
    """
    return get_forecast(
        place=place,
        latlon=latlon,
        start_time=start_time,
        end_time=end_time,
        parameters=[
            FORECAST_PARAMETERS['wind_speed'],
            FORECAST_PARAMETERS['wind_direction'],
            FORECAST_PARAMETERS['wind_gust']
        ],
        timestep=timestep
    )


def get_precipitation_forecast(
    place: Optional[str] = None,
    latlon: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get precipitation forecast.
    
    Args:
        place: Place name
        latlon: Latitude,longitude
        start_time: Start time in ISO format
        end_time: End time in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with precipitation forecast in mm
    """
    return get_forecast(
        place=place,
        latlon=latlon,
        start_time=start_time,
        end_time=end_time,
        parameters=[FORECAST_PARAMETERS['precipitation_1h']],
        timestep=timestep
    )


def get_comprehensive_weather_forecast(
    place: Optional[str] = None,
    latlon: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get comprehensive weather forecast including temperature, wind, precipitation, etc.
    
    Args:
        place: Place name
        latlon: Latitude,longitude
        start_time: Start time in ISO format
        end_time: End time in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with comprehensive forecast data
    """
    parameters = [
        FORECAST_PARAMETERS['temperature'],
        FORECAST_PARAMETERS['humidity'],
        FORECAST_PARAMETERS['wind_speed'],
        FORECAST_PARAMETERS['wind_direction'],
        FORECAST_PARAMETERS['precipitation_1h'],
        FORECAST_PARAMETERS['pressure'],
        FORECAST_PARAMETERS['cloud_cover']
    ]
    
    return get_forecast(
        place=place,
        latlon=latlon,
        start_time=start_time,
        end_time=end_time,
        parameters=parameters,
        timestep=timestep
    )


def pivot_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot forecast DataFrame so each parameter becomes a column.
    
    Args:
        df: DataFrame from get_forecast
        
    Returns:
        Pivoted DataFrame with parameters as columns
    """
    if df.empty:
        return df
    
    # Determine time column name
    time_col = 'forecast_time' if 'forecast_time' in df.columns else 'timestamp'
    
    index_cols = ['station_id', 'station_name', 'latitude', 'longitude', time_col]
    # Only include columns that exist
    index_cols = [col for col in index_cols if col in df.columns]
    
    if not index_cols:
        return df
    
    pivoted = df.pivot_table(
        index=index_cols,
        columns='parameter',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return pivoted


