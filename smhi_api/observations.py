"""
High-level functions for fetching specific meteorological observations
"""

import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
from .api_client import SMHIClient


def get_temperature_data(latitude: float, longitude: float, 
                        period: str = 'latest-months',
                        return_format: str = 'dataframe') -> pd.DataFrame:
    """
    Get air temperature observations for a location
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        period: Data period ('latest-hour', 'latest-day', 'latest-months', 'corrected-archive')
        return_format: 'dataframe' or 'dict'
        
    Returns:
        DataFrame with timestamp and temperature in Celsius
        
    Example:
        >>> df = get_temperature_data(65.92, 21.05)  # Piteå, Sweden
        >>> print(df.head())
    """
    client = SMHIClient()
    
    data = client.get_data_by_location('temperature', latitude, longitude, period)
    observations = client.parse_observations(data['observations'])
    
    if not observations:
        raise Exception(f"No temperature data available for location ({latitude}, {longitude})")
    
    # Create DataFrame
    df = pd.DataFrame(observations)
    df = df[['timestamp', 'value', 'quality']]
    df.columns = ['timestamp', 'temperature_celsius', 'quality']
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Add station metadata
    station = data['station']
    metadata = {
        'station_name': station.get('name'),
        'station_id': station.get('id'),
        'station_lat': station.get('latitude'),
        'station_lon': station.get('longitude'),
        'parameter': 'Air Temperature',
        'unit': '°C'
    }
    
    if return_format == 'dataframe':
        df.attrs['metadata'] = metadata
        return df
    else:
        return {
            'data': df.to_dict('records'),
            'metadata': metadata
        }


def get_wind_speed_data(latitude: float, longitude: float,
                       period: str = 'latest-months',
                       return_format: str = 'dataframe') -> pd.DataFrame:
    """
    Get wind speed observations for a location
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        period: Data period
        return_format: 'dataframe' or 'dict'
        
    Returns:
        DataFrame with timestamp and wind speed in m/s
        
    Example:
        >>> df = get_wind_speed_data(65.92, 21.05)  # Markbygden wind farm area
        >>> print(df.head())
    """
    client = SMHIClient()
    
    data = client.get_data_by_location('wind_speed', latitude, longitude, period)
    observations = client.parse_observations(data['observations'])
    
    if not observations:
        raise Exception(f"No wind speed data available for location ({latitude}, {longitude})")
    
    df = pd.DataFrame(observations)
    df = df[['timestamp', 'value', 'quality']]
    df.columns = ['timestamp', 'wind_speed_ms', 'quality']
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    station = data['station']
    metadata = {
        'station_name': station.get('name'),
        'station_id': station.get('id'),
        'station_lat': station.get('latitude'),
        'station_lon': station.get('longitude'),
        'parameter': 'Wind Speed',
        'unit': 'm/s'
    }
    
    if return_format == 'dataframe':
        df.attrs['metadata'] = metadata
        return df
    else:
        return {
            'data': df.to_dict('records'),
            'metadata': metadata
        }


def get_humidity_data(latitude: float, longitude: float,
                     period: str = 'latest-months',
                     return_format: str = 'dataframe') -> pd.DataFrame:
    """
    Get relative humidity observations for a location
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        period: Data period
        return_format: 'dataframe' or 'dict'
        
    Returns:
        DataFrame with timestamp and relative humidity in %
        
    Example:
        >>> df = get_humidity_data(57.48, 15.84)  # Hultsfreds, Sweden
        >>> print(df.head())
    """
    client = SMHIClient()
    
    data = client.get_data_by_location('humidity', latitude, longitude, period)
    observations = client.parse_observations(data['observations'])
    
    if not observations:
        raise Exception(f"No humidity data available for location ({latitude}, {longitude})")
    
    df = pd.DataFrame(observations)
    df = df[['timestamp', 'value', 'quality']]
    df.columns = ['timestamp', 'relative_humidity_percent', 'quality']
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    station = data['station']
    metadata = {
        'station_name': station.get('name'),
        'station_id': station.get('id'),
        'station_lat': station.get('latitude'),
        'station_lon': station.get('longitude'),
        'parameter': 'Relative Humidity',
        'unit': '%'
    }
    
    if return_format == 'dataframe':
        df.attrs['metadata'] = metadata
        return df
    else:
        return {
            'data': df.to_dict('records'),
            'metadata': metadata
        }


def get_precipitation_data(latitude: float, longitude: float,
                          period: str = 'latest-months',
                          return_format: str = 'dataframe',
                          precipitation_type: str = 'intensity') -> pd.DataFrame:
    """
    Get precipitation observations for a location
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        period: Data period
        return_format: 'dataframe' or 'dict'
        precipitation_type: 'intensity' (mm/h) or 'amount' (mm)
        
    Returns:
        DataFrame with timestamp and precipitation
        
    Example:
        >>> df = get_precipitation_data(66.60, 20.40)  # Jokkmokk, Sweden
        >>> print(df.head())
    """
    client = SMHIClient()
    
    param_name = 'precipitation' if precipitation_type == 'intensity' else 'precipitation_amount'
    unit = 'mm/h' if precipitation_type == 'intensity' else 'mm'
    
    data = client.get_data_by_location(param_name, latitude, longitude, period)
    observations = client.parse_observations(data['observations'])
    
    if not observations:
        raise Exception(f"No precipitation data available for location ({latitude}, {longitude})")
    
    df = pd.DataFrame(observations)
    df = df[['timestamp', 'value', 'quality']]
    col_name = 'precipitation_intensity_mmh' if precipitation_type == 'intensity' else 'precipitation_amount_mm'
    df.columns = ['timestamp', col_name, 'quality']
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    station = data['station']
    metadata = {
        'station_name': station.get('name'),
        'station_id': station.get('id'),
        'station_lat': station.get('latitude'),
        'station_lon': station.get('longitude'),
        'parameter': f'Precipitation ({precipitation_type.capitalize()})',
        'unit': unit
    }
    
    if return_format == 'dataframe':
        df.attrs['metadata'] = metadata
        return df
    else:
        return {
            'data': df.to_dict('records'),
            'metadata': metadata
        }


def get_all_weather_data(latitude: float, longitude: float,
                        period: str = 'latest-months') -> Dict[str, pd.DataFrame]:
    """
    Get all four weather parameters for a location
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        period: Data period
        
    Returns:
        Dictionary with DataFrames for temperature, wind_speed, humidity, and precipitation
        
    Example:
        >>> weather = get_all_weather_data(65.92, 21.05)
        >>> print(weather['temperature'].head())
        >>> print(weather['wind_speed'].head())
    """
    results = {}
    
    try:
        results['temperature'] = get_temperature_data(latitude, longitude, period)
    except Exception as e:
        print(f"Warning: Could not fetch temperature data: {e}")
        results['temperature'] = None
    
    try:
        results['wind_speed'] = get_wind_speed_data(latitude, longitude, period)
    except Exception as e:
        print(f"Warning: Could not fetch wind speed data: {e}")
        results['wind_speed'] = None
    
    try:
        results['humidity'] = get_humidity_data(latitude, longitude, period)
    except Exception as e:
        print(f"Warning: Could not fetch humidity data: {e}")
        results['humidity'] = None
    
    try:
        results['precipitation'] = get_precipitation_data(latitude, longitude, period)
    except Exception as e:
        print(f"Warning: Could not fetch precipitation data: {e}")
        results['precipitation'] = None
    
    return results


