"""
Functions for retrieving weather observations from FMI Open Data
"""

from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

from .fmi_client import FMIClient
from .config import WEATHER_PARAMETERS


# XML namespaces used by FMI
NAMESPACES = {
    'wfs': 'http://www.opengis.net/wfs/2.0',
    'gml': 'http://www.opengis.net/gml/3.2',
    'BsWfs': 'http://xml.fmi.fi/schema/wfs/2.0',
    'om': 'http://www.opengis.net/om/2.0',
    'omso': 'http://inspire.ec.europa.eu/schemas/omso/3.0',
    'ompr': 'http://inspire.ec.europa.eu/schemas/ompr/3.0',
    'gmd': 'http://www.isotc211.org/2005/gmd',
    'gco': 'http://www.isotc211.org/2005/gco',
    'swe': 'http://www.opengis.net/swe/2.0',
    'gmlcov': 'http://www.opengis.net/gmlcov/1.0',
    'sam': 'http://www.opengis.net/sampling/2.0',
    'sams': 'http://www.opengis.net/samplingSpatial/2.0',
    'target': 'http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1',
    'wml2': 'http://www.opengis.net/waterml/2.0'
}


def _parse_time_value_pairs(root: ET.Element) -> pd.DataFrame:
    """
    Parse time-value pairs from FMI XML response (O&M/WaterML format).
    
    Args:
        root: XML root element
        
    Returns:
        DataFrame with parsed observations
    """
    records = []
    
    # Find all members (each member is one time series for one parameter at one location)
    for member in root.findall('.//wfs:member', NAMESPACES):
        # Get location/station info
        # Try to find sampling point
        sampling_point = member.find('.//sams:SF_SpatialSamplingFeature', NAMESPACES)
        
        station_id = None
        station_name = None
        latitude = None
        longitude = None
        
        if sampling_point is not None:
            # Get station identifier
            id_elem = sampling_point.find('.//sam:identifier', NAMESPACES)
            if id_elem is None:
                id_elem = sampling_point.find('.//gml:identifier', NAMESPACES)
            if id_elem is not None:
                station_id = id_elem.text
            
            # Get station name  
            name_elem = sampling_point.find('.//sam:name', NAMESPACES)
            if name_elem is None:
                name_elem = sampling_point.find('.//gml:name', NAMESPACES)
            if name_elem is not None:
                station_name = name_elem.text
            
            # Get position
            pos_elem = sampling_point.find('.//gml:pos', NAMESPACES)
            if pos_elem is not None:
                pos_parts = pos_elem.text.split()
                if len(pos_parts) >= 2:
                    latitude = float(pos_parts[0])
                    longitude = float(pos_parts[1])
        
        # Get parameter name from observedProperty
        param_elem = member.find('.//om:observedProperty', NAMESPACES)
        parameter = None
        if param_elem is not None:
            param_href = param_elem.get('{http://www.w3.org/1999/xlink}href', '')
            # Extract parameter name from URL like ".../param=Temperature&..."
            if 'param=' in param_href:
                # Extract param=X from URL
                for part in param_href.split('&'):
                    if part.startswith('param='):
                        parameter = part.split('=')[1]
                        break
            # Fallback: extract from end of URL path
            if not parameter and '/' in param_href:
                parameter = param_href.split('/')[-1]
        
        # Find the measurement time series (WaterML format)
        timeseries = member.find('.//wml2:MeasurementTimeseries', NAMESPACES)
        
        if timeseries is not None:
            # Get all time-value points
            points = timeseries.findall('.//wml2:point', NAMESPACES)
            
            for point in points:
                # Get time
                time_elem = point.find('.//wml2:time', NAMESPACES)
                # Get value
                value_elem = point.find('.//wml2:value', NAMESPACES)
                
                if time_elem is not None and value_elem is not None:
                    try:
                        # Parse ISO format time
                        timestamp = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                        # Parse value
                        value_text = value_elem.text
                        value = float(value_text) if value_text and value_text.lower() != 'nan' else None
                        
                        records.append({
                            'station_id': station_id,
                            'station_name': station_name,
                            'latitude': latitude,
                            'longitude': longitude,
                            'timestamp': timestamp,
                            'parameter': parameter,
                            'value': value
                        })
                    except (ValueError, AttributeError) as e:
                        # Skip invalid entries
                        continue
    
    df = pd.DataFrame(records)
    
    if not df.empty:
        # Convert timezone-aware timestamps to naive UTC
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values(['timestamp', 'station_id', 'parameter'])
        df = df.reset_index(drop=True)
    
    return df


def get_observations(
    place: Optional[str] = None,
    fmisid: Optional[str] = None,
    bbox: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get weather observations from FMI.
    
    Args:
        place: Place name (e.g., 'Helsinki', 'Tampere')
        fmisid: Station ID(s), comma-separated (e.g., '100971')
        bbox: Bounding box as 'minLon,minLat,maxLon,maxLat'
        start_date: Start date in ISO format (e.g., '2023-01-01T00:00:00Z')
        end_date: End date in ISO format
        parameters: List of parameters to retrieve (use config.WEATHER_PARAMETERS)
        timestep: Time step in minutes (default: 60)
        
    Returns:
        DataFrame with observations
        
    Example:
        >>> df = get_observations(
        ...     place='Helsinki',
        ...     start_date='2023-01-01T00:00:00Z',
        ...     end_date='2023-01-31T23:59:59Z',
        ...     parameters=['t2m', 'ws_10min', 'r_1h']
        ... )
    """
    # Default to last 24 hours if no dates specified
    if start_date is None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=1)
        start_date = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    with FMIClient() as client:
        root = client.get_observations(
            place=place,
            fmisid=fmisid,
            bbox=bbox,
            start_time=start_date,
            end_time=end_date,
            parameters=parameters,
            timestep=timestep
        )
        
        return _parse_time_value_pairs(root)


def get_temperature_observations(
    place: Optional[str] = None,
    fmisid: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get temperature observations.
    
    Args:
        place: Place name
        fmisid: Station ID(s)
        start_date: Start date in ISO format
        end_date: End date in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with temperature data in Celsius
    """
    return get_observations(
        place=place,
        fmisid=fmisid,
        start_date=start_date,
        end_date=end_date,
        parameters=[WEATHER_PARAMETERS['temperature']],
        timestep=timestep
    )


def get_wind_observations(
    place: Optional[str] = None,
    fmisid: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get wind observations (speed, direction, gust).
    
    Args:
        place: Place name
        fmisid: Station ID(s)
        start_date: Start date in ISO format
        end_date: End date in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with wind data
    """
    return get_observations(
        place=place,
        fmisid=fmisid,
        start_date=start_date,
        end_date=end_date,
        parameters=[
            WEATHER_PARAMETERS['wind_speed'],
            WEATHER_PARAMETERS['wind_direction'],
            WEATHER_PARAMETERS['wind_gust']
        ],
        timestep=timestep
    )


def get_precipitation_observations(
    place: Optional[str] = None,
    fmisid: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get precipitation observations.
    
    Args:
        place: Place name
        fmisid: Station ID(s)
        start_date: Start date in ISO format
        end_date: End date in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with precipitation data in mm
    """
    return get_observations(
        place=place,
        fmisid=fmisid,
        start_date=start_date,
        end_date=end_date,
        parameters=[WEATHER_PARAMETERS['precipitation']],
        timestep=timestep
    )


def get_comprehensive_weather_observations(
    place: Optional[str] = None,
    fmisid: Optional[str] = None,
    bbox: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timestep: int = 60
) -> pd.DataFrame:
    """
    Get comprehensive weather observations including temperature, wind, precipitation, etc.
    
    Args:
        place: Place name
        fmisid: Station ID(s)
        bbox: Bounding box
        start_date: Start date in ISO format
        end_date: End date in ISO format
        timestep: Time step in minutes
        
    Returns:
        DataFrame with comprehensive weather data
    """
    parameters = [
        WEATHER_PARAMETERS['temperature'],
        WEATHER_PARAMETERS['humidity'],
        WEATHER_PARAMETERS['wind_speed'],
        WEATHER_PARAMETERS['wind_direction'],
        WEATHER_PARAMETERS['precipitation'],
        WEATHER_PARAMETERS['pressure'],
        WEATHER_PARAMETERS['snow_depth']
    ]
    
    return get_observations(
        place=place,
        fmisid=fmisid,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
        timestep=timestep
    )


def pivot_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot observations DataFrame so each parameter becomes a column.
    
    Args:
        df: DataFrame from get_observations
        
    Returns:
        Pivoted DataFrame with parameters as columns
    """
    if df.empty:
        return df
    
    pivoted = df.pivot_table(
        index=['station_id', 'station_name', 'latitude', 'longitude', 'timestamp'],
        columns='parameter',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    return pivoted


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly observations to daily averages.
    
    Args:
        df: DataFrame with observations
        
    Returns:
        DataFrame with daily aggregated data
    """
    if df.empty or 'timestamp' not in df.columns:
        return df
    
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    
    # Group by date, station, and parameter
    daily = df.groupby(['station_id', 'station_name', 'date', 'parameter']).agg({
        'value': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    return daily


