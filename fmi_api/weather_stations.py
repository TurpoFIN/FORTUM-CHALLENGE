"""
Functions for finding and managing FMI weather stations
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt

from .fmi_client import FMIClient
from .observations import NAMESPACES


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    
    return c * r


def get_all_stations() -> pd.DataFrame:
    """
    Get information about all FMI weather stations.
    
    Returns:
        DataFrame with station information
        
    Note: This queries for a recent time period to get active stations.
    """
    from datetime import datetime, timedelta, timezone
    
    # Query recent observations to get list of stations
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=1)
    
    with FMIClient() as client:
        # Query all stations by using a wide bounding box covering Finland
        # Finland roughly: 59.5°N to 70.1°N, 19.5°E to 31.6°E
        root = client.get_observations(
            bbox='19.5,59.5,31.6,70.1',
            start_time=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            end_time=end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            parameters=['t2m'],  # Just get temperature to find stations
            timestep=60
        )
        
        stations = []
        seen_ids = set()
        
        # Parse station information
        for member in root.findall('.//wfs:member', NAMESPACES):
            # Get station ID
            station_elem = member.find('.//gml:identifier', NAMESPACES)
            if station_elem is not None:
                station_id = station_elem.text
                
                # Skip if we've already seen this station
                if station_id in seen_ids:
                    continue
                seen_ids.add(station_id)
                
                # Get station name
                name_elem = member.find('.//gml:name', NAMESPACES)
                station_name = name_elem.text if name_elem is not None else None
                
                # Get position (lat, lon)
                pos_elem = member.find('.//gml:pos', NAMESPACES)
                if pos_elem is not None:
                    pos_parts = pos_elem.text.split()
                    latitude = float(pos_parts[0]) if len(pos_parts) > 0 else None
                    longitude = float(pos_parts[1]) if len(pos_parts) > 1 else None
                else:
                    latitude, longitude = None, None
                
                stations.append({
                    'fmisid': station_id,
                    'name': station_name,
                    'latitude': latitude,
                    'longitude': longitude
                })
        
        df = pd.DataFrame(stations)
        
        if not df.empty:
            df = df.sort_values('name')
            df = df.reset_index(drop=True)
        
        return df


def get_nearby_stations(
    latitude: float,
    longitude: float,
    max_distance_km: float = 100,
    max_count: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Find weather stations near a given location.
    
    Args:
        latitude: Target latitude
        longitude: Target longitude
        max_distance_km: Maximum distance in kilometers (default: 100)
        max_count: Maximum number of stations to return (default: all within distance)
        
    Returns:
        List of station dictionaries sorted by distance
        
    Example:
        >>> stations = get_nearby_stations(60.17, 24.94, max_count=5)
        >>> for station in stations:
        ...     print(f"{station['name']}: {station['distance_km']:.1f} km")
    """
    # Get all stations
    all_stations = get_all_stations()
    
    if all_stations.empty:
        return []
    
    # Calculate distances
    distances = []
    for _, station in all_stations.iterrows():
        if pd.notna(station['latitude']) and pd.notna(station['longitude']):
            dist = haversine_distance(
                latitude, longitude,
                station['latitude'], station['longitude']
            )
            distances.append(dist)
        else:
            distances.append(float('inf'))
    
    all_stations['distance_km'] = distances
    
    # Filter by distance
    nearby = all_stations[all_stations['distance_km'] <= max_distance_km]
    
    # Sort by distance
    nearby = nearby.sort_values('distance_km')
    
    # Limit count if specified
    if max_count is not None:
        nearby = nearby.head(max_count)
    
    # Convert to list of dicts
    return nearby.to_dict('records')


def find_station_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Find a station by name (case-insensitive partial match).
    
    Args:
        name: Station name or partial name
        
    Returns:
        Station dictionary or None if not found
        
    Example:
        >>> station = find_station_by_name('Helsinki')
        >>> print(station['fmisid'])
    """
    all_stations = get_all_stations()
    
    if all_stations.empty:
        return None
    
    # Case-insensitive search
    matches = all_stations[
        all_stations['name'].str.contains(name, case=False, na=False)
    ]
    
    if matches.empty:
        return None
    
    # Return first match
    return matches.iloc[0].to_dict()


def find_stations_for_power_plants(
    power_plants_file: str = 'finnish_powerplants.txt',
    max_distance_km: float = 100,
    max_count_per_plant: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find weather stations near each power plant.
    
    Args:
        power_plants_file: Path to power plants file (default: 'finnish_powerplants.txt')
        max_distance_km: Maximum distance in kilometers
        max_count_per_plant: Maximum number of stations per plant
        
    Returns:
        Dictionary mapping plant names to lists of nearby stations
        
    Example:
        >>> stations_by_plant = find_stations_for_power_plants()
        >>> for plant, stations in stations_by_plant.items():
        ...     print(f"{plant}: {len(stations)} stations found")
    """
    import re
    
    # Read power plants file
    with open(power_plants_file, 'r') as f:
        content = f.read()
    
    # Parse power plant locations
    # Format: Plant Name: lat, lon
    pattern = r'([^:]+):\s*([\d.]+),\s*([\d.]+)'
    matches = re.findall(pattern, content)
    
    stations_by_plant = {}
    
    for plant_name, lat_str, lon_str in matches:
        plant_name = plant_name.strip()
        latitude = float(lat_str)
        longitude = float(lon_str)
        
        # Find nearby stations
        stations = get_nearby_stations(
            latitude, longitude,
            max_distance_km=max_distance_km,
            max_count=max_count_per_plant
        )
        
        stations_by_plant[plant_name] = stations
    
    return stations_by_plant


def get_station_info(fmisid: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific station.
    
    Args:
        fmisid: FMI station ID
        
    Returns:
        Station information dictionary
        
    Example:
        >>> info = get_station_info('100971')
        >>> print(info['name'])
    """
    all_stations = get_all_stations()
    
    if all_stations.empty:
        return None
    
    matches = all_stations[all_stations['fmisid'] == fmisid]
    
    if matches.empty:
        return None
    
    return matches.iloc[0].to_dict()


