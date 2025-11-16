"""
SMHI API Client
Base client for interacting with SMHI's Open Data API
"""

import requests
import time
from typing import Dict, List, Optional
from datetime import datetime
import json


class SMHIClient:
    """Client for SMHI Open Data API"""
    
    BASE_URL = "https://opendata-download-metobs.smhi.se/api"
    
    # SMHI Parameter codes for meteorological observations
    PARAMETERS = {
        'temperature': 1,        # Air temperature (°C)
        'humidity': 6,           # Relative humidity (%)
        'wind_speed': 4,         # Wind speed (m/s)
        'precipitation': 7,      # Precipitation intensity (mm/h)
        'precipitation_amount': 5,  # Precipitation amount (mm)
    }
    
    def __init__(self, retry_attempts=3, retry_delay=1, debug: bool = False):
        """
        Initialize SMHI API client
        
        Args:
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay in seconds between retry attempts
            debug: If True, prints request URLs and response status codes
        """
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.debug = debug
        self.session = requests.Session()
        self._station_details_cache: Dict[str, Dict] = {}
    
    def _log(self, message: str) -> None:
        if self.debug:
            print(message)
        
    def get_stations(self, parameter: str) -> List[Dict]:
        """
        Get all available stations for a specific parameter
        
        Args:
            parameter: Parameter name (e.g., 'temperature', 'wind_speed')
            
        Returns:
            List of station dictionaries with metadata
        """
        param_code = self.PARAMETERS.get(parameter)
        if not param_code:
            raise ValueError(f"Unknown parameter: {parameter}. Available: {list(self.PARAMETERS.keys())}")
        
        url = f"{self.BASE_URL}/version/1.0/parameter/{param_code}.json"
        
        for attempt in range(self.retry_attempts):
            try:
                self._log(f"[SMHI] GET {url}")
                response = self.session.get(url, timeout=30)
                self._log(f"[SMHI] Response {response.status_code} for stations ({parameter})")
                if response.status_code == 404:
                    # Do not retry on 404
                    raise requests.exceptions.HTTPError(f"404 Client Error: Not Found for url: {url}")
                response.raise_for_status()
                data = response.json()
                stations = data.get('station', [])
                # Normalize station id field: SMHI uses 'key' as station identifier
                for s in stations:
                    if 'id' not in s and 'key' in s:
                        s['id'] = s['key']
                return stations
            except requests.exceptions.RequestException as e:
                self._log(f"[SMHI] Request error fetching stations for {parameter}: {e}")
                # Do not retry on 404
                if '404' in str(e):
                    raise
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"Failed to fetch stations for {parameter}: {str(e)}")
    
    def find_nearest_station(self, parameter: str, latitude: float, longitude: float, 
                            max_stations: int = 1, required_period: Optional[str] = None) -> Optional[Dict]:
        """
        Find the nearest station(s) to given coordinates
        
        Args:
            parameter: Parameter name
            latitude: Target latitude
            longitude: Target longitude
            max_stations: Maximum number of nearest stations to return (default: 1)
            required_period: If provided, only consider stations that support this period
            
        Returns:
            Station dictionary (or list of stations if max_stations > 1) with metadata
        """
        stations = self.get_stations(parameter)
        
        if not stations:
            return None
        
        # Optionally filter by supported period if metadata available
        if required_period:
            filtered = []
            for station in stations:
                periods = station.get('period') or station.get('periods') or []
                has_required = False
                # Periods can be a list of dicts with keys like {'key': 'corrected-archive', ...}
                for p in periods if isinstance(periods, list) else []:
                    if isinstance(p, dict):
                        key = p.get('key') or p.get('name')
                        if key == required_period:
                            has_required = True
                            break
                    elif isinstance(p, str) and p == required_period:
                        has_required = True
                        break
                # Only keep stations that explicitly advertise the required period
                if has_required:
                    filtered.append(station)
            stations = filtered
        
        # Calculate distances for all active (and period-compatible) stations
        station_distances = []
        
        for station in stations:
            # For corrected-archive, allow inactive stations (archive still exists)
            if required_period != 'corrected-archive':
                if not station.get('active'):
                    continue
                
            lat = station.get('latitude')
            lon = station.get('longitude')
            
            if lat is None or lon is None:
                continue
            
            # Simple Euclidean distance (good enough for small distances)
            distance = ((lat - latitude)**2 + (lon - longitude)**2)**0.5
            station_distances.append((distance, station))
        
        if not station_distances:
            return None
        
        # Sort by distance
        station_distances.sort(key=lambda x: x[0])
        
        if max_stations == 1:
            return station_distances[0][1]
        else:
            return [s[1] for s in station_distances[:max_stations]]
    
    def find_nearest_stations(self, parameter: str, latitude: float, longitude: float,
                             max_stations: int = 5, required_period: Optional[str] = None) -> List[Dict]:
        """
        Find multiple nearest stations to given coordinates
        
        Args:
            parameter: Parameter name
            latitude: Target latitude
            longitude: Target longitude
            max_stations: Maximum number of stations to return
            required_period: If provided, only consider stations that support this period
            
        Returns:
            List of station dictionaries, sorted by distance
        """
        result = self.find_nearest_station(
            parameter, latitude, longitude, max_stations, required_period
        )
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]
    
    def get_station_details(self, parameter: str, station_id: int) -> Dict:
        """
        Fetch detailed station metadata for a given parameter/station.
        Used to verify period availability to avoid 404s.
        """
        cache_key = f"{parameter}:{station_id}"
        if cache_key in self._station_details_cache:
            return self._station_details_cache[cache_key]
        
        param_code = self.PARAMETERS.get(parameter)
        if not param_code:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        url = f"{self.BASE_URL}/version/1.0/parameter/{param_code}/station/{station_id}.json"
        for attempt in range(self.retry_attempts):
            try:
                self._log(f"[SMHI] GET {url}")
                response = self.session.get(url, timeout=30)
                self._log(f"[SMHI] Response {response.status_code} for station details {station_id} ({parameter})")
                if response.status_code == 404:
                    # Do not retry on 404
                    raise requests.exceptions.HTTPError(f"404 Client Error: Not Found for url: {url}")
                response.raise_for_status()
                data = response.json()
                self._station_details_cache[cache_key] = data
                return data
            except requests.exceptions.RequestException as e:
                self._log(f"[SMHI] Request error fetching station details {station_id}: {e}")
                # Do not retry on 404
                if '404' in str(e):
                    raise
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise
    
    def station_supports_period(self, parameter: str, station_id: int, period: str) -> bool:
        """
        Check if a station supports a given period for a parameter by inspecting station metadata.
        """
        # Prefer the explicit periods listing endpoint if available
        param_code = self.PARAMETERS.get(parameter)
        if not param_code:
            return False
        
        period_list_urls = [
            f"{self.BASE_URL}/version/1.0/parameter/{param_code}/station/{station_id}/period.json",
            f"{self.BASE_URL}/version/1.0/parameter/{param_code}/station/{station_id}/periods.json"
        ]
        
        for url in period_list_urls:
            try:
                self._log(f"[SMHI] GET {url}")
                resp = self.session.get(url, timeout=30)
                self._log(f"[SMHI] Response {resp.status_code} for station periods {station_id} ({parameter})")
                if resp.status_code != 200:
                    continue
                data = resp.json()
                periods = data.get('period') or data.get('periods') or data
                # Normalize into iterable
                if isinstance(periods, dict):
                    periods_iter = [periods]
                elif isinstance(periods, list):
                    periods_iter = periods
                else:
                    periods_iter = []
                
                for p in periods_iter:
                    if isinstance(p, dict):
                        key = p.get('key') or p.get('name') or p.get('title')
                        if self.debug:
                            self._log(f"[SMHI] Station {station_id} period key: {key}")
                        if key == period:
                            self._log(f"[SMHI] Station {station_id} SUPPORTS period '{period}'")
                            return True
                    elif isinstance(p, str) and p == period:
                        if self.debug:
                            self._log(f"[SMHI] Station {station_id} SUPPORTS period '{period}' (string)")
                        return True
            except requests.exceptions.RequestException as e:
                self._log(f"[SMHI] Error fetching periods for station {station_id}: {e}")
                continue
            except ValueError:
                continue
        
        # Fallback: inspect station details
        try:
            details = self.get_station_details(parameter, station_id)
            periods = details.get('period') or details.get('periods') or []
            if isinstance(periods, list):
                for p in periods:
                    if isinstance(p, dict):
                        key = p.get('key') or p.get('name')
                        if self.debug:
                            self._log(f"[SMHI] Station {station_id} (details) period key: {key}")
                        if key == period:
                            self._log(f"[SMHI] Station {station_id} SUPPORTS period '{period}' (details)")
                            return True
                    elif isinstance(p, str) and p == period:
                        if self.debug:
                            self._log(f"[SMHI] Station {station_id} SUPPORTS period '{period}' (details string)")
                        return True
            elif isinstance(periods, dict):
                key = periods.get('key') or periods.get('name')
                if self.debug:
                    self._log(f"[SMHI] Station {station_id} (details) single period key: {key}")
                return key == period
        except Exception:
            return False
        
        return False
    
    def get_station_data(self, parameter: str, station_id: int, 
                        period: str = 'latest-months',
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict]:
        """
        Get observation data from a specific station
        
        Args:
            parameter: Parameter name
            station_id: Station ID
            period: Data period ('latest-hour', 'latest-day', 'latest-months', 'corrected-archive')
            start_date: Optional start date (datetime) to limit results
            end_date: Optional end date (datetime) to limit results
            
        Returns:
            List of observation dictionaries
        """
        param_code = self.PARAMETERS.get(parameter)
        if not param_code:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        # Archive period only provides CSV data; JSON endpoint returns 404.
        is_archive = period == 'corrected-archive'
        data_ext = 'csv' if is_archive else 'json'
        url = f"{self.BASE_URL}/version/1.0/parameter/{param_code}/station/{station_id}/period/{period}/data.{data_ext}"
        # Append date constraints if provided (supported by CSV and JSON)
        if start_date or end_date:
            params = []
            if start_date:
                params.append(f"from={start_date.date().isoformat()}")
            if end_date:
                params.append(f"to={end_date.date().isoformat()}")
            if params:
                url = f"{url}?{'&'.join(params)}"
        
        for attempt in range(self.retry_attempts):
            try:
                self._log(f"[SMHI] GET {url}")
                response = self.session.get(url, timeout=30)
                self._log(f"[SMHI] Response {response.status_code} for station {station_id} ({parameter}, {period})")
                if response.status_code == 404:
                    # Do not retry on 404
                    raise requests.exceptions.HTTPError(f"404 Client Error: Not Found for url: {url}")
                response.raise_for_status()
                if is_archive:
                    # Parse CSV into observations matching JSON shape
                    text = response.text
                    observations: List[Dict] = []
                    for line in text.splitlines():
                        # Expect lines like: YYYY-MM-DD;HH:MM:SS;value;quality
                        parts = line.strip().split(';')
                        if len(parts) != 4:
                            continue
                        date_part, time_part, val_str, quality = parts
                        try:
                            dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
                            value = float(val_str.replace(',', '.'))
                        except Exception:
                            continue
                        observations.append({
                            'date': int(dt.timestamp() * 1000),
                            'value': value,
                            'quality': quality
                        })
                    return observations
                else:
                    data = response.json()
                    return data.get('value', [])
            except requests.exceptions.RequestException as e:
                self._log(f"[SMHI] Request error for station {station_id} ({parameter}, {period}): {e}")
                # Do not retry on 404
                if '404' in str(e):
                    raise
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise Exception(f"Failed to fetch data for station {station_id}: {str(e)}")
    
    def get_data_by_location(self, parameter: str, latitude: float, longitude: float,
                           period: str = 'latest-months', try_multiple_stations: bool = True) -> Dict:
        """
        Get observation data for the nearest station to specified coordinates
        
        Args:
            parameter: Parameter name
            latitude: Target latitude
            longitude: Target longitude
            period: Data period
            try_multiple_stations: If True, try up to 3 nearest stations if first fails
            
        Returns:
            Dictionary containing station info and observations
        """
        if try_multiple_stations:
            # Get top 3 nearest stations as fallbacks
            stations = self.find_nearest_stations(parameter, latitude, longitude, max_stations=3)
        else:
            station = self.find_nearest_station(parameter, latitude, longitude)
            stations = [station] if station else []
        
        if not stations:
            raise Exception(f"No active station found for parameter: {parameter}")
        
        last_error = None
        
        for station in stations:
            try:
                station_id = station['id']
                observations = self.get_station_data(parameter, station_id, period)
                
                # If we got data, return it
                if observations:
                    return {
                        'station': station,
                        'observations': observations,
                        'parameter': parameter,
                        'parameter_code': self.PARAMETERS[parameter]
                    }
            except Exception as e:
                last_error = e
                # Try next station
                continue
        
        # If we tried all stations and none worked, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception(f"No data available from any nearby station for parameter: {parameter}")
    
    def parse_observations(self, observations: List[Dict]) -> List[Dict]:
        """
        Parse observations into a more readable format
        
        Args:
            observations: Raw observation data from API
            
        Returns:
            List of parsed observations with timestamp and value
        """
        parsed = []
        
        for obs in observations:
            try:
                # SMHI timestamps are in milliseconds since epoch
                timestamp_ms = obs['date']
                
                # Convert from milliseconds to seconds for datetime
                timestamp = datetime.utcfromtimestamp(timestamp_ms / 1000)
                
                value = float(obs['value'])
                quality = obs.get('quality', 'unknown')
                
                parsed.append({
                    'timestamp': timestamp,
                    'value': value,
                    'quality': quality,
                    'date_str': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            except (KeyError, ValueError, TypeError) as e:
                # Skip malformed observations
                continue
        
        return parsed


