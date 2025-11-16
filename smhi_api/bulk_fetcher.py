"""
Bulk data fetcher for multiple power plants
Fetches historical observation data for power plants from JSON file
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

from .api_client import SMHIClient


class PowerPlantWeatherFetcher:
    """Fetch weather data for multiple power plants"""
    
    def __init__(
        self,
        json_file_path: str,
        *,
        debug: bool = False,
        retry_attempts: int = 1,
        max_general_nearby: int = 20,
        max_period_nearby: int = 8,
        validate_from_nearby: int = 10,
        max_expand_nearby: int = 40,
        max_validated_expand: int = 8
    ):
        """
        Initialize the fetcher with a JSON file of power plants
        
        Args:
            json_file_path: Path to the renewable_powerplants_sweden.json file
            debug: Enable verbose SMHI logging (URLs/status codes)
            retry_attempts: Number of retries for SMHI requests (1 recommended)
            max_general_nearby: Initial pool size of nearest stations
            max_period_nearby: Max stations that explicitly advertise the period
            validate_from_nearby: How many nearest to validate via metadata when needed
            max_expand_nearby: Expanded pool size if nothing found
            max_validated_expand: Max validated stations from the expanded pool
        """
        self.json_file_path = json_file_path
        # Configurable SMHI client
        self.client = SMHIClient(debug=debug, retry_attempts=retry_attempts)
        self.power_plants = self._load_power_plants()
        # Station search bounds
        self.max_general_nearby = max_general_nearby
        self.max_period_nearby = max_period_nearby
        self.validate_from_nearby = validate_from_nearby
        self.max_expand_nearby = max_expand_nearby
        self.max_validated_expand = max_validated_expand
        
    def _load_power_plants(self) -> List[Dict]:
        """Load power plants from JSON file"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            plants = json.load(f)
        
        # Filter out plants without coordinates
        valid_plants = [
            p for p in plants 
            if p.get('latitude') is not None and p.get('longitude') is not None
        ]
        
        print(f"Loaded {len(plants)} power plants, {len(valid_plants)} have valid coordinates")
        return valid_plants
    
    def _filter_by_date_range(self, df: pd.DataFrame, 
                             start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """
        Filter DataFrame by date range
        
        Args:
            df: DataFrame with 'timestamp' column
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Ensure timestamps are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df[mask].copy()
        
        return filtered_df
    
    def _resample_to_hourly(self, df: pd.DataFrame, 
                           value_column: str) -> pd.DataFrame:
        """
        Resample data to hourly frequency
        
        Args:
            df: DataFrame with timestamp column
            value_column: Name of the value column to resample
            
        Returns:
            Resampled DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Set timestamp as index
        df_copy = df.copy()
        df_copy.set_index('timestamp', inplace=True)
        
        # Resample to hourly, taking mean for multiple readings in same hour
        hourly = df_copy[value_column].resample('H').mean()
        
        # Convert back to DataFrame
        result = pd.DataFrame({
            'timestamp': hourly.index,
            value_column: hourly.values
        })
        
        # Forward fill missing hours (up to 3 hours)
        result[value_column] = result[value_column].ffill(limit=3)
        
        return result.reset_index(drop=True)
    
    def _check_data_coverage(self, observations: List[Dict], 
                           start_date: datetime, 
                           end_date: datetime,
                           min_coverage_pct: float = 60.0) -> Tuple[bool, float, datetime, datetime]:
        """
        Check if observations have sufficient coverage for the requested date range
        
        Args:
            observations: Parsed observations list
            start_date: Requested start date
            end_date: Requested end date
            min_coverage_pct: Minimum percentage of date range that must be covered
            
        Returns:
            Tuple of (has_sufficient_coverage, coverage_pct, data_start, data_end)
        """
        if not observations:
            return False, 0.0, None, None
        
        df = pd.DataFrame(observations)
        data_start = df['timestamp'].min()
        data_end = df['timestamp'].max()
        
        # Check if data covers requested range
        requested_days = (end_date - start_date).days
        
        # Calculate actual overlap
        overlap_start = max(start_date, data_start)
        overlap_end = min(end_date, data_end)
        
        if overlap_start > overlap_end:
            # No overlap
            return False, 0.0, data_start, data_end
        
        overlap_days = (overlap_end - overlap_start).days
        coverage_pct = (overlap_days / requested_days) * 100
        
        has_sufficient = coverage_pct >= min_coverage_pct
        
        return has_sufficient, coverage_pct, data_start, data_end
    
    def fetch_weather_for_plant(self, 
                               plant: Dict,
                               start_date: datetime,
                               end_date: datetime,
                               parameters: List[str] = None,
                               min_coverage_pct: float = 60.0) -> Dict[str, pd.DataFrame]:
        """
        Fetch weather data for a single power plant
        
        Args:
            plant: Power plant dictionary with latitude/longitude
            start_date: Start date for historical data
            end_date: End date for historical data
            parameters: List of parameters to fetch (default: all available)
            min_coverage_pct: Minimum percentage of date range that must be covered (default: 60%)
            
        Returns:
            Dictionary of DataFrames, one per parameter
        """
        if parameters is None:
            # Default parameters based on plant type
            source = plant.get('source', '').lower()
            if 'wind' in source:
                parameters = ['temperature', 'wind_speed', 'precipitation']
            elif 'hydro' in source:
                parameters = ['temperature', 'precipitation', 'humidity']
            elif 'solar' in source or 'sun' in source:
                parameters = ['temperature', 'humidity']
            else:
                parameters = ['temperature', 'wind_speed', 'humidity', 'precipitation']
        
        lat = plant['latitude']
        lon = plant['longitude']
        
        results = {}
        
        for param in parameters:
            try:
                # Determine which periods to try based on requested date range
                from datetime import datetime, timedelta
                now = datetime.now()
                months_ago_4 = now - timedelta(days=120)
                
                # If requesting old data, we MUST have corrected-archive
                if start_date < months_ago_4:
                    periods_to_try = ['corrected-archive']
                else:
                    periods_to_try = ['latest-months', 'corrected-archive']
                
                # Get general nearest stations (for display/reference and fallback)
                general_nearby = self.client.find_nearest_stations(
                    param, lat, lon, max_stations=self.max_general_nearby
                )
                
                if not general_nearby:
                    print(f"  ✗ {param}: No active stations nearby")
                    continue
                
                # Try each station until we find one with sufficient coverage
                data = None
                successful_period = None
                successful_station = None
                best_coverage = 0
                best_data = None
                stations_considered = 0
                
                # Try period-specific nearest stations first to avoid 404s
                for period in periods_to_try:
                    # First try stations that advertise the period in the list response
                    period_nearby = self.client.find_nearest_stations(
                        param, lat, lon, max_stations=self.max_period_nearby, required_period=period
                    )
                    if self.client.debug:
                        print(f"[SMHI] Nearby stations advertising '{period}': {len(period_nearby)}")
                    
                    candidates = period_nearby
                    
                    # If none advertised, validate by fetching station details for nearest stations
                    if not candidates:
                        validated = []
                        # Check the first N nearest by details endpoint
                        for s in general_nearby[:self.validate_from_nearby]:
                            sid = s.get('id')
                            if not sid:
                                continue
                            if self.client.station_supports_period(param, sid, period):
                                validated.append(s)
                            # Stop if we have a reasonable number
                            if len(validated) >= self.max_period_nearby:
                                break
                        candidates = validated
                        if self.client.debug:
                            print(f"[SMHI] Validated by details '{period}': {len(candidates)}")
                    
                    # If still none, expand search radius by considering top 100 nearest stations
                    if not candidates:
                        expanded = self.client.find_nearest_stations(
                            param, lat, lon, max_stations=self.max_expand_nearby
                        )
                        validated = []
                        for s in expanded:
                            sid = s.get('id')
                            if not sid:
                                continue
                            if self.client.station_supports_period(param, sid, period):
                                validated.append(s)
                            if len(validated) >= self.max_validated_expand:
                                break
                        candidates = validated
                        if self.client.debug:
                            print(f"[SMHI] Expanded search '{period}': {len(candidates)}")
                    
                    if not candidates:
                        continue
                    
                    for station in candidates:
                        stations_considered += 1
                        try:
                            # Fetch data from this specific station
                            station_id = station['id']
                            raw_observations = self.client.get_station_data(
                                param, station_id, period, start_date, end_date
                            )
                            
                            if not raw_observations:
                                continue
                            
                            # Parse and check coverage
                            observations = self.client.parse_observations(raw_observations)
                            
                            if not observations:
                                continue
                            
                            # Check if this station has sufficient coverage
                            has_sufficient, coverage_pct, data_start, data_end = self._check_data_coverage(
                                observations, start_date, end_date, min_coverage_pct
                            )
                            
                            # Keep track of best option even if not sufficient
                            if coverage_pct > best_coverage:
                                best_coverage = coverage_pct
                                best_data = {
                                    'station': station,
                                    'observations': raw_observations,
                                    'parameter': param,
                                    'parameter_code': self.client.PARAMETERS[param],
                                    'period': period,
                                    'coverage_pct': coverage_pct,
                                    'data_start': data_start,
                                    'data_end': data_end
                                }
                            
                            # If sufficient, use this one
                            if has_sufficient:
                                data = best_data
                                successful_period = period
                                successful_station = station
                                break
                                
                        except Exception as e:
                            # Try next period or station
                            if '404' not in str(e):
                                pass  # Silent fail, try next
                            continue
                    
                    # If we found sufficient coverage, stop trying stations
                    if data and data.get('coverage_pct', 0) >= min_coverage_pct:
                        break
                
                # If no sufficient coverage found, use best available
                if not data and best_data:
                    data = best_data
                    successful_period = best_data['period']
                    successful_station = best_data['station']
                    
                    if best_coverage < min_coverage_pct:
                        tried_note = stations_considered if stations_considered else len(general_nearby)
                        print(f"  ⚠️  {param}: Only {best_coverage:.1f}% coverage (tried {tried_note} stations)")
                        print(f"      Available: {best_data['data_start'].date()} to {best_data['data_end'].date()}")
                
                if data is None:
                    print(f"  ✗ {param}: No data available from any nearby station")
                    continue
                
                # Parse observations from the best/sufficient station
                observations = self.client.parse_observations(data['observations'])
                
                if not observations:
                    print(f"  ⚠️  No {param} data available")
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(observations)
                df = df[['timestamp', 'value', 'quality']]
                
                # Check available date range before filtering
                if not df.empty:
                    available_start = df['timestamp'].min()
                    available_end = df['timestamp'].max()
                    
                    # Filter by date range
                    df_filtered = self._filter_by_date_range(df, start_date, end_date)
                    
                    if df_filtered.empty:
                        print(f"  ⚠️  No {param} data in specified date range")
                        print(f"      Available: {available_start.date()} to {available_end.date()}")
                        print(f"      Requested: {start_date.date()} to {end_date.date()}")
                        continue
                else:
                    print(f"  ⚠️  No {param} data available")
                    continue
                
                # Resample to hourly
                df_hourly = self._resample_to_hourly(df_filtered, 'value')
                
                # Add metadata
                df_hourly.attrs['station'] = data['station']
                df_hourly.attrs['parameter'] = param
                df_hourly.attrs['plant_name'] = plant['name']
                df_hourly.attrs['plant_source'] = plant.get('source')
                df_hourly.attrs['data_period'] = successful_period
                df_hourly.attrs['coverage_pct'] = data.get('coverage_pct', 100.0)
                
                results[param] = df_hourly
                
                # Build status message
                period_note = f" (from {successful_period})" if successful_period != 'corrected-archive' else ""
                station_name = data['station']['name']
                
                # Calculate distance to station
                station_lat = data['station']['latitude']
                station_lon = data['station']['longitude']
                distance_deg = ((station_lat - lat)**2 + (station_lon - lon)**2)**0.5
                distance_km = distance_deg * 111  # Rough conversion
                
                # Show which station if not the nearest
                station_note = ""
                if len(general_nearby) > 0 and data['station']['id'] != general_nearby[0]['id']:
                    station_note = f" from {station_name} (~{distance_km:.0f}km)"
                
                coverage = data.get('coverage_pct', 100.0)
                if coverage < 100:
                    print(f"  ✓ {param}: {len(df_hourly)} hourly records{period_note}{station_note} [{coverage:.0f}% coverage]")
                else:
                    print(f"  ✓ {param}: {len(df_hourly)} hourly records{period_note}{station_note}")
                
            except Exception as e:
                error_msg = str(e)
                # Make error message more concise
                if '404' in error_msg:
                    print(f"  ✗ {param}: No data available at nearest station")
                elif 'No active station found' in error_msg:
                    print(f"  ✗ {param}: No active station nearby")
                else:
                    print(f"  ✗ {param}: {error_msg[:80]}...")
                continue
        
        return results
    
    def _validate_date_range(self, start_date: datetime, end_date: datetime):
        """Check if date range is reasonable and provide warnings"""
        from datetime import datetime
        
        now = datetime.now()
        days_ago_start = (now - start_date).days
        days_ago_end = (now - end_date).days
        
        warnings = []
        
        # Check if dates are very recent
        if days_ago_end < 7:
            warnings.append(
                f"⚠️  WARNING: Requesting data up to {days_ago_end} days ago. "
                f"SMHI data typically has a 3-7 day delay."
            )
        
        if days_ago_start < 30 and days_ago_end < 7:
            warnings.append(
                f"💡 SUGGESTION: For more reliable data, try dates ending at least "
                f"7-10 days ago. Example: end_date=datetime(2024, 9, 30)"
            )
        
        # Check if date range is too short
        date_diff = (end_date - start_date).days
        if date_diff < 7:
            warnings.append(
                f"ℹ️  NOTE: Short date range ({date_diff} days). Some stations may have sparse data."
            )
        
        return warnings
    
    def fetch_all_plants(self,
                        start_date: datetime,
                        end_date: datetime,
                        max_plants: Optional[int] = None,
                        delay_between_requests: float = 0.5,
                        parameters: List[str] = None,
                        min_coverage_pct: float = 60.0) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch weather data for all power plants
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            max_plants: Maximum number of plants to process (None for all)
            delay_between_requests: Delay in seconds between API requests
            parameters: List of parameters to fetch (None for auto-select)
            min_coverage_pct: Minimum percentage of date range coverage required (default: 60%)
            
        Returns:
            Dictionary mapping plant names to their weather data
        """
        plants_to_process = self.power_plants[:max_plants] if max_plants else self.power_plants
        
        print(f"\n{'='*70}")
        print(f"Fetching weather data for {len(plants_to_process)} power plants")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Validate and show warnings
        warnings = self._validate_date_range(start_date, end_date)
        if warnings:
            print()
            for warning in warnings:
                print(warning)
        
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for i, plant in enumerate(plants_to_process, 1):
            plant_name = plant['name']
            source = plant.get('source', 'unknown')
            power = plant.get('power', 'N/A')
            
            print(f"[{i}/{len(plants_to_process)}] {plant_name}")
            print(f"  Type: {source}, Power: {power}")
            print(f"  Location: ({plant['latitude']:.4f}, {plant['longitude']:.4f})")
            
            try:
                weather_data = self.fetch_weather_for_plant(
                    plant, start_date, end_date, parameters, min_coverage_pct
                )
                
                if weather_data:
                    all_results[plant_name] = {
                        'plant_info': plant,
                        'weather_data': weather_data
                    }
                else:
                    print(f"  ⚠️  No weather data available for this plant")
                
            except Exception as e:
                print(f"  ✗ Error processing plant: {str(e)}")
                continue
            
            # Delay between requests to be nice to the API
            if i < len(plants_to_process):
                time.sleep(delay_between_requests)
            
            print()
        
        print(f"{'='*70}")
        print(f"✓ Completed: {len(all_results)}/{len(plants_to_process)} plants processed successfully")
        print(f"{'='*70}\n")
        
        return all_results
    
    def export_to_csv(self, results: Dict, output_dir: str = 'weather_data_output'):
        """
        Export results to CSV files
        
        Args:
            results: Results from fetch_all_plants()
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Exporting data to {output_path.absolute()}...")
        
        for plant_name, data in results.items():
            # Create safe filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' 
                              for c in plant_name)
            safe_name = safe_name.replace(' ', '_')
            
            plant_dir = output_path / safe_name
            plant_dir.mkdir(exist_ok=True)
            
            # Save plant info
            with open(plant_dir / 'plant_info.json', 'w', encoding='utf-8') as f:
                json.dump(data['plant_info'], f, indent=2, ensure_ascii=False)
            
            # Save each parameter's data
            for param, df in data['weather_data'].items():
                csv_path = plant_dir / f'{param}.csv'
                df.to_csv(csv_path, index=False)
                print(f"  ✓ {plant_name} - {param}: {csv_path}")
        
        print(f"\n✓ Export completed!")
    
    def create_combined_dataset(self, results: Dict) -> pd.DataFrame:
        """
        Create a single combined DataFrame with all plants and parameters
        
        Args:
            results: Results from fetch_all_plants()
            
        Returns:
            Combined DataFrame with columns: timestamp, plant_name, parameter, value
        """
        all_data = []
        
        for plant_name, data in results.items():
            plant_info = data['plant_info']
            
            for param, df in data['weather_data'].items():
                df_copy = df.copy()
                df_copy['plant_name'] = plant_name
                df_copy['plant_type'] = plant_info.get('source')
                df_copy['parameter'] = param
                df_copy.rename(columns={'value': 'measurement'}, inplace=True)
                
                all_data.append(df_copy[['timestamp', 'plant_name', 'plant_type', 
                                        'parameter', 'measurement']])
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['plant_name', 'parameter', 'timestamp'])
        
        return combined.reset_index(drop=True)


def fetch_powerplant_weather_data(
    json_file_path: str,
    start_date: datetime,
    end_date: datetime,
    max_plants: Optional[int] = None,
    parameters: Optional[List[str]] = None,
    export_csv: bool = False,
    output_dir: str = 'weather_data_output',
    min_coverage_pct: float = 60.0,
    *,
    debug: bool = False,
    retry_attempts: int = 1,
    max_general_nearby: int = 20,
    max_period_nearby: int = 8,
    validate_from_nearby: int = 10,
    max_expand_nearby: int = 40,
    max_validated_expand: int = 8
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Main function to fetch historical weather data for Swedish power plants
    
    Args:
        json_file_path: Path to renewable_powerplants_sweden.json
        start_date: Start date for historical data (datetime object)
        end_date: End date for historical data (datetime object)
        max_plants: Maximum number of plants to process (None for all)
        parameters: List of weather parameters to fetch 
                   (options: 'temperature', 'wind_speed', 'humidity', 'precipitation')
                   None = auto-select based on plant type
        export_csv: Whether to export results to CSV files
        output_dir: Directory for CSV exports
        min_coverage_pct: Minimum percentage of date range that must be covered (default: 60%)
                         System will try up to 5 nearby stations to find sufficient coverage
        debug: Enable verbose SMHI logging (URLs/status codes)
        retry_attempts: Number of retries for SMHI requests (1 recommended)
        max_general_nearby: Initial pool size of nearest stations
        max_period_nearby: Max stations that explicitly advertise the period
        validate_from_nearby: How many nearest to validate via metadata when needed
        max_expand_nearby: Expanded pool size if nothing found
        max_validated_expand: Max validated stations from the expanded pool
        
    Returns:
        Dictionary mapping plant names to their weather data
        
    Example:
        >>> from datetime import datetime
        >>> results = fetch_powerplant_weather_data(
        ...     'powerplants/sweden powerplants/renewable_powerplants_sweden.json',
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 12, 31),
        ...     max_plants=10,  # Process first 10 plants
        ...     export_csv=True
        ... )
    """
    fetcher = PowerPlantWeatherFetcher(
        json_file_path,
        debug=debug,
        retry_attempts=retry_attempts,
        max_general_nearby=max_general_nearby,
        max_period_nearby=max_period_nearby,
        validate_from_nearby=validate_from_nearby,
        max_expand_nearby=max_expand_nearby,
        max_validated_expand=max_validated_expand
    )
    
    results = fetcher.fetch_all_plants(
        start_date=start_date,
        end_date=end_date,
        max_plants=max_plants,
        parameters=parameters,
        min_coverage_pct=min_coverage_pct
    )
    
    if export_csv and results:
        fetcher.export_to_csv(results, output_dir)
    
    return results

