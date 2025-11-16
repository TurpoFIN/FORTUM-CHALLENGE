# FMI Open Data API Integration

Python integration for the Finnish Meteorological Institute's (FMI) Open Data API. This package provides easy access to weather observations and forecasts from Finland.

## Features

- **Historical Weather Observations**: Temperature, wind, precipitation, humidity, pressure, and more
- **Weather Forecasts**: Up to 48-hour forecasts with hourly resolution
- **Station Management**: Find stations by location, name, or near power plants
- **Data Export**: Save data to Parquet or CSV formats
- **No Authentication Required**: FMI Open Data is completely free and open

## Installation

The required dependencies are already in `requirements.txt`:

```bash
pip install requests pandas pyarrow
```

## Quick Start

### Getting Historical Observations

```python
from fmi_api import get_observations, pivot_observations

# Get observations for Helsinki
df = get_observations(
    place='Helsinki',
    start_date='2023-01-01T00:00:00Z',
    end_date='2023-01-31T23:59:59Z',
    parameters=['t2m', 'ws_10min', 'r_1h'],  # temp, wind, precipitation
    timestep=60  # hourly
)

# Pivot to get parameters as columns
df_pivoted = pivot_observations(df)
print(df_pivoted.head())
```

### Getting Weather Forecasts

```python
from fmi_api import get_forecast, pivot_forecast

# Get 48-hour forecast for Tampere
forecast_df = get_forecast(
    place='Tampere',
    parameters=['Temperature', 'WindSpeedMS', 'Precipitation1h'],
    timestep=60  # hourly
)

# Pivot forecast data
forecast_pivoted = pivot_forecast(forecast_df)
print(forecast_pivoted.head())
```

### Finding Weather Stations

```python
from fmi_api import get_nearby_stations, find_station_by_name

# Find stations near a location (lat, lon)
stations = get_nearby_stations(60.17, 24.94, max_count=5)
for station in stations:
    print(f"{station['name']}: {station['distance_km']:.1f} km away")

# Find a station by name
helsinki_station = find_station_by_name('Helsinki')
print(f"Station ID: {helsinki_station['fmisid']}")
```

### Using Specific Station IDs

```python
from fmi_api import get_observations

# Get data from specific station (using FMISID)
df = get_observations(
    fmisid='100971',  # Helsinki Kaisaniemi
    start_date='2023-06-01T00:00:00Z',
    end_date='2023-06-30T23:59:59Z',
    parameters=['t2m']
)
```

### Find Stations for Power Plants

```python
from fmi_api import find_stations_for_power_plants

# Find stations near Finnish power plants
stations_by_plant = find_stations_for_power_plants()

for plant_name, stations in stations_by_plant.items():
    print(f"\n{plant_name}:")
    for station in stations[:3]:  # Top 3 closest
        print(f"  - {station['name']}: {station['distance_km']:.1f} km")
```

## Available Parameters

### Observation Parameters (use with `get_observations`)

```python
from fmi_api import WEATHER_PARAMETERS

# Common parameters:
't2m'          # Air temperature (°C)
'rh'           # Relative humidity (%)
'ws_10min'     # Wind speed 10min avg (m/s)
'wd_10min'     # Wind direction (°)
'wg_10min'     # Wind gust (m/s)
'r_1h'         # Precipitation 1h (mm)
'p_sea'        # Sea level pressure (hPa)
'snow_aws'     # Snow depth (cm)
```

### Forecast Parameters (use with `get_forecast`)

```python
from fmi_api import FORECAST_PARAMETERS

# Common parameters:
'Temperature'      # Air temperature (°C)
'Humidity'         # Relative humidity (%)
'WindSpeedMS'      # Wind speed (m/s)
'WindDirection'    # Wind direction (°)
'WindGust'         # Wind gust (m/s)
'Precipitation1h'  # Precipitation 1h (mm)
'Pressure'         # Pressure (hPa)
'TotalCloudCover'  # Cloud cover (%)
```

## Convenience Functions

### Temperature Only

```python
from fmi_api import get_temperature_observations, get_temperature_forecast

# Historical temperature
temp_obs = get_temperature_observations(
    place='Helsinki',
    start_date='2023-01-01T00:00:00Z',
    end_date='2023-12-31T23:59:59Z'
)

# Temperature forecast
temp_forecast = get_temperature_forecast(place='Helsinki')
```

### Wind Data

```python
from fmi_api import get_wind_observations, get_wind_forecast

# Historical wind
wind_obs = get_wind_observations(place='Oulu')

# Wind forecast
wind_forecast = get_wind_forecast(latlon='65.01,25.47')
```

### Comprehensive Weather Data

```python
from fmi_api import (
    get_comprehensive_weather_observations,
    get_comprehensive_weather_forecast
)

# All observation parameters at once
weather_obs = get_comprehensive_weather_observations(
    place='Turku',
    start_date='2023-01-01T00:00:00Z',
    end_date='2023-12-31T23:59:59Z'
)

# All forecast parameters at once
weather_forecast = get_comprehensive_weather_forecast(place='Turku')
```

## Data Processing

### Pivot Data

```python
from fmi_api import pivot_observations

# Convert long format to wide format (parameters as columns)
df_wide = pivot_observations(df_long)
```

### Aggregate to Daily

```python
from fmi_api import aggregate_to_daily

# Convert hourly to daily averages
df_daily = aggregate_to_daily(df_hourly)
```

### Save and Load Data

```python
from fmi_api import save_to_parquet, load_from_parquet

# Save to Parquet (efficient, compressed)
save_to_parquet(df, 'helsinki_weather_2023')

# Load from Parquet
df = load_from_parquet('data/helsinki_weather_2023.parquet')
```

## Advanced Usage

### Using the Client Directly

```python
from fmi_api import FMIClient

with FMIClient() as client:
    # Get observations
    root = client.get_observations(
        place='Helsinki',
        start_time='2023-01-01T00:00:00Z',
        end_time='2023-01-02T00:00:00Z',
        parameters=['t2m'],
        timestep=60
    )
    
    # root is an XML ElementTree that you can parse manually
```

### Bounding Box Queries

```python
from fmi_api import get_observations

# Get all stations in a bounding box
# Format: 'minLon,minLat,maxLon,maxLat'
df = get_observations(
    bbox='24.0,60.0,25.0,61.0',  # Area around Helsinki
    start_date='2023-01-01T00:00:00Z',
    end_date='2023-01-02T00:00:00Z',
    parameters=['t2m']
)
```

## Time Formats

All times should be in ISO 8601 format with UTC timezone:

```python
# Correct format
'2023-01-01T00:00:00Z'
'2024-06-15T12:30:00Z'

# Alternative: let Python handle it
from datetime import datetime, timedelta, timezone

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=7)

df = get_observations(
    place='Helsinki',
    start_date=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    end_date=end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
)
```

## Tips for the Fortum Challenge

### 1. Get Historical Weather for Training

```python
from fmi_api import get_comprehensive_weather_observations, save_to_parquet

# Get 2-3 years of historical data for your power plant locations
for plant, stations in find_stations_for_power_plants().items():
    if stations:
        station_id = stations[0]['fmisid']  # Use closest station
        
        df = get_comprehensive_weather_observations(
            fmisid=station_id,
            start_date='2021-01-01T00:00:00Z',
            end_date='2023-12-31T23:59:59Z'
        )
        
        save_to_parquet(df, f'weather_history_{plant}')
```

### 2. Get Forecasts for Your Predictions

```python
from fmi_api import get_comprehensive_weather_forecast

# Get 48-hour forecast for short-term predictions
forecast = get_comprehensive_weather_forecast(
    place='Helsinki',
    timestep=60  # Hourly
)

# Use this as features for your 48-hour electricity demand forecast
```

### 3. Handle Missing Data

```python
# FMI data quality is generally excellent, but handle missing values
df_pivoted = pivot_observations(df)

# Forward fill small gaps
df_pivoted = df_pivoted.fillna(method='ffill', limit=2)

# Or interpolate
df_pivoted = df_pivoted.interpolate(method='linear')
```

## Troubleshooting

### No Data Returned

- Check that the place name is correct (try 'Helsinki', 'Tampere', 'Oulu')
- Verify date format: `'YYYY-MM-DDTHH:MM:SSZ'`
- Some parameters may not be available at all stations
- Try using `fmisid` instead of `place` for more reliable results

### XML Parsing Errors

- The FMI API might be temporarily down - retry in a few minutes
- Very large date ranges might timeout - split into smaller chunks

### Station Not Found

```python
# Find available stations first
from fmi_api import get_all_stations

all_stations = get_all_stations()
print(all_stations[['fmisid', 'name', 'latitude', 'longitude']])
```

## References

- [FMI Open Data Homepage](https://en.ilmatieteenlaitos.fi/open-data)
- [FMI Open Data Manual](https://en.ilmatieteenlaitos.fi/open-data-manual)
- [WFS Service Endpoint](https://opendata.fmi.fi/wfs)

## License

This integration package is provided for the Fortum Challenge. The FMI Open Data is licensed under Creative Commons Attribution 4.0 International License.


