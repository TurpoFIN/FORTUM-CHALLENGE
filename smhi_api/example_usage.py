"""
Example usage of SMHI API integration
Demonstrates how to fetch weather data for Fortum power plant locations
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from smhi_api.observations import (
    get_temperature_data,
    get_wind_speed_data, 
    get_humidity_data,
    get_precipitation_data,
    get_all_weather_data
)
from smhi_api.locations import POWER_PLANTS, get_location_coordinates


def example_single_parameter():
    """Example: Fetch a single weather parameter"""
    print("=" * 60)
    print("Example 1: Fetching temperature data for Markbygden wind farm")
    print("=" * 60)
    
    lat, lon = get_location_coordinates('markbygden_wind')
    df = get_temperature_data(lat, lon, period='latest-months')
    
    print(f"\nStation: {df.attrs['metadata']['station_name']}")
    print(f"Parameter: {df.attrs['metadata']['parameter']}")
    print(f"Unit: {df.attrs['metadata']['unit']}")
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst 5 observations:")
    print(df.head())
    print(f"\nLast 5 observations:")
    print(df.tail())
    print(f"\nTemperature statistics:")
    print(df['temperature_celsius'].describe())


def example_wind_farm():
    """Example: Fetch wind speed for wind farm"""
    print("\n" + "=" * 60)
    print("Example 2: Wind speed data for Markbygden wind farm")
    print("=" * 60)
    
    plant = POWER_PLANTS['markbygden_wind']
    lat, lon = plant['latitude'], plant['longitude']
    
    df = get_wind_speed_data(lat, lon, period='latest-months')
    
    print(f"\nPlant: {plant['name']}")
    print(f"Location: {plant['location']}")
    print(f"\nWind speed statistics (m/s):")
    print(df['wind_speed_ms'].describe())
    
    # Calculate potential wind power (simplified)
    # Power ∝ velocity³
    df['relative_power'] = df['wind_speed_ms'] ** 3
    print(f"\nRelative power statistics:")
    print(df['relative_power'].describe())


def example_hydro_plant():
    """Example: Fetch precipitation for hydropower plant"""
    print("\n" + "=" * 60)
    print("Example 3: Precipitation data for Harsprånget hydropower")
    print("=" * 60)
    
    lat, lon = get_location_coordinates('harspranget_hydro')
    
    # Get both precipitation intensity and amount
    df_intensity = get_precipitation_data(lat, lon, period='latest-months', 
                                         precipitation_type='intensity')
    df_amount = get_precipitation_data(lat, lon, period='latest-months',
                                      precipitation_type='amount')
    
    print(f"\nPrecipitation intensity data:")
    print(df_intensity.head())
    print(f"\nIntensity statistics (mm/h):")
    print(df_intensity['precipitation_intensity_mmh'].describe())
    
    print(f"\nPrecipitation amount data:")
    print(df_amount.head())
    print(f"\nAmount statistics (mm):")
    print(df_amount['precipitation_amount_mm'].describe())


def example_all_parameters():
    """Example: Fetch all weather parameters for a location"""
    print("\n" + "=" * 60)
    print("Example 4: All weather data for Hultsfreds solar farm")
    print("=" * 60)
    
    plant = POWER_PLANTS['hultsfreds_solar']
    lat, lon = plant['latitude'], plant['longitude']
    
    weather = get_all_weather_data(lat, lon, period='latest-months')
    
    print(f"\nPlant: {plant['name']}")
    print(f"Type: {plant['type'].upper()}")
    
    for param_name, df in weather.items():
        if df is not None:
            print(f"\n{param_name.upper()}:")
            print(f"  Records: {len(df)}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            value_col = [col for col in df.columns if col != 'timestamp' and col != 'quality'][0]
            print(f"  Mean: {df[value_col].mean():.2f}")
        else:
            print(f"\n{param_name.upper()}: No data available")


def example_all_plants():
    """Example: Fetch temperature for all power plants"""
    print("\n" + "=" * 60)
    print("Example 5: Temperature comparison across all plants")
    print("=" * 60)
    
    for plant_key, plant_info in POWER_PLANTS.items():
        print(f"\n{plant_info['name']} ({plant_info['type'].upper()}):")
        
        try:
            lat, lon = plant_info['latitude'], plant_info['longitude']
            df = get_temperature_data(lat, lon, period='latest-months')
            
            print(f"  Station: {df.attrs['metadata']['station_name']}")
            print(f"  Records: {len(df)}")
            print(f"  Mean temp: {df['temperature_celsius'].mean():.2f}°C")
            print(f"  Min temp: {df['temperature_celsius'].min():.2f}°C")
            print(f"  Max temp: {df['temperature_celsius'].max():.2f}°C")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    print("\n🌤️  SMHI API Integration - Example Usage\n")
    
    # Run examples
    # Note: Uncomment the examples you want to run
    # Each makes API calls which may take some time
    
    example_single_parameter()
    # example_wind_farm()
    # example_hydro_plant()
    # example_all_parameters()
    # example_all_plants()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)


