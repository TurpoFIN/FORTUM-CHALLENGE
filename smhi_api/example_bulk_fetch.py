"""
Example: Fetch historical weather data for all Swedish power plants
Demonstrates the bulk fetching functionality
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from smhi_api import fetch_powerplant_weather_data


def example_fetch_few_plants():
    """
    Example 1: Fetch data for first 5 plants (quick test)
    """
    print("\n" + "="*70)
    print("Example 1: Fetching data for 5 power plants")
    print("="*70)
    
    results = fetch_powerplant_weather_data(
        json_file_path='powerplants/sweden powerplants/renewable_powerplants_sweden.json',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),  # Just January for quick test
        max_plants=5,
        export_csv=True,
        output_dir='weather_data_test'
    )
    
    print(f"\n✓ Fetched data for {len(results)} plants")
    
    # Display summary
    for plant_name, data in list(results.items())[:3]:
        print(f"\n{plant_name}:")
        print(f"  Type: {data['plant_info']['source']}")
        print(f"  Parameters available: {list(data['weather_data'].keys())}")
        
        for param, df in data['weather_data'].items():
            print(f"    - {param}: {len(df)} hourly records")


def example_fetch_specific_parameters():
    """
    Example 2: Fetch only specific weather parameters
    """
    print("\n" + "="*70)
    print("Example 2: Fetching only temperature and wind speed")
    print("="*70)
    
    results = fetch_powerplant_weather_data(
        json_file_path='powerplants/sweden powerplants/renewable_powerplants_sweden.json',
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2023, 6, 30),
        max_plants=3,
        parameters=['temperature', 'wind_speed'],  # Only these two
        export_csv=False
    )
    
    print(f"\n✓ Fetched data for {len(results)} plants")


def example_full_year_data():
    """
    Example 3: Fetch full year of data (warning: this takes time!)
    """
    print("\n" + "="*70)
    print("Example 3: Fetching full year 2023 for 10 plants")
    print("WARNING: This will take several minutes!")
    print("="*70)
    
    results = fetch_powerplant_weather_data(
        json_file_path='powerplants/sweden powerplants/renewable_powerplants_sweden.json',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        max_plants=10,
        export_csv=True,
        output_dir='weather_data_2023'
    )
    
    print(f"\n✓ Fetched data for {len(results)} plants")
    
    # Create combined dataset
    from smhi_api.bulk_fetcher import PowerPlantWeatherFetcher
    fetcher = PowerPlantWeatherFetcher(
        'powerplants/sweden powerplants/renewable_powerplants_sweden.json'
    )
    
    combined_df = fetcher.create_combined_dataset(results)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Save combined dataset
    combined_df.to_csv('weather_data_2023/combined_all_plants.csv', index=False)
    print("✓ Saved combined dataset to weather_data_2023/combined_all_plants.csv")


def example_fortum_challenge_dates():
    """
    Example 4: Fetch data for Fortum challenge time period
    """
    print("\n" + "="*70)
    print("Example 4: Fetching data up to September 2024")
    print("(As per Fortum challenge requirements)")
    print("="*70)
    
    results = fetch_powerplant_weather_data(
        json_file_path='powerplants/sweden powerplants/renewable_powerplants_sweden.json',
        start_date=datetime(2022, 1, 1),  # 2+ years of history
        end_date=datetime(2024, 9, 30),   # Up to end of Sept 2024
        max_plants=20,  # Top 20 plants
        export_csv=True,
        output_dir='weather_data_fortum_challenge'
    )
    
    print(f"\n✓ Fetched data for {len(results)} plants")
    print("Data range: 2022-01-01 to 2024-09-30")
    print("This data can be used for training models!")


if __name__ == "__main__":
    print("\n🌤️  SMHI Bulk Weather Data Fetcher - Examples\n")
    
    # Choose which example to run
    print("Select an example to run:")
    print("1. Quick test with 5 plants (January 2023)")
    print("2. Specific parameters only (temperature + wind)")
    print("3. Full year 2023 for 10 plants (takes time!)")
    print("4. Fortum challenge dates (2022-2024, up to Sept)")
    
    choice = input("\nEnter choice (1-4) or press Enter for default (1): ").strip()
    
    if not choice:
        choice = "1"
    
    if choice == "1":
        example_fetch_few_plants()
    elif choice == "2":
        example_fetch_specific_parameters()
    elif choice == "3":
        example_full_year_data()
    elif choice == "4":
        example_fortum_challenge_dates()
    else:
        print("Invalid choice!")
    
    print("\n" + "="*70)
    print("✅ Example completed!")
    print("="*70)

