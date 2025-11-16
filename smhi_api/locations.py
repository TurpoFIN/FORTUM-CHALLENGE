"""
Pre-defined power plant locations from Fortum challenge
"""

# Swedish power plant locations
POWER_PLANTS = {
    'markbygden_wind': {
        'name': 'Markbygden Wind Farm',
        'location': 'Piteå, Sweden',
        'latitude': 65.92,
        'longitude': 21.05,
        'type': 'wind',
        'relevant_parameters': ['wind_speed', 'temperature', 'precipitation']
    },
    'harspranget_hydro': {
        'name': 'Harsprånget Hydropower',
        'location': 'Jokkmokk, Sweden',
        'latitude': 66.60,
        'longitude': 20.40,
        'type': 'hydro',
        'relevant_parameters': ['precipitation', 'temperature', 'humidity']
    },
    'hultsfreds_solar': {
        'name': 'Hultsfreds Solar Farm',
        'location': 'Hultsfreds, Sweden',
        'latitude': 57.48,
        'longitude': 15.84,
        'type': 'solar',
        'relevant_parameters': ['temperature', 'humidity']
    }
}


def get_location_coordinates(plant_name: str):
    """
    Get coordinates for a power plant by name
    
    Args:
        plant_name: Key from POWER_PLANTS dictionary
        
    Returns:
        Tuple of (latitude, longitude)
    """
    if plant_name not in POWER_PLANTS:
        raise ValueError(f"Unknown plant: {plant_name}. Available: {list(POWER_PLANTS.keys())}")
    
    plant = POWER_PLANTS[plant_name]
    return plant['latitude'], plant['longitude']


def get_plant_info(plant_name: str):
    """
    Get full information about a power plant
    
    Args:
        plant_name: Key from POWER_PLANTS dictionary
        
    Returns:
        Dictionary with plant information
    """
    if plant_name not in POWER_PLANTS:
        raise ValueError(f"Unknown plant: {plant_name}. Available: {list(POWER_PLANTS.keys())}")
    
    return POWER_PLANTS[plant_name]


