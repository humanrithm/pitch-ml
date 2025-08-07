import requests

def get_player_metadata(
        mlbam_id: int,
        url: str = "https://statsapi.mlb.com/api/v1/people"
) -> dict:
    """ 
    Fetch player metadata from MLB API. 
    
    Args:
        mlbam_id (int): MLBAM ID of the player.
        url (str): Base URL for the MLB API.
    Returns:
        dict: Player metadata including height, weight, and mlbam ID.

    """
    response = requests.get(f"{url}/{mlbam_id}")
    data = response.json()
    
    if 'people' in data and len(data['people']) > 0:
        player_data = data['people'][0]
        return {
            'mlbam_id': player_data['id'],
            'full_name': player_data['fullName'],
            'height': convert_height_to_meters(player_data['height']),
            'mass': player_data['weight'] * 0.453592,
        }
    else:
        return {}
    
def convert_height_to_meters(height_str: str) -> float:
    """ 
    Convert height from string format to meters.
    
    Args:
        height_str (str): Height in the format "6' 4\"".
    Returns:
        float: Height in meters.
    """
    feet, inches = height_str.split("'")
    feet = int(feet.strip())
    inches = int(inches.strip().replace('"', ''))
    return round((feet * 12 + inches) * 0.0254, 3)
