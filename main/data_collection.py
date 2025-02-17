import requests
import os
import pandas as pd
import subprocess

# Set your OpenWeatherMap API key here
api_key = '1372afb04efaf355aaa403f7894a99ea'  # Replace with your actual API key

def test_openweather_pollution_api(city='London'):
    """Test OpenWeatherMap Air Pollution API and collect data"""
    if not api_key:
        print("Error: API key is missing.")
        return None

    # Use the latitude and longitude of the city. For now, using example lat, lon.
    base_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=50&lon=50&appid={api_key}"
    
    # Make the API request
    response = requests.get(base_url)
    
    # Check the status code
    if response.status_code == 200:
        print("API Request Successful!")
        data = response.json()
        print("API Response:")
        print(data)  # You can format this output as per your needs

        # If the response contains data, display air quality information
        if 'list' in data and len(data['list']) > 0:
            air_quality = data['list'][0]
            pollution_data = {
                'timestamp': pd.Timestamp.now(),
                'aqi': air_quality['main']['aqi'],
                'co': air_quality['components']['co'],
                'no2': air_quality['components']['no2'],
                'o3': air_quality['components']['o3'],
                'pm2_5': air_quality['components']['pm2_5'],
                'pm10': air_quality['components']['pm10']
            }
            return pd.DataFrame([pollution_data])
        else:
            print("No data found for the specified location.")
            return None
    else:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response Text: {response.text}")
        return None

def collect_data():
    """Collect and save data using DVC"""
    df = test_openweather_pollution_api()

    if df is not None:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)

        # Save data with timestamp
        filename = f'data/pollution_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        
        # # Add and commit with DVC (Assuming DVC and git are set up)
        # subprocess.run(['dvc', 'add', filename])
        # subprocess.run(['git', 'add', f'{filename}.dvc'])
        # subprocess.run(['git', 'commit', '-m', 'Add new pollution data'])
        # subprocess.run(['dvc', 'push'])
    else:
        print("No data to save.")

# Call collect_data to fetch and save the pollution data
collect_data()
