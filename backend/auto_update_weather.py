import requests
import pandas as pd
from datetime import datetime
import json
import random
import os

# Path to coordinates file (assume same folder)
coord_file = os.path.join(os.path.dirname(__file__), "saudi_city_coords.json")
csv_file = os.path.join(os.path.dirname(__file__), "historical_data.csv")

# Load coordinates
with open(coord_file, "r") as f:
    city_coords = json.load(f)

API_KEY = "b38b0e5e6e4417145d959465c24d0823"  
today = datetime.now().strftime("%Y-%m-%d")
data = []

for city, (lat, lon) in city_coords.items():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url)
        weather = res.json()

        data.append({
            "City": city,
            "Date": today,
            "Temperature": weather["main"]["temp"],
            "Humidity": weather["main"]["humidity"],
            "Wind": weather["wind"]["speed"],
            "Pressure": weather["main"]["pressure"],
            "Consumption": round(random.uniform(1000, 1800), 2)
        })
    except Exception as e:
        print(f"Failed for {city}: {e}")

# Append to CSV
df_existing = pd.read_csv(csv_file)
df_new = pd.DataFrame(data)
df_combined = pd.concat([df_existing, df_new])
df_combined.drop_duplicates(subset=["City", "Date"], keep="last", inplace=True)
df_combined.to_csv(csv_file, index=False)

print("✅ Weather data updated.")
