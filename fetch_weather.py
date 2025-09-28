# fetch_weather.py
import requests
from datetime import datetime

API_KEY = "b38b0e5e6e4417145d959465c24d0823"

# Optional: override city names for OpenWeather compatibility
CITY_NAME_OVERRIDES = {
    "Makkah": "Mecca",
    "Madinah": "Medina",
    "Al Ahsa": "Hofuf",
    "Al Khobar": "Khobar",
    "Jazan": "Jizan"
}

def get_forecast_weather(city, date_str):
    """
    Fetches weather forecast data for a given city and date from OpenWeather API.

    Args:
        city (str): Saudi city name (e.g., "Riyadh")
        date_str (str): Date in format YYYY-MM-DD

    Returns:
        dict: Weather features (temp, humidity, wind_speed, pressure, clouds, lat, lon)
    """
    original_city = city
    city = CITY_NAME_OVERRIDES.get(city, city)

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200 or "list" not in data:
        print(f"❌ Failed to fetch forecast for {original_city} → {city}: {data.get('message', 'No data')}")
        return None

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    closest_entry = None
    min_hour_diff = float('inf')

    # Try to find entry closest to 12 PM on target date
    for entry in data["list"]:
        dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
        if dt.date() == target_date:
            diff = abs(dt.hour - 12)
            if diff < min_hour_diff:
                min_hour_diff = diff
                closest_entry = entry

    # Fallback: use the first available entry on the date
    if not closest_entry:
        for entry in data["list"]:
            dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
            if dt.date() == target_date:
                closest_entry = entry
                break

    if not closest_entry:
        print(f"⚠️ No forecast data found for {original_city} on {target_date}")
        return None

    result = {
        "date": closest_entry["dt_txt"].split(" ")[0],
        "temp": closest_entry["main"]["temp"],
        "humidity": closest_entry["main"]["humidity"],
        "wind_speed": closest_entry["wind"]["speed"],
        "pressure": closest_entry["main"]["pressure"],
        "clouds": closest_entry["clouds"]["all"],
        "lat": data["city"]["coord"]["lat"],
        "lon": data["city"]["coord"]["lon"]
    }

    print(f"✅ Weather fetched for {original_city} ({city}): {result}")
    return result
