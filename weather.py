# fetch_weather.py
import requests
from datetime import datetime

API_KEY = "b38b0e5e6e4417145d959465c24d0823"

def get_forecast_weather(city, date_str):
    """Returns weather for a specific date (closest to noon)"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200 or "list" not in data:
        print(f"Failed to fetch forecast for {city}: {data.get('message', 'No data')}")
        return None

    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    closest_entry = None
    min_hour_diff = float('inf')

    for entry in data["list"]:
        dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
        if dt.date() == target_date:
            diff = abs(dt.hour - 12)
            if diff < min_hour_diff:
                min_hour_diff = diff
                closest_entry = entry

    if not closest_entry:
        return None

    return {
        "date": closest_entry["dt_txt"].split(" ")[0],
        "temp": closest_entry["main"]["temp"],
        "humidity": closest_entry["main"]["humidity"],
        "wind_speed": closest_entry["wind"]["speed"],
        "pressure": closest_entry["main"]["pressure"],
        "clouds": closest_entry["clouds"]["all"],
        "lat": data["city"]["coord"]["lat"],
        "lon": data["city"]["coord"]["lon"]
    }

def get_forecast_weather_range(city, start_date, end_date):
    """Returns list of daily weather entries (12:00 pm) for a date range"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200 or "list" not in data:
        raise Exception(f"Failed to fetch forecast for {city}: {data.get('message', 'No data')}")

    forecasts = data["list"]
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    output = []
    added_dates = set()

    for entry in forecasts:
        dt = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
        if start_dt <= dt.date() <= end_dt and dt.hour == 12:
            date_str = dt.strftime("%Y-%m-%d")
            if date_str not in added_dates:
                output.append({
                    "date": date_str,
                    "temp": entry["main"]["temp"],
                    "humidity": entry["main"]["humidity"],
                    "wind_speed": entry["wind"]["speed"],
                    "pressure": entry["main"]["pressure"],
                    "clouds": entry["clouds"]["all"],
                    "lat": data["city"]["coord"]["lat"],
                    "lon": data["city"]["coord"]["lon"]
                })
                added_dates.add(date_str)

    return output
