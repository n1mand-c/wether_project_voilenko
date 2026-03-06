"""
fetch_data.py — завантаження даних з Open-Meteo та збереження у CSV.
Запуск: python fetch_data.py
"""

import requests
import pandas as pd
from datetime import date, timedelta

# ─── Параметри ───────────────────────────────────────────────────
LATITUDE  = 50.45   # Київ
LONGITUDE = 30.52
START_DATE = str(date.today() - timedelta(days=365))
END_DATE   = str(date.today() - timedelta(days=1))
OUTPUT_CSV = "weather_daily.csv"
# ─────────────────────────────────────────────────────────────────


def fetch_weather(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily": [
            "precipitation_sum",
            "rain_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "windspeed_10m_max",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
            "precipitation_hours",
        ],
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={"time": "date"}, inplace=True)
    return df


if __name__ == "__main__":
    print(f"Завантаження даних для координат ({LATITUDE}, {LONGITUDE})")
    print(f"Період: {START_DATE} — {END_DATE}")
    df = fetch_weather(LATITUDE, LONGITUDE, START_DATE, END_DATE)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Збережено {len(df)} рядків у файл '{OUTPUT_CSV}'")
    print(df.head())
