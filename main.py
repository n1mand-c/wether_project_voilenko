"""
main.py — Варіант B: скрипт без Streamlit.
Завантажує дані, навчає модель, виводить метрики та прогноз.
Запуск: python main.py
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ─── Параметри ───────────────────────────────────────────────────
LATITUDE   = 50.45    # Київ
LONGITUDE  = 30.52
START_DATE = str(date.today() - timedelta(days=365))
END_DATE   = str(date.today() - timedelta(days=1))
CSV_FILE   = "weather_daily.csv"
# ─────────────────────────────────────────────────────────────────


def fetch_or_load():
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=["date"])
        print(f"✅ Дані завантажені з '{CSV_FILE}' ({len(df)} рядків)\n")
    except FileNotFoundError:
        print("CSV не знайдено — завантажуємо з Open-Meteo...")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": START_DATE, "end_date": END_DATE,
            "daily": [
                "precipitation_sum", "rain_sum",
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "windspeed_10m_max",
                "relative_humidity_2m_max", "relative_humidity_2m_min",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration",
                "precipitation_hours",
            ],
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json()["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df.rename(columns={"time": "date"}, inplace=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"✅ Збережено у '{CSV_FILE}' ({len(df)} рядків)\n")
    return df


def prepare(df):
    df = df.copy()
    df["rain_label"] = (df["precipitation_sum"] > 0).astype(int)
    df["precip_lag1"] = df["precipitation_sum"].shift(1)
    df["precip_lag2"] = df["precipitation_sum"].shift(2)
    df["precip_lag3"] = df["precipitation_sum"].shift(3)
    df["label_lag1"]  = df["rain_label"].shift(1)
    df["temp_rolling7"] = df["temperature_2m_mean"].rolling(7).mean()
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feat = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "windspeed_10m_max", "relative_humidity_2m_max", "relative_humidity_2m_min",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration", "precipitation_hours",
        "precip_lag1", "precip_lag2", "precip_lag3", "label_lag1",
        "temp_rolling7", "temp_range", "month_sin", "month_cos",
    ]
    return df, df[feat], df["rain_label"], feat


def print_metrics(name, y_test, y_pred, y_prob):
    print(f"\n{'='*50}")
    print(f"  Модель: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\n  Матриця помилок:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                Прогноз: 0   Прогноз: 1")
    print(f"  Факт: 0 (сухо)    {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  Факт: 1 (дощ)     {cm[1,0]:5d}       {cm[1,1]:5d}")


def main():
    print("🌧️  Прогноз опадів — Open-Meteo + ML\n")

    # 1. Дані
    df_raw = fetch_or_load()
    rainy = (df_raw["precipitation_sum"] > 0).sum()
    print(f"  Загалом днів  : {len(df_raw)}")
    print(f"  Дощових днів  : {rainy} ({rainy/len(df_raw)*100:.1f}%)")
    print(f"  Сухих днів    : {len(df_raw) - rainy}")

    # 2. Ознаки
    df, X, y, feat_cols = prepare(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Навчання двох моделей
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_tr_sc, y_train)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]
        print_metrics(name, y_test, y_pred, y_prob)
        trained[name] = model

    # 4. Прогноз для останнього дня датасету
    print(f"\n{'='*50}")
    print("  ПРОГНОЗ ДЛЯ ОСТАННЬОГО ДНЯ ДАТАСЕТУ")
    print(f"{'='*50}")

    last_row = df.iloc[[-1]]
    forecast_date = last_row["date"].values[0]
    X_last = scaler.transform(last_row[feat_cols].values)

    for name, model in trained.items():
        pred = model.predict(X_last)[0]
        prob = model.predict_proba(X_last)[0]
        emoji = "☔" if pred == 1 else "☀️"
        result = "Очікуються опади" if pred == 1 else "Опадів не очікується"
        print(f"\n  {emoji} [{name}]")
        print(f"     Дата: {str(forecast_date)[:10]}")
        print(f"     Прогноз: {result}")
        print(f"     Ймовірність опадів   : {prob[1]*100:.1f}%")
        print(f"     Ймовірність без опадів: {prob[0]*100:.1f}%")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    main()
