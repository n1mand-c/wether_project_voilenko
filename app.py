"""
Прогноз опадів на основі даних Open-Meteo
Streamlit-застосунок з ML-класифікацією — преміум дизайн
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ══════════════════════════════════════════
# Конфігурація
# ══════════════════════════════════════════
st.set_page_config(page_title="RainCast · ML", page_icon="🌧️", layout="wide")

# ══════════════════════════════════════════
# Глобальний CSS — елегантна темна тема
# ══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

/* Базові скидання */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Загальний фон */
.stApp {
    background: #0a0f1a;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,102,241,0.06) 0%, transparent 60%);
}

/* Сайдбар */
[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

/* Інпути в сайдбарі */
[data-testid="stSidebar"] input {
    background: #161f33 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Кнопки */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1.09rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(14,165,233,0.3) !important;
}

/* Секондарні кнопки в сайдбарі */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    font-size: 0.94rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.12) !important;
    box-shadow: none !important;
}

/* Таблиці */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Метрики */
[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.97rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 2.05rem !important;
    font-weight: 600 !important;
}

/* Алерти */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #111827 !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* Toggle */
[data-testid="stToggle"] { accent-color: #0ea5e9; }

/* Приховати зайве */
#MainMenu, footer, header { visibility: hidden; }

/* Горизонтальний роздільник */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 2rem 0 !important; }

/* Заголовки секцій */
h2 { 
    color: #f1f5f9 !important; 
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.69rem !important;
    margin-top: 2rem !important;
}
h3 { color: #cbd5e1 !important; font-size: 1.29rem !important; font-weight: 500 !important; }

/* Текст */
p, li, span { color: #94a3b8 !important; font-size: 1.06rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Хедер
# ══════════════════════════════════════════
st.markdown("""
<div style="padding: 2rem 0 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 2rem;">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="width:48px;height:48px;background:linear-gradient(135deg,#0ea5e9,#6366f1);
                    border-radius:14px;display:flex;align-items:center;justify-content:center;
                    font-size:1.4rem;box-shadow:0 8px 24px rgba(14,165,233,0.25);">🌧️</div>
        <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.9rem;color:#f1f5f9;line-height:1.1;">
                RainCast <span style="font-style:italic;color:#0ea5e9;">ML</span>
            </div>
            <div style="color:#475569;font-size:0.85rem;margin-top:2px;letter-spacing:0.03em;">
                Прогнозування опадів · Open-Meteo + Machine Learning
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Пошук міста
# ══════════════════════════════════════════
@st.cache_data(show_spinner=False)
def search_city(name):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": 5, "language": "uk"}, timeout=10)
    r.raise_for_status()
    return r.json().get("results", [])

# ══════════════════════════════════════════
# Сайдбар
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem;">
        <div style="font-size:0.83rem;letter-spacing:0.15em;text-transform:uppercase;
                    color:#334155;font-weight:600;margin-bottom:0.3rem;">Налаштування</div>
    </div>
    """, unsafe_allow_html=True)

    # Стан координат
    if "latitude" not in st.session_state:
        st.session_state.update({"latitude":50.45,"longitude":30.52,"city_name":"Київ"})

    st.markdown('<div style="font-size:0.91rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">📍 Місто</div>', unsafe_allow_html=True)
    city_input = st.text_input("", placeholder="Введіть назву міста...", label_visibility="collapsed")

    if city_input and st.button("Знайти →", use_container_width=True):
        with st.spinner(""):
            try:
                res = search_city(city_input)
                st.session_state["city_results"] = res if res else []
                if not res:
                    st.error("Місто не знайдено")
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("city_results"):
        res = st.session_state["city_results"]
        opts = [f"{r['name']}, {r.get('admin1','')}, {r.get('country','')}" for r in res]
        chosen = st.selectbox("", opts, label_visibility="collapsed")
        if st.button("✓ Застосувати", use_container_width=True):
            sel = res[opts.index(chosen)]
            st.session_state.update({"latitude":sel["latitude"],"longitude":sel["longitude"],
                                      "city_name":sel["name"],"city_results":[]})
            st.rerun()

    lat_val   = st.session_state["latitude"]
    lon_val   = st.session_state["longitude"]
    city_name = st.session_state["city_name"]

    st.markdown(f"""
    <div style="background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2);
                border-radius:10px;padding:10px 14px;margin:12px 0;">
        <div style="color:#0ea5e9 !important;font-size:0.95rem;font-weight:600;">{city_name}</div>
        <div style="color:#475569 !important;font-size:0.88rem;margin-top:2px;">{lat_val:.4f}° N, {lon_val:.4f}° E</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    show_map = st.toggle("🗺️ Карта вибору", value=False)

    st.markdown("---")
    st.markdown('<div style="font-size:0.91rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">📅 Період</div>', unsafe_allow_html=True)
    today = date.today()
    start_date = st.date_input("Від", value=today - timedelta(days=365), label_visibility="visible")
    end_date   = st.date_input("До",  value=today - timedelta(days=1),   label_visibility="visible")

    st.markdown("---")
    st.markdown('<div style="font-size:0.91rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">🤖 Модель</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("", ["Random Forest","Logistic Regression","Обидва (порівняння)"], label_visibility="collapsed")

    st.markdown("""
    <div style="position:absolute;bottom:1.5rem;left:1.5rem;right:1.5rem;
                font-size:0.85rem;color:#1e293b;text-align:center;line-height:1.5;">
        Open-Meteo · scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# Карта
# ══════════════════════════════════════════
if show_map:
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#f1f5f9;margin-bottom:1rem;">
        🗺️ Оберіть місце на карті
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.html(f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{margin:0;padding:0;box-sizing:border-box;}}
  body {{background:#0a0f1a;}}
  #map {{height:400px;width:100%;border-radius:16px;border:1px solid rgba(255,255,255,0.08);}}
  #coords {{
    background:rgba(13,20,36,0.95);
    backdrop-filter:blur(12px);
    border:1px solid rgba(14,165,233,0.25);
    color:#94a3b8;
    padding:12px 20px;
    font-family:'DM Sans',system-ui,sans-serif;
    font-size:13px;
    border-radius:12px;
    margin-top:10px;
    display:flex;
    gap:28px;
    align-items:center;
  }}
  #coords span b {{color:#0ea5e9;}}
  .leaflet-tile-pane {{ filter: brightness(0.85) saturate(0.9); }}
</style>
</head><body>
<div id="map"></div>
<div id="coords">
  <span>📍 Широта: <b id="lv">{lat_val:.4f}</b></span>
  <span>Довгота: <b id="lnv">{lon_val:.4f}</b></span>
  <span style="color:#334155;font-size:12px;">← Клікніть на карті або перетягніть маркер</span>
</div>
<script>
var map = L.map('map', {{zoomControl:true}}).setView([{lat_val},{lon_val}],6);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'© OSM'}}).addTo(map);
var icon = L.divIcon({{
  html:'<div style="width:16px;height:16px;background:#0ea5e9;border:3px solid white;border-radius:50%;box-shadow:0 0 12px rgba(14,165,233,0.6);"></div>',
  iconSize:[16,16], iconAnchor:[8,8], className:''
}});
var marker = L.marker([{lat_val},{lon_val}],{{draggable:true,icon:icon}}).addTo(map);
function upd(lat,lon){{
  document.getElementById('lv').textContent=lat.toFixed(4);
  document.getElementById('lnv').textContent=lon.toFixed(4);
}}
map.on('click',function(e){{ marker.setLatLng(e.latlng); upd(e.latlng.lat,e.latlng.lng); }});
marker.on('dragend',function(){{ var p=marker.getLatLng(); upd(p.lat,p.lng); }});
</script></body></html>""", height=480)
    st.caption("Координати з карти введіть вручну у поле пошуку міста вище або скористайтесь пошуком за назвою.")

# ══════════════════════════════════════════
# ML функції
# ══════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_data(lat, lon, start, end):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude":lat,"longitude":lon,"start_date":str(start),"end_date":str(end),
        "daily":["precipitation_sum","rain_sum","temperature_2m_max","temperature_2m_min",
                 "temperature_2m_mean","windspeed_10m_max","relative_humidity_2m_max",
                 "relative_humidity_2m_min","shortwave_radiation_sum",
                 "et0_fao_evapotranspiration","precipitation_hours"],
        "timezone":"auto"}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time":"date"})

def prepare_features(df):
    df = df.copy()
    df["rain_label"]    = (df["precipitation_sum"] > 0).astype(int)
    df["precip_lag1"]   = df["precipitation_sum"].shift(1)
    df["precip_lag2"]   = df["precipitation_sum"].shift(2)
    df["precip_lag3"]   = df["precipitation_sum"].shift(3)
    df["label_lag1"]    = df["rain_label"].shift(1)
    df["temp_rolling7"] = df["temperature_2m_mean"].rolling(7).mean()
    df["temp_range"]    = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["month_sin"]     = np.sin(2*np.pi*df["date"].dt.month/12)
    df["month_cos"]     = np.cos(2*np.pi*df["date"].dt.month/12)
    df.dropna(inplace=True); df.reset_index(drop=True,inplace=True)
    feat = ["temperature_2m_max","temperature_2m_min","temperature_2m_mean",
            "windspeed_10m_max","relative_humidity_2m_max","relative_humidity_2m_min",
            "shortwave_radiation_sum","et0_fao_evapotranspiration","precipitation_hours",
            "precip_lag1","precip_lag2","precip_lag3","label_lag1","temp_rolling7","temp_range",
            "month_sin","month_cos"]
    return df, df[feat], df["rain_label"], feat

def train_model(Xtr, ytr, name):
    sc = StandardScaler(); Xs = sc.fit_transform(Xtr)
    m = (RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")
         if name=="Random Forest"
         else LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced"))
    m.fit(Xs,ytr); return m,sc

def evaluate(m,sc,Xte,yte):
    Xs=sc.transform(Xte); yp=m.predict(Xs); ypr=m.predict_proba(Xs)[:,1]
    return {"Accuracy":accuracy_score(yte,yp),"Precision":precision_score(yte,yp,zero_division=0),
            "Recall":recall_score(yte,yp,zero_division=0),"F1-Score":f1_score(yte,yp,zero_division=0)
            }, confusion_matrix(yte,yp), yp, ypr

# ══════════════════════════════════════════
# Крок 1 — Дані
# ══════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin:2rem 0 1.2rem;">
    <div style="width:32px;height:32px;background:linear-gradient(135deg,#0ea5e9,#6366f1);
                border-radius:8px;display:flex;align-items:center;justify-content:center;
                font-size:0.85rem;font-weight:700;color:white;flex-shrink:0;">1</div>
    <div style="font-family:'DM Serif Display',serif;font-size:1.64rem;color:#f1f5f9;">Завантаження даних</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,1], gap="medium")
with col1:
    if st.button("🌐  Отримати дані з Open-Meteo", use_container_width=True):
        with st.spinner(f"Завантаження для {city_name}..."):
            try:
                df_raw = fetch_data(lat_val,lon_val,start_date,end_date)
                st.session_state["df_raw"] = df_raw
                st.success(f"✅  Завантажено {len(df_raw)} рядків · {city_name}")
            except Exception as e:
                st.error(f"Помилка: {e}")
with col2:
    up = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if up:
        df_raw = pd.read_csv(up, parse_dates=["date"])
        st.session_state["df_raw"] = df_raw
        st.success(f"✅  CSV завантажено · {len(df_raw)} рядків")

if "df_raw" in st.session_state:
    df_raw = st.session_state["df_raw"]

    rainy = (df_raw["precipitation_sum"] > 0).sum()
    dry   = len(df_raw) - rainy
    pct   = rainy / len(df_raw) * 100

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Всього днів", len(df_raw))
    with c2: st.metric("☔ З опадами", rainy)
    with c3: st.metric("☀️ Без опадів", dry)
    with c4: st.metric("Частка дощових", f"{pct:.1f}%")

    with st.expander("📋 Переглянути дані", expanded=False):
        st.dataframe(df_raw, use_container_width=True, height=280)

    st.download_button("💾  Зберегти weather_daily.csv",
                       df_raw.to_csv(index=False).encode("utf-8"),
                       "weather_daily.csv","text/csv")

# ══════════════════════════════════════════
# Крок 2 — Навчання
# ══════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin:2.5rem 0 1.2rem;">
    <div style="width:32px;height:32px;background:linear-gradient(135deg,#0ea5e9,#6366f1);
                border-radius:8px;display:flex;align-items:center;justify-content:center;
                font-size:0.85rem;font-weight:700;color:white;flex-shrink:0;">2</div>
    <div style="font-family:'DM Serif Display',serif;font-size:1.64rem;color:#f1f5f9;">Навчання моделі</div>
</div>
""", unsafe_allow_html=True)

if "df_raw" not in st.session_state:
    st.info("Спочатку завантажте дані на кроці 1.")
else:
    if st.button("🤖  Навчити модель", use_container_width=True):
        with st.spinner("Підготовка ознак та навчання..."):
            df_f,X,y,fcols = prepare_features(st.session_state["df_raw"])
            Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=False)
            mlist = (["Random Forest","Logistic Regression"] if model_choice=="Обидва (порівняння)" else [model_choice])
            res = {}
            for mn in mlist:
                m,sc = train_model(Xtr,ytr,mn)
                met,cm,_,_ = evaluate(m,sc,Xte,yte)
                res[mn] = {"model":m,"scaler":sc,"metrics":met,"cm":cm}
            st.session_state.update({"results":res,"feature_cols":fcols,"df_feat":df_f})
        st.success("✅  Модель навчена!")

    if "results" in st.session_state:
        res   = st.session_state["results"]
        fcols = st.session_state["feature_cols"]

        # Метрики — красиві картки
        for mn, rv in res.items():
            st.markdown(f'<div style="font-size:0.97rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.6rem;">{mn}</div>', unsafe_allow_html=True)
            m_cols = st.columns(4)
            icons  = {"Accuracy":"🎯","Precision":"🔬","Recall":"📡","F1-Score":"⚡"}
            for i,(k,v) in enumerate(rv["metrics"].items()):
                with m_cols[i]:
                    color = "#22c55e" if v>=0.75 else "#f59e0b" if v>=0.6 else "#ef4444"
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
                                border-radius:14px;padding:1.1rem 1.3rem;text-align:center;">
                        <div style="font-size:0.91rem;color:#475569;text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:0.5rem;">{icons[k]} {k}</div>
                        <div style="font-size:2.09rem;font-weight:600;color:{color};line-height:1;">{v:.3f}</div>
                    </div>""", unsafe_allow_html=True)

        # Матриця помилок
        st.markdown('<div style="font-size:0.97rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:1.5rem 0 0.6rem;">Матриця помилок</div>', unsafe_allow_html=True)
        for mn,rv in res.items():
            cm = rv["cm"]
            st.markdown(f"**{mn}**")
            st.dataframe(pd.DataFrame(cm,
                index=["Факт: Без опадів","Факт: Опади"],
                columns=["Прогноз: Без опадів","Прогноз: Опади"]), use_container_width=True)

        # Важливість ознак
        if "Random Forest" in res:
            rf = res["Random Forest"]["model"]
            imp = pd.DataFrame({"Ознака":fcols,"Важливість":rf.feature_importances_}).sort_values("Важливість",ascending=False).head(10)
            st.markdown('<div style="font-size:0.97rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin:1.5rem 0 0.6rem;">🏆 Топ-10 важливих ознак (Random Forest)</div>', unsafe_allow_html=True)
            st.bar_chart(imp.set_index("Ознака")["Важливість"])

# ══════════════════════════════════════════
# Крок 3 — Прогноз
# ══════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin:2.5rem 0 1.2rem;">
    <div style="width:32px;height:32px;background:linear-gradient(135deg,#0ea5e9,#6366f1);
                border-radius:8px;display:flex;align-items:center;justify-content:center;
                font-size:0.85rem;font-weight:700;color:white;flex-shrink:0;">3</div>
    <div style="font-family:'DM Serif Display',serif;font-size:1.64rem;color:#f1f5f9;">Прогноз опадів</div>
</div>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.info("Спочатку навчіть модель на кроці 2.")
else:
    df_f  = st.session_state["df_feat"]
    fcols = st.session_state["feature_cols"]
    res   = st.session_state["results"]
    dates = df_f["date"].dt.date.tolist()

    pc1, pc2 = st.columns([2,1])
    with pc1:
        fd = st.selectbox("📅 Дата прогнозу", options=dates, index=len(dates)-1)
    with pc2:
        am = list(res.keys())[0]
        if len(res)>1:
            am = st.radio("Модель", list(res.keys()), horizontal=True)

    if st.button("🔮  Зробити прогноз", use_container_width=True):
        row = df_f[df_f["date"].dt.date == fd]
        if row.empty:
            st.warning("Дату не знайдено.")
        else:
            m  = res[am]["model"]
            sc = res[am]["scaler"]
            Xs = sc.transform(row[fcols].values)
            pred = m.predict(Xs)[0]
            prob = m.predict_proba(Xs)[0]

            p_rain = prob[1]*100
            p_dry  = prob[0]*100

            # Велика картка результату
            if pred == 1:
                bg      = "linear-gradient(135deg, #0c2340 0%, #0a1628 100%)"
                accent  = "#0ea5e9"
                glow    = "rgba(14,165,233,0.15)"
                emoji   = "☔"
                label   = "Очікуються опади"
                pct_val = p_rain
            else:
                bg      = "linear-gradient(135deg, #0a2218 0%, #071a12 100%)"
                accent  = "#22c55e"
                glow    = "rgba(34,197,94,0.15)"
                emoji   = "☀️"
                label   = "Опадів не очікується"
                pct_val = p_dry

            st.markdown(f"""
            <div style="background:{bg};border:1px solid {accent}33;border-radius:20px;
                        padding:2.5rem;text-align:center;margin:1.5rem 0;
                        box-shadow:0 0 60px {glow};">
                <div style="font-size:3.5rem;margin-bottom:0.8rem;">{emoji}</div>
                <div style="font-family:'DM Serif Display',serif;font-size:2.19rem;
                            color:{accent};margin-bottom:0.5rem;">{label}</div>
                <div style="color:#475569;font-size:1.09rem;margin-bottom:1.5rem;">{str(fd)} · {am}</div>
                <div style="display:inline-flex;align-items:center;gap:8px;
                            background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
                            border-radius:50px;padding:8px 20px;">
                    <span style="color:#475569;font-size:1.04rem;">Ймовірність:</span>
                    <span style="color:{accent};font-size:1.59rem;font-weight:600;">{pct_val:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Детальні метрики
            dc1,dc2,dc3 = st.columns(3)
            with dc1: st.metric("☔ Ймовірність опадів",     f"{p_rain:.1f}%")
            with dc2: st.metric("☀️ Ймовірність без опадів", f"{p_dry:.1f}%")
            with dc3:
                actual = df_f[df_f["date"].dt.date==fd]["rain_label"].values[0]
                st.metric("📋 Фактично (датасет)", "Опади були" if actual==1 else "Без опадів")

# ══════════════════════════════════════════
# Футер
# ══════════════════════════════════════════
st.markdown("""
<div style="margin-top:4rem;padding:1.5rem 0;border-top:1px solid rgba(255,255,255,0.05);
            text-align:center;">
    <span style="color:#1e293b;font-size:0.78rem;letter-spacing:0.05em;">
        RAINCAST ML · Open-Meteo Archive API · scikit-learn · Streamlit \ Voilenko Yegor 472 group
    </span>
</div>
""", unsafe_allow_html=True)
