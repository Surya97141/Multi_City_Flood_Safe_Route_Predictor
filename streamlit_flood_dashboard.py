import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests
import pydeck as pdk
import time
from math import radians, cos, sin, sqrt, atan2

CLEAN_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&family=Oswald:wght@500;700&display=swap" rel="stylesheet">
<style>
body { background: #23252B; }
.block-container { background: #F2F3F7 !important; border-radius: 11px !important; color: #111312; font-family: 'Barlow', 'Segoe UI', Arial, sans-serif; padding-left: 30px !important; padding-right: 30px !important;}
h1 { font-family: 'Oswald', Arial, 'Segoe UI', sans-serif !important; font-weight: 700 !important; text-transform: uppercase; color: #232830; letter-spacing: .03em !important; margin-top: .3em; margin-bottom: .25em;}
.stNumberInput label, .stTextInput label, .stSelectbox label { color: #21292e !important; font-weight: 600 !important; font-size: 1.02rem; }
.stNumberInput label div, .stTextInput label div, .stSelectbox label div { color: #232830 !important;}
.stButton>button, .stDownloadButton>button {
  background: #375284; color: #fff; border-radius: 8px; font-size: 1rem;
  font-family: 'Barlow', Arial, sans-serif; font-weight: 600; padding: 10px 24px;}
.stNumberInput, .stTextInput, .stSelectbox { background: #d5deea !important; }
</style>
"""
st.set_page_config(layout="wide", page_title="Flood Risk Dashboard")
st.markdown(CLEAN_CSS, unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')
model = joblib.load(os.path.join(models_dir, 'flood_rf_model.pkl'))
features = joblib.load(os.path.join(models_dir, 'flood_rf_features.pkl'))

# ---- MULTI-CITY DATA ----
CITIES_DATA = {
    "Mumbai": {
        "flood_areas": pd.DataFrame({
            "name": ["Mumbai Central", "Kurla", "Sion", "Bandra East"],
            "lat": [19.0176, 19.0728, 19.0457, 19.0544],
            "lon": [72.8562, 72.8826, 72.8686, 72.8407],
            "risk": [0.95, 0.85, 0.9, 0.83],
            "Population": [180000, 320000, 150000, 125000],
            "DFSI": [15, 11, 12, 10]
        }),
        "safe_zones": pd.DataFrame({
            "name": ["Siddhivinayak Temple", "NESCO Goregaon", "VJTI College", "Malabar Hill", "Borivali Gate"],
            "lat": [19.0176, 19.1517, 19.0173, 18.9718, 19.2168],
            "lon": [72.8305, 72.8491, 72.8562, 72.7956, 72.8625],
            "country": ["India"]*5
        }),
        "center_lat": 19.076, "center_lon": 72.8776, "zoom": 11
    },
    "Bangalore": {
        "flood_areas": pd.DataFrame({
            "name": ["Bangalore Central", "Marathahalli", "Indiranagar", "Yeshwanthpur"],
            "lat": [12.9716, 12.9689, 12.9716, 13.0013],
            "lon": [77.5946, 77.7499, 77.6412, 77.5706],
            "risk": [0.88, 0.79, 0.84, 0.76],
            "Population": [200000, 150000, 180000, 140000],
            "DFSI": [12, 9, 11, 8]
        }),
        "safe_zones": pd.DataFrame({
            "name": ["Vidhana Soudha", "Cubbon Park", "Bangalore Palace", "Lal Bagh", "Trinity Church"],
            "lat": [12.9901, 12.9729, 13.0013, 12.9352, 12.9789],
            "lon": [77.5905, 77.5917, 77.5706, 77.5920, 77.6245],
            "country": ["India"]*5
        }),
        "center_lat": 12.9716, "center_lon": 77.5946, "zoom": 11
    },
    "Delhi": {
        "flood_areas": pd.DataFrame({
            "name": ["Central Delhi", "East Delhi", "South Delhi", "West Delhi"],
            "lat": [28.6139, 28.5931, 28.5244, 28.6431],
            "lon": [77.2090, 77.3149, 77.1855, 77.0521],
            "risk": [0.92, 0.87, 0.80, 0.85],
            "Population": [250000, 200000, 180000, 160000],
            "DFSI": [14, 13, 10, 12]
        }),
        "safe_zones": pd.DataFrame({
            "name": ["Red Fort", "India Gate", "Rajpath", "Talkatora Stadium", "Delhi Zoo"],
            "lat": [28.6562, 28.6129, 28.6061, 28.5769, 28.6163],
            "lon": [77.2410, 77.1892, 77.1954, 77.2324, 77.2546],
            "country": ["India"]*5
        }),
        "center_lat": 28.6139, "center_lon": 77.2090, "zoom": 10
    },
    "Kolkata": {
        "flood_areas": pd.DataFrame({
            "name": ["Kolkata Central", "Howrah", "Dakshin", "North Kolkata"],
            "lat": [22.5726, 22.5891, 22.5244, 22.6274],
            "lon": [88.3639, 88.2676, 88.3617, 88.4043],
            "risk": [0.96, 0.91, 0.86, 0.89],
            "Population": [220000, 190000, 140000, 170000],
            "DFSI": [16, 14, 11, 13]
        }),
        "safe_zones": pd.DataFrame({
            "name": ["Victoria Memorial", "Indian Museum", "Jadavpur University", "Fort William", "Kalighat Temple"],
            "lat": [22.5448, 22.5597, 22.4984, 22.5667, 22.5126],
            "lon": [88.3426, 88.3910, 88.3892, 88.3704, 88.3680],
            "country": ["India"]*5
        }),
        "center_lat": 22.5726, "center_lon": 88.3639, "zoom": 11
    },
    "Chennai": {
        "flood_areas": pd.DataFrame({
            "name": ["Chennai Central", "Mylapore", "Anna Nagar", "Guindy"],
            "lat": [13.0827, 13.0312, 13.0830, 13.0020],
            "lon": [80.2707, 80.2706, 80.2108, 80.2156],
            "risk": [0.91, 0.85, 0.82, 0.79],
            "Population": [210000, 160000, 140000, 130000],
            "DFSI": [13, 11, 10, 9]
        }),
        "safe_zones": pd.DataFrame({
            "name": ["Fort St. George", "Kapaleeshwarar Temple", "San Thome Basilica", "Parthasarathy Temple", "Marina Beach"],
            "lat": [13.1627, 13.0312, 13.0439, 13.0263, 13.0499],
            "lon": [80.2823, 80.2706, 80.2769, 80.2666, 80.2806],
            "country": ["India"]*5
        }),
        "center_lat": 13.0827, "center_lon": 80.2707, "zoom": 11
    }
}

def get_elevation(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()["results"][0]["elevation"]
    except:
        pass
    return None

def load_elevations(city_name, safe_zones):
    elev_file = f"safe_zones_{city_name.lower()}_elevation.csv"
    if os.path.exists(elev_file):
        return pd.read_csv(elev_file)
    else:
        with st.spinner(f"Fetching elevations for {city_name}..."):
            elevations = []
            for _, row in safe_zones.iterrows():
                elev = get_elevation(row["lat"], row["lon"])
                elevations.append(elev if elev is not None else -999)
                time.sleep(0.5)
            safe_zones["elevation"] = elevations
            safe_zones.to_csv(elev_file, index=False)
        return safe_zones

SAFE_ELEVATION_MIN = 8

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

st.title("Multi-City Flood Risk and Safe Route Dashboard")
st.write("Select a city to view flood-prone areas and nearby high-ground safe zones with evacuation routes.")

st.sidebar.subheader("Select City")
selected_city = st.sidebar.selectbox("City:", list(CITIES_DATA.keys()))

FLOOD_PRONE_AREAS = CITIES_DATA[selected_city]["flood_areas"]
SAFE_ZONES_RAW = CITIES_DATA[selected_city]["safe_zones"]
SAFE_ZONES = load_elevations(selected_city, SAFE_ZONES_RAW.copy())

st.sidebar.subheader("Flood-Prone Area")
flood_area_names = ["(manual entry)"] + FLOOD_PRONE_AREAS["name"].tolist()
selected_flood_area = st.sidebar.selectbox("Flood location:", flood_area_names)

if selected_flood_area != "(manual entry)":
    area_row = FLOOD_PRONE_AREAS[FLOOD_PRONE_AREAS["name"] == selected_flood_area].iloc[0]
    lat = area_row["lat"]
    lon = area_row["lon"]
    country = "India"
    st.sidebar.metric("Historic Risk (%)", f"{area_row['risk']*100:.0f}")
else:
    lat = st.sidebar.number_input("Latitude", value=CITIES_DATA[selected_city]["center_lat"], format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=CITIES_DATA[selected_city]["center_lon"], format="%.6f")
    country = "India"

SAFE_ZONES["dist"] = SAFE_ZONES.apply(lambda row: haversine(lat, lon, row["lat"], row["lon"]), axis=1)
filtered_zones = SAFE_ZONES[(SAFE_ZONES["country"].str.lower() == country.lower())
                            & (SAFE_ZONES["elevation"] >= SAFE_ELEVATION_MIN)
                            & (SAFE_ZONES["dist"] <= 10)].sort_values("dist")

if len(filtered_zones) > 0:
    safe_names = ["(manual entry)"] + [f"{row['name']} ({int(row['elevation'])}m, {row['dist']:.1f}km)" for _, row in filtered_zones.iterrows()]
else:
    safe_names = ["(manual entry)"] + [f"{row['name']} ({int(row['elevation'])}m, {row['dist']:.1f}km)" for _, row in SAFE_ZONES.sort_values("dist").head(3).iterrows()]

st.sidebar.subheader("Nearby Elevated Safe Zones")
selected_safe = st.sidebar.selectbox("Safe Zone:", safe_names)
if selected_safe != "(manual entry)":
    idx = safe_names.index(selected_safe) - 1
    safe_row = filtered_zones.iloc[idx] if len(filtered_zones) > 0 else SAFE_ZONES.sort_values("dist").head(3).iloc[idx]
    safe_lat = safe_row["lat"]
    safe_lon = safe_row["lon"]
    safe_label = f"Safe Zone ({int(safe_row['elevation'])}m)"
else:
    safe_lat = st.sidebar.number_input("Safe Zone Latitude", value=lat + 0.03, format="%.6f")
    safe_lon = st.sidebar.number_input("Safe Zone Longitude", value=lon - 0.03, format="%.6f")
    safe_label = "Safe Zone (manual entry)"

param_cols = st.columns(3)
feature_inputs = {}
for i, feat in enumerate(features):
    if feat.lower() not in ["latitude", "longitude"]:
        with param_cols[i % 3]:
            if selected_flood_area != "(manual entry)" and feat in area_row:
                val = float(area_row[feat]) if pd.notnull(area_row[feat]) else 5.0
            else:
                val = 10.0 if "dfsi" in feat.lower() else 50000.0 if "population" in feat.lower() else 5.0
            feature_inputs[feat] = st.number_input(feat.replace("_", " "), value=val, step=0.1, key=feat, help=f"{feat}")

inputs = {}
for feat in features:
    if feat.lower() == "latitude":
        inputs[feat] = lat
    elif feat.lower() == "longitude":
        inputs[feat] = lon
    else:
        inputs[feat] = feature_inputs.get(feat, 5.0)
input_vector = [inputs[f] for f in features]

if "last_query_time" not in st.session_state: st.session_state["last_query_time"] = 0
COOLDOWN_SECONDS = 15

map_placeholder = st.empty()
analyze = st.button("Analyze & View Evacuation Route", use_container_width=True)
show_map_only = st.checkbox("Show Map Only", value=False)

if analyze:
    now = time.time()
    elapsed = now - st.session_state["last_query_time"]
    if elapsed < COOLDOWN_SECONDS and not show_map_only:
        st.warning(f"Please wait {int(COOLDOWN_SECONDS - elapsed)} seconds before getting a new route.")
    else:
        if not show_map_only:
            st.session_state["last_query_time"] = now

        X = np.array(input_vector).reshape(1, -1)
        risk = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][risk])
        panel_bg = '#ffeded' if risk == 1 else '#eaffed'

        st.markdown(
            f"<div style='margin:24px 0 8px 0; padding:18px 12px; background:{panel_bg}; border-radius:8px; "
            f"color:#283346; font-size:1.18rem; font-family:Barlow,Arial,sans-serif;'>"
            f"{'FLOOD RISK DETECTED' if risk==1 else 'NO SIGNIFICANT RISK'} • Confidence: {proba:.2%}"
            "</div>",
            unsafe_allow_html=True
        )

        map_data = [
            {"lat": lat, "lon": lon, "color": [200, 40, 40], "label": "Flood Location", "size": 250},
            {"lat": safe_lat, "lon": safe_lon, "color": [50, 180, 100], "label": safe_label, "size": 215}
        ]
        for _, row in FLOOD_PRONE_AREAS.iterrows():
            if abs(row["lat"] - lat) > .02 or abs(row["lon"] - lon) > .02:
                map_data.append({"lat": row["lat"], "lon": row["lon"], "color": [200, 160, 30], "label": row["name"], "size": 100})
        map_df = pd.DataFrame(map_data)

        if not show_map_only:
            osrm_url = f"https://router.project-osrm.org/route/v1/driving/{lon},{lat};{safe_lon},{safe_lat}?overview=full&geometries=geojson"
            try:
                r = requests.get(osrm_url, timeout=12)
                if r.status_code == 200:
                    route = r.json()["routes"][0]["geometry"]["coordinates"]
                    route_path = [[x[0], x[1]] for x in route]
                    route_layer = pdk.Layer(
                        "PathLayer", [{"path": route_path}],
                        get_path="path", get_color=[60, 145, 240], width_scale=20, width_min_pixels=5
                    )
                else:
                    route_layer = None
            except:
                route_layer = None
        else:
            route_layer = None

        points_layer = pdk.Layer(
            "ScatterplotLayer", map_df,
            get_position=["lon", "lat"],
            get_fill_color="color", get_radius="size", pickable=True,
        )

        view = pdk.ViewState(
            latitude=np.mean([lat, safe_lat]),
            longitude=np.mean([lon, safe_lon]),
            zoom=CITIES_DATA[selected_city]["zoom"],
            pitch=0
        )
        width = 1100; height = 600
        layers = [points_layer] if route_layer is None else [route_layer, points_layer]
        deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"html": "<b>{label}</b> ({lat:.3f}, {lon:.3f})"})
        map_placeholder.pydeck_chart(deck, use_container_width=False, width=width, height=height)

        if risk == 1:
            st.markdown(
                "<div style='background:#ffe5e5;color:#aa2222;padding:18px 20px;border-radius:15px;font-size:1rem;'>"
                "Evacuate to the nearest high-ground safe zone shown above. Help others. Monitor all official updates and rescue orders."
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background:#ddffe6;color:#245730;padding:16px 18px;border-radius:15px;font-size:1rem;'>"
                "Currently safe. Continue to monitor, but no evacuation required at this time."
                "</div>", unsafe_allow_html=True
            )