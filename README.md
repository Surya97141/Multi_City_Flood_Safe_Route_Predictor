# 🌊 Multi-City Disaster Safe Route Predictor

> A comprehensive disaster-aware navigation and prediction system covering **floods, cyclones, earthquakes, and landslides** across multiple Indian cities. Built as a modular ML pipeline with real-time safe route planning and interactive dashboards.

---

## 🎯 Overview

Natural disasters cause immense damage due to delayed response and poor route planning. This project addresses that by building an **end-to-end multi-disaster prediction and safe route planning system** that:

- Predicts risk levels for **4 disaster types**: Flood, Cyclone, Earthquake, Landslide
- Plans **safe evacuation routes** using elevation data across multiple cities
- Provides an **interactive Streamlit dashboard** for real-time visualization
- Exposes predictions via a **REST API** for integration with other systems

Originally developed for **Smart India Hackathon (SIH) 2025**.

---

## 🗂️ Project Structure

```
├── api/
│   └── flood_api.py                  # REST API for flood predictions
│
├── app/
│   ├── safe_zones_delhi_elevation.csv
│   ├── safe_zones_mumbai_elevation.csv
│   └── streamlit_flood_dashboard.py  # Interactive Streamlit dashboard
│
├── config/                           # Configuration files
│
├── data/
│   ├── external/                     # External data sources
│   ├── processed/                    # Cleaned & feature-engineered data
│   └── raw/                          # Original raw datasets
│
├── notebooks/
│   ├── cyclone/
│   ├── earthquake/
│   ├── flood/
│   ├── landslide/
│   ├── 01_EDA_Flood.ipynb
│   ├── 02_EDA_Cyclone.ipynb
│   ├── 03_EDA_Landslide.ipynb
│   ├── 04_EDA_Earthquake.ipynb
│   ├── 05_Create_Flood_Model.ipynb
│   ├── 05_Safe_Route_Map.ipynb
│   ├── 06_audit_new.ipynb
│   └── 07_Flood_Model.ipynb
│
├── src/
│   ├── common/                       # Shared utilities
│   ├── cyclone/                      # Cyclone prediction module
│   ├── earthquake/                   # Earthquake prediction module
│   ├── flood/                        # Flood prediction module
│   ├── landslide/                    # Landslide prediction module
│   ├── pipeline/                     # ML training pipelines
│   └── route_planner/                # Safe route planning logic
│
├── models/                           # Saved trained models (.pkl)
├── logs/                             # Training and inference logs
├── scripts/                          # Utility and automation scripts
├── resources/                        # Supporting assets
├── tests/                            # Unit and integration tests
├── safe_zones_with_elevation.csv     # Master safe zones dataset
├── requirements.txt
└── README.md
```

---

## 🧠 Disaster Modules

### 🌊 Flood Prediction
- Random Forest model trained on historical flood data
- Features include rainfall, river levels, soil saturation, and elevation
- Safe zone mapping for Delhi and Mumbai using elevation CSVs
- Streamlit dashboard for real-time flood risk visualization

### 🌀 Cyclone Prediction
- EDA and modeling pipeline for cyclone intensity and landfall prediction
- Feature engineering on wind speed, pressure, and sea surface temperature

### 🏔️ Earthquake Risk
- Risk scoring based on seismic zone data and historical magnitude patterns
- City-level risk classification

### 🌧️ Landslide Risk
- Prediction based on slope, rainfall intensity, and soil type features
- Integrated into the unified route planner

---

## 🗺️ Safe Route Planner

The `route_planner` module combines outputs from all four disaster models to:
- Identify **high-risk zones** in real time
- Compute **safest evacuation paths** using elevation and risk scores
- Dynamically reroute users away from active disaster zones

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML Models | Random Forest, scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Streamlit, Matplotlib |
| API | Flask / FastAPI (`flood_api.py`) |
| Notebooks | Jupyter |
| Environment | Python venv (`gbm_sih_env`) |
| Version Control | Git |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Surya97141/<repo-name>.git
cd <repo-name>

# Create and activate virtual environment
python -m venv gbm_sih_env
source gbm_sih_env/bin/activate      # Linux/Mac
gbm_sih_env\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the App

### Streamlit Dashboard
```bash
streamlit run app/streamlit_flood_dashboard.py
```

### Flood Prediction API
```bash
python api/flood_api.py
```

### Run Notebooks
Open any notebook in `notebooks/` folder — start with `01_EDA_Flood.ipynb` for an overview.

---

## ⚠️ Note on Dataset Files

The raw training datasets are **not included** in this repository due to their large size (several hundred MBs), which exceeds GitHub's file size limits. Only pre-trained model files and curated safe zone CSVs are present.

To reproduce model training from scratch, source the datasets from:
- [India Meteorological Department (IMD)](https://mausam.imd.gov.in)
- [National Disaster Management Authority (NDMA)](https://ndma.gov.in)
- [USGS Earthquake Data](https://earthquake.usgs.gov/earthquakes/feed/)
- [Kaggle Disaster Datasets](https://www.kaggle.com/search?q=flood+cyclone+earthquake+landslide)

Once downloaded, place raw files inside `data/raw/` and run the respective notebooks.

---

## 📊 Model Performance

| Disaster | Model | Key Metric |
|----------|-------|-----------|
| Flood | Random Forest | F1-score, ROC-AUC |
| Cyclone | In progress | — |
| Earthquake | In progress | — |
| Landslide | In progress | — |

---

## 🔮 Future Scope

- Real-time data ingestion from IMD and NDMA APIs
- Mobile-friendly interface for field responders
- Integration with Google Maps API for live route navigation
- Unified multi-disaster risk score per location

---

## 👤 Author

**Surya Pratap Singh**
B.Tech CSE (Data Science) — Galgotias College of Engineering & Technology
[GitHub](https://github.com/Surya97141) · [LinkedIn](https://www.linkedin.com/in/surya-singh-b575591b5)

---

> Built with ❤️ for Smart India Hackathon 2025 — because better data saves lives.
