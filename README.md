# Multi-Source Precision Agriculture Intelligence
*Data Warehousing & Data Mining Mini-Project*

An end-to-end, lightweight analytical data warehouse and data mining pipeline with an interactive web dashboard for precision agriculture.

---

## 🌟 Highlights & Features
1. **Multi-Source Data Integration**: Ingests and cleans heterogeneous agricultural sources including Crop Production, Weather, Soil Health, and Mandi/Market Prices.
2. **Star Schema Data Warehouse**: Implemented in SQLite (`warehouse.db`) with:
   - **Fact Table**: `fact_agriculture`
   - **Dimension Tables**: `dim_time`, `dim_location`, `dim_crop`, `dim_soil`
3. **OLAP Operations**: Demonstrates **Roll-up**, **Drill-down**, **Slice**, **Dice**, and **Pivot** with interactive multidimensional views.
4. **Data Mining & Machine Learning**:
   - **Crop Yield Regressor (Random Forest)**: Forecasts yield ($t/ha$) and total production ($t$) based on climate, soil NPK, and area ($R^2 \approx 0.998$).
   - **Intelligent Crop Recommendation (Random Forest Classifier)**: Recommends the optimal crop for a given soil/weather profile ($96.9\%$ accuracy).
   - **Agro-Ecological Clustering (K-Means)**: Segments agricultural districts into performance & risk profiles.
5. **Interactive Web Dashboard**: Streamlit web interface with zero desktop/GUI clutter.

---

## 📁 Repository Structure
```text
ayan-dwm/
├── data/                                 # Raw multi-source CSV files
│   ├── raw_crop_production.csv
│   ├── raw_weather.csv
│   ├── raw_soil.csv
│   └── raw_market.csv
├── warehouse.db                          # Star-schema SQLite Data Warehouse
├── generate_data.py                      # Data generation and agronomic modeling
├── etl.py                                # Data cleaning, transformation, and warehouse loading
├── olap.py                               # SQL OLAP scripts (Roll-up, Drill-down, Slice, Dice)
├── ml_models.py                          # ML model training and evaluation
├── trained_models.joblib                 # Serialized ML models & metrics
├── app.py                                # Streamlit Web Dashboard
├── requirements.txt                      # Python dependencies
├── run.sh                                # 1-Click project runner
└── musfrMulti_Source_Precision_Agriculture_Intelligence_Report.docx
```

---

## 🚀 Quick Start

### 1. One-Click Run
Simply run:
```bash
./run.sh
```
This script handles environment setup, data generation, ETL, model training, and launches the web app automatically at `http://localhost:8501`.

### 2. Manual Step-by-Step Execution
If you prefer running individual components:

```bash
# 1. Activate environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate raw multi-source datasets
python3 generate_data.py

# 3. Run ETL pipeline & build warehouse.db
python3 etl.py

# 4. Run terminal OLAP queries
python3 olap.py

# 5. Train Data Mining & ML models
python3 ml_models.py

# 6. Launch the Web Dashboard
streamlit run app.py
```

---

## 🌐 Web Dashboard Modules

* **🏛️ Warehouse & Schema**: View high-level KPIs, examine the Star Schema structure, and inspect raw warehouse tables.
* **🔄 OLAP Multidimensional Cube**: Interactive tabs for Roll-up, Drill-down by State, Single-Dimension Slicing, Multi-Dimension Dicing, and Yearly Pivot Cross-Tabulations.
* **🤖 Data Mining & Predictions**:
  - Live interactive sliders for rainfall, temperature, soil nutrients ($N, P, K, pH$) to predict crop yield.
  - Soil & climate input forms for intelligent crop suitability recommendation with class probabilities.
* **🌾 Cluster & Zone Insights**: K-Means clustering scatter charts and agro-climatic profile summaries.
