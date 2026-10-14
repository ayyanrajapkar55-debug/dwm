import sqlite3
import joblib
import pandas as pd
import numpy as np
import streamlit as st

DB_PATH = "warehouse.db"
MODEL_PATH = "trained_models.joblib"

st.set_page_config(
    page_title="Multi-Source Precision Agriculture Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4f5b66;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-left: 5px solid #2e7d32;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #37474f;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

models_data = load_models()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/tractor.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Module",
    [
        "🏛️ Warehouse & Schema",
        "🔄 OLAP Multidimensional Cube",
        "🤖 Data Mining & Predictions",
        "🌾 Cluster & Zone Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data Warehousing & Data Mining Mini-Project")
st.sidebar.caption("Academic Year 2026–2027")

# ==============================================================================
# PAGE 1: WAREHOUSE & SCHEMA OVERVIEW
# ==============================================================================
if page == "🏛️ Warehouse & Schema":
    st.markdown('<p class="main-header">🌾 Multi-Source Precision Agriculture Warehouse</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Centralized Star-Schema Analytical Warehouse integrating Crop, Weather, Soil & Market Data</p>', unsafe_allow_html=True)

    with get_db_connection() as conn:
        kpi_query = """
        SELECT 
            COUNT(f.fact_id) as total_facts,
            ROUND(SUM(f.production_tonnes), 1) as total_prod,
            ROUND(AVG(f.yield_t_ha), 2) as avg_yield,
            ROUND(AVG(f.rainfall_mm), 1) as avg_rain,
            ROUND(AVG(f.market_price_per_quintal), 1) as avg_price
        FROM fact_agriculture f
        """
        kpi = pd.read_sql_query(kpi_query, conn).iloc[0]
        crops_count = pd.read_sql_query("SELECT COUNT(*) as c FROM dim_crop", conn).iloc[0]["c"]
        dist_count = pd.read_sql_query("SELECT COUNT(*) as c FROM dim_location", conn).iloc[0]["c"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Fact Records</div><div class="metric-val">{int(kpi["total_facts"]):,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Production</div><div class="metric-val">{int(kpi["total_prod"]):,} t</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Average Yield</div><div class="metric-val">{kpi["avg_yield"]} t/ha</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Coverage</div><div class="metric-val">{dist_count} Dist / {crops_count} Crops</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Mandi Price</div><div class="metric-val">₹{kpi["avg_price"]:,.0f} /q</div></div>', unsafe_allow_html=True)

    st.markdown("### 📐 Star Schema Architecture")
    col_schema, col_details = st.columns([1, 1])

    with col_schema:
        st.info("""
        **Star Schema Design:**
        - **Central Fact Table**: `fact_agriculture` (Measures: Area, Production, Yield, Rainfall, Temp, Humidity, Market Price)
        - **Dimension 1**: `dim_time` (Time hierarchy: Year, Season, Quarter)
        - **Dimension 2**: `dim_location` (Geographical hierarchy: Zone, State, District)
        - **Dimension 3**: `dim_crop` (Crop hierarchy: Category, Crop Name)
        - **Dimension 4**: `dim_soil` (Soil properties: Soil Type, N, P, K, pH)
        """)

    with col_details:
        table_selected = st.selectbox(
            "Inspect Warehouse Table",
            ["fact_agriculture", "dim_crop", "dim_location", "dim_time", "dim_soil"]
        )
        with get_db_connection() as conn:
            sample_df = pd.read_sql_query(f"SELECT * FROM {table_selected} LIMIT 10", conn)
            st.dataframe(sample_df, width="stretch")

# ==============================================================================
# PAGE 2: OLAP MULTIDIMENSIONAL CUBE
# ==============================================================================
elif page == "🔄 OLAP Multidimensional Cube":
    st.markdown('<p class="main-header">🔄 OLAP Multidimensional Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Roll-up, Drill-down, Slice, Dice & Pivot operations on the Star Schema</p>', unsafe_allow_html=True)

    olap_tab1, olap_tab2, olap_tab3, olap_tab4, olap_tab5 = st.tabs([
        "1. Roll-up", "2. Drill-down", "3. Slice", "4. Dice", "5. Pivot Table"
    ])

    with get_db_connection() as conn:
        all_states = sorted(pd.read_sql_query("SELECT DISTINCT state FROM dim_location", conn)["state"].tolist())
        all_crops = sorted(pd.read_sql_query("SELECT DISTINCT crop_name FROM dim_crop", conn)["crop_name"].tolist())
        all_zones = sorted(pd.read_sql_query("SELECT DISTINCT zone FROM dim_location", conn)["zone"].tolist())
        all_seasons = sorted(pd.read_sql_query("SELECT DISTINCT season FROM dim_time", conn)["season"].tolist())

    # 1. Roll-up
    with olap_tab1:
        st.subheader("⬆️ Roll-up: Aggregating from District to State & Zone")
        st.caption("Climbing up the geographical hierarchy to compare agricultural performance across zones and states.")
        with get_db_connection() as conn:
            rollup_df = pd.read_sql_query("""
            SELECT 
                l.zone AS Zone,
                l.state AS State,
                COUNT(DISTINCT l.district) AS Districts,
                ROUND(SUM(f.production_tonnes), 1) AS Total_Production_Tonnes,
                ROUND(AVG(f.yield_t_ha), 2) AS Avg_Yield_T_HA,
                ROUND(AVG(f.rainfall_mm), 1) AS Avg_Rainfall_MM
            FROM fact_agriculture f
            JOIN dim_location l ON f.location_id = l.location_id
            GROUP BY l.zone, l.state
            ORDER BY Total_Production_Tonnes DESC
            """, conn)
        
        st.dataframe(rollup_df, width="stretch")
        st.bar_chart(rollup_df.set_index("State")["Total_Production_Tonnes"])

    # 2. Drill-down
    with olap_tab2:
        st.subheader("⬇️ Drill-down: Stepping down from State to District & Crops")
        selected_state = st.selectbox("Select State to Drill Down", all_states, index=0)
        with get_db_connection() as conn:
            drill_df = pd.read_sql_query("""
            SELECT 
                l.district AS District,
                c.crop_name AS Crop,
                t.year AS Year,
                t.season AS Season,
                f.area_hectares AS Area_HA,
                f.production_tonnes AS Production_Tonnes,
                f.yield_t_ha AS Yield_T_HA,
                f.market_price_per_quintal AS Market_Price_INR
            FROM fact_agriculture f
            JOIN dim_location l ON f.location_id = l.location_id
            JOIN dim_crop c ON f.crop_id = c.crop_id
            JOIN dim_time t ON f.time_id = t.time_id
            WHERE l.state = ?
            ORDER BY l.district, t.year DESC
            """, conn, params=[selected_state])
        
        st.dataframe(drill_df, width="stretch")

    # 3. Slice
    with olap_tab3:
        st.subheader("🔪 Slice: Filtering along 1 Single Dimension")
        selected_crop = st.selectbox("Select Crop to Slice", all_crops, index=0)
        with get_db_connection() as conn:
            slice_df = pd.read_sql_query("""
            SELECT 
                t.year AS Year,
                t.season AS Season,
                l.state AS State,
                l.district AS District,
                c.crop_name AS Crop,
                f.yield_t_ha AS Yield_T_HA,
                f.rainfall_mm AS Rainfall_MM,
                f.avg_temp_c AS Temp_C,
                f.market_price_per_quintal AS Market_Price_INR
            FROM fact_agriculture f
            JOIN dim_crop c ON f.crop_id = c.crop_id
            JOIN dim_time t ON f.time_id = t.time_id
            JOIN dim_location l ON f.location_id = l.location_id
            WHERE c.crop_name = ?
            ORDER BY t.year DESC, f.yield_t_ha DESC
            """, conn, params=[selected_crop])
        
        st.dataframe(slice_df, width="stretch")

    # 4. Dice
    with olap_tab4:
        st.subheader("🎲 Dice: Multi-dimensional Sub-Cube Filtering")
        c1, c2, c3 = st.columns(3)
        with c1:
            dice_crop = st.selectbox("Crop", all_crops, index=1)
        with c2:
            dice_season = st.selectbox("Season", all_seasons, index=0)
        with c3:
            dice_zone = st.selectbox("Zone", all_zones, index=0)

        with get_db_connection() as conn:
            dice_df = pd.read_sql_query("""
            SELECT 
                t.year AS Year,
                l.state AS State,
                l.district AS District,
                c.crop_name AS Crop,
                t.season AS Season,
                l.zone AS Zone,
                f.production_tonnes AS Production_Tonnes,
                f.yield_t_ha AS Yield_T_HA,
                f.rainfall_mm AS Rainfall_MM
            FROM fact_agriculture f
            JOIN dim_crop c ON f.crop_id = c.crop_id
            JOIN dim_time t ON f.time_id = t.time_id
            JOIN dim_location l ON f.location_id = l.location_id
            WHERE c.crop_name = ? AND t.season = ? AND l.zone = ?
            ORDER BY t.year DESC
            """, conn, params=[dice_crop, dice_season, dice_zone])

        if dice_df.empty:
            st.warning("No records found for this specific multi-dimensional combination.")
        else:
            st.dataframe(dice_df, width="stretch")

    # 5. Pivot Table
    with olap_tab5:
        st.subheader("📊 Pivot: Cross-Tabulation Matrix (Crop vs Year Yield)")
        with get_db_connection() as conn:
            p_df = pd.read_sql_query("""
            SELECT 
                c.crop_name AS Crop,
                t.year AS Year,
                ROUND(AVG(f.yield_t_ha), 2) AS Avg_Yield
            FROM fact_agriculture f
            JOIN dim_crop c ON f.crop_id = c.crop_id
            JOIN dim_time t ON f.time_id = t.time_id
            GROUP BY c.crop_name, t.year
            """, conn)
            pivot_matrix = p_df.pivot(index="Crop", columns="Year", values="Avg_Yield")
        st.dataframe(pivot_matrix, width="stretch")

# ==============================================================================
# PAGE 3: DATA MINING & PREDICTIONS
# ==============================================================================
elif page == "🤖 Data Mining & Predictions":
    st.markdown('<p class="main-header">🤖 Data Mining & Predictive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Trained Machine Learning Models for Crop Yield Forecasting and Intelligent Crop Suitability Recommendation</p>', unsafe_allow_html=True)

    pred_tab1, pred_tab2 = st.tabs(["🌱 Crop Yield Forecasting (Regression)", "🎯 Crop Suitability Recommendation (Classification)"])

    # 1. Yield Forecasting
    with pred_tab1:
        st.subheader("🌾 Crop Yield Regressor (Random Forest)")
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.success(f"Model Metric: **R² Score = {models_data['reg_metrics']['r2']:.4f}** (RMSE = {models_data['reg_metrics']['rmse']:.2f})")
        
        st.write("Adjust environmental conditions to predict agricultural output:")

        yc1, yc2, yc3 = st.columns(3)
        with yc1:
            p_crop = st.selectbox("Crop to Forecast", models_data["all_crops"], index=0)
            p_area = st.slider("Cultivated Area (Hectares)", 100, 15000, 2500, step=100)
            p_rain = st.slider("Seasonal Rainfall (mm)", 100, 2000, 800, step=25)
        with yc2:
            p_temp = st.slider("Average Temperature (°C)", 12.0, 42.0, 26.0, step=0.5)
            p_humid = st.slider("Relative Humidity (%)", 30, 95, 65, step=1)
            p_ph = st.slider("Soil pH Level", 5.0, 8.5, 6.8, step=0.1)
        with yc3:
            p_n = st.slider("Soil Nitrogen (N)", 10, 200, 80, step=5)
            p_p = st.slider("Soil Phosphorus (P)", 10, 100, 45, step=5)
            p_k = st.slider("Soil Potassium (K)", 10, 100, 40, step=5)

        if st.button("🚀 Forecast Yield & Production", type="primary"):
            # Construct feature vector
            feat_dict = {col: 0.0 for col in models_data["reg_feature_columns"]}
            feat_dict["nitrogen_n"] = p_n
            feat_dict["phosphorus_p"] = p_p
            feat_dict["potassium_k"] = p_k
            feat_dict["ph"] = p_ph
            feat_dict["rainfall_mm"] = p_rain
            feat_dict["avg_temp_c"] = p_temp
            feat_dict["humidity_pct"] = p_humid
            feat_dict["area_hectares"] = p_area

            # Set dummy variable for the crop
            crop_col = f"crop_name_{p_crop}"
            if crop_col in feat_dict:
                feat_dict[crop_col] = 1.0

            input_df = pd.DataFrame([feat_dict])
            pred_yield = float(models_data["reg_model"].predict(input_df)[0])
            pred_prod = round(pred_yield * p_area, 1)

            st.balloons()
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric("Predicted Yield", f"{pred_yield:.2f} t/ha")
            with res_c2:
                st.metric("Total Estimated Production", f"{pred_prod:,.1f} tonnes")

    # 2. Crop Suitability Recommendation
    with pred_tab2:
        st.subheader("🎯 Intelligent Crop Recommendation (Classifier)")
        st.success(f"Model Metric: **Test Accuracy = {models_data['clf_metrics']['accuracy'] * 100:.2f}%**")
        st.write("Enter the local soil and weather profile to find the most suitable crop to cultivate:")

        rc1, rc2 = st.columns(2)
        with rc1:
            c_n = st.slider("Nitrogen (N) Content", 10, 200, 95, step=5, key="c_n")
            c_p = st.slider("Phosphorus (P) Content", 10, 100, 50, step=5, key="c_p")
            c_k = st.slider("Potassium (K) Content", 10, 100, 45, step=5, key="c_k")
            c_ph = st.slider("Soil pH", 5.0, 8.5, 6.7, step=0.1, key="c_ph")
        with rc2:
            c_rain = st.slider("Annual / Seasonal Rainfall (mm)", 100, 2000, 450, step=25, key="c_rain")
            c_temp = st.slider("Mean Temperature (°C)", 10.0, 42.0, 21.0, step=0.5, key="c_temp")
            c_humid = st.slider("Humidity (%)", 30, 95, 55, step=1, key="c_humid")

        if st.button("🔍 Recommend Best Crop", type="primary"):
            c_feats = pd.DataFrame([[c_n, c_p, c_k, c_ph, c_rain, c_temp, c_humid]], columns=models_data["clf_feature_columns"])
            rec_crop = models_data["clf_model"].predict(c_feats)[0]
            probs = models_data["clf_model"].predict_proba(c_feats)[0]
            classes = models_data["clf_model"].classes_

            st.markdown(f"### Recommended Crop: **🌾 {rec_crop}**")
            prob_df = pd.DataFrame({
                "Crop": classes,
                "Suitability Score (%)": [round(p * 100, 1) for p in probs]
            }).sort_values(by="Suitability Score (%)", ascending=False)
            st.dataframe(prob_df, width="stretch")

# ==============================================================================
# PAGE 4: CLUSTER & ZONE INSIGHTS
# ==============================================================================
elif page == "🌾 Cluster & Zone Insights":
    st.markdown('<p class="main-header">🌾 Agricultural Clustering & Zone Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Unsupervised K-Means Clustering for Agro-Ecological Profiling and Risk Segmentation</p>', unsafe_allow_html=True)

    with get_db_connection() as conn:
        cluster_df = pd.read_sql_query("""
        SELECT 
            f.fact_id,
            l.state,
            l.district,
            l.zone,
            c.crop_name,
            s.soil_type,
            s.nitrogen_n,
            s.phosphorus_p,
            s.potassium_k,
            f.rainfall_mm,
            f.avg_temp_c,
            f.yield_t_ha
        FROM fact_agriculture f
        JOIN dim_location l ON f.location_id = l.location_id
        JOIN dim_crop c ON f.crop_id = c.crop_id
        JOIN dim_soil s ON f.soil_id = s.soil_id
        """, conn)

    # Scale and predict cluster
    X_scale = models_data["cluster_scaler"].transform(cluster_df[models_data["cluster_features"]])
    cluster_df["Cluster"] = models_data["kmeans_model"].predict(X_scale)
    cluster_label_map = {
        0: "Cluster 0: Moderate-Rainfall High-Productivity Zones",
        1: "Cluster 1: High-Yield Intensive Commercial Agriculture",
        2: "Cluster 2: Low-Rainfall Arid / Rabi Belt"
    }
    cluster_df["Cluster_Profile"] = cluster_df["Cluster"].map(cluster_label_map)

    st.subheader("Agro-Climatic Clusters Summary")
    summary = cluster_df.groupby("Cluster_Profile")[["yield_t_ha", "rainfall_mm", "avg_temp_c", "nitrogen_n"]].mean().reset_index()
    st.dataframe(summary, width="stretch")

    st.subheader("Clustering Distribution (Rainfall vs Average Yield)")
    st.scatter_chart(data=cluster_df, x="rainfall_mm", y="yield_t_ha", color="Cluster_Profile")

    st.subheader("Cluster Membership by District")
    district_view = cluster_df.groupby(["state", "district", "Cluster_Profile"]).size().reset_index(name="Sample_Records")
    st.dataframe(district_view, width="stretch")
