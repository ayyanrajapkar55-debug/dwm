import sqlite3
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

DB_PATH = "warehouse.db"
MODEL_PATH = "trained_models.joblib"

def get_training_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        f.fact_id,
        c.crop_name,
        c.category,
        s.soil_type,
        s.nitrogen_n,
        s.phosphorus_p,
        s.potassium_k,
        s.ph,
        f.rainfall_mm,
        f.avg_temp_c,
        f.humidity_pct,
        f.area_hectares,
        f.yield_t_ha,
        l.state,
        l.district,
        l.zone
    FROM fact_agriculture f
    JOIN dim_crop c ON f.crop_id = c.crop_id
    JOIN dim_soil s ON f.soil_id = s.soil_id
    JOIN dim_location l ON f.location_id = l.location_id
    JOIN dim_time t ON f.time_id = t.time_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def train_all_models():
    print("--- Training Data Mining & Machine Learning Models ---")
    df = get_training_data()

    # -------------------------------------------------------------
    # 1. Regression: Crop Yield Prediction
    # -------------------------------------------------------------
    # Features for yield prediction
    reg_features = [
        "nitrogen_n", "phosphorus_p", "potassium_k", "ph",
        "rainfall_mm", "avg_temp_c", "humidity_pct", "area_hectares"
    ]
    
    # We include one-hot encoded crop_name to allow accurate yield prediction across various crops
    df_reg = pd.get_dummies(df[reg_features + ["crop_name", "yield_t_ha"]], columns=["crop_name"], drop_first=False)
    X_reg = df_reg.drop(columns=["yield_t_ha"])
    y_reg = df_reg["yield_t_ha"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_r, y_train_r)

    y_pred_r = reg_model.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    print(f"[1/3] Yield Regressor Trained | R² Score: {r2:.4f} | RMSE: {rmse:.4f}")

    # -------------------------------------------------------------
    # 2. Classification: Crop Recommendation / Suitability
    # -------------------------------------------------------------
    clf_features = [
        "nitrogen_n", "phosphorus_p", "potassium_k", "ph",
        "rainfall_mm", "avg_temp_c", "humidity_pct"
    ]
    X_clf = df[clf_features]
    y_clf = df["crop_name"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c, y_train_c)

    y_pred_c = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"[2/3] Crop Classifier Trained | Accuracy: {acc * 100:.2f}%")

    # -------------------------------------------------------------
    # 3. Clustering: Agricultural Zone Clustering (K-Means)
    # -------------------------------------------------------------
    cluster_features = ["rainfall_mm", "avg_temp_c", "yield_t_ha", "nitrogen_n", "phosphorus_p", "potassium_k"]
    cluster_scaler = StandardScaler()
    X_clust_scaled = cluster_scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(X_clust_scaled)

    # Label clusters intuitively
    cluster_summary = df.groupby("cluster_id")[["yield_t_ha", "rainfall_mm", "avg_temp_c"]].mean()
    print("[3/3] K-Means Clustering Trained (k=3)")
    print(cluster_summary)

    # -------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------
    artifacts = {
        "reg_model": reg_model,
        "reg_feature_columns": list(X_reg.columns),
        "reg_metrics": {"r2": float(r2), "rmse": float(rmse)},
        "clf_model": clf_model,
        "clf_feature_columns": clf_features,
        "clf_metrics": {"accuracy": float(acc)},
        "kmeans_model": kmeans,
        "cluster_scaler": cluster_scaler,
        "cluster_features": cluster_features,
        "all_crops": sorted(df["crop_name"].unique().tolist())
    }

    joblib.dump(artifacts, MODEL_PATH)
    print(f"Models successfully saved to '{MODEL_PATH}'")

if __name__ == "__main__":
    train_all_models()
