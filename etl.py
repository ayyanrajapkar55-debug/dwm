import os
import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "warehouse.db"

def run_etl():
    print("--- Starting ETL Pipeline ---")
    
    # 1. Extract
    print("[1/4] Extracting data from CSV files...")
    df_prod = pd.read_csv("data/raw_crop_production.csv")
    df_weather = pd.read_csv("data/raw_weather.csv")
    df_soil = pd.read_csv("data/raw_soil.csv")
    df_market = pd.read_csv("data/raw_market.csv")

    # 2. Transform & Clean
    print("[2/4] Cleaning and validating records...")
    df_prod = df_prod.dropna()
    df_weather = df_weather.dropna()
    df_soil = df_soil.dropna()
    df_market = df_market.dropna()

    df_prod["yield_t_ha"] = np.round(df_prod["production_tonnes"] / df_prod["area_hectares"], 2)

    # 3. Build Dimension Tables
    print("[3/4] Building Star Schema Dimensions...")
    
    # Dim_Time
    time_df = df_prod[["year", "season"]].drop_duplicates().reset_index(drop=True)
    time_df["time_id"] = time_df.index + 1
    season_q_map = {"Kharif": "Q2-Q3", "Rabi": "Q4-Q1", "Annual": "Full Year", "Zaid": "Q2"}
    time_df["quarter"] = time_df["season"].map(season_q_map).fillna("Q1")
    dim_time = time_df[["time_id", "year", "season", "quarter"]]

    # Dim_Location
    loc_df = df_soil[["state", "district", "zone"]].drop_duplicates().reset_index(drop=True)
    loc_df["location_id"] = loc_df.index + 1
    dim_location = loc_df[["location_id", "state", "district", "zone"]]

    # Dim_Crop
    crop_df = df_prod[["crop", "category"]].drop_duplicates().reset_index(drop=True)
    crop_df["crop_id"] = crop_df.index + 1
    crop_df.rename(columns={"crop": "crop_name"}, inplace=True)
    dim_crop = crop_df[["crop_id", "crop_name", "category"]]

    # Dim_Soil
    dim_soil = df_soil[["soil_id", "district", "soil_type", "nitrogen_n", "phosphorus_p", "potassium_k", "ph"]].copy()

    # 4. Assemble Fact Table
    print("[4/4] Joining and constructing Fact_Agriculture_Performance...")
    
    fact_merged = df_prod.merge(
        dim_time, on=["year", "season"], how="inner"
    ).merge(
        dim_location, on=["state", "district"], how="inner"
    ).merge(
        dim_crop, left_on=["crop", "category"], right_on=["crop_name", "category"], how="inner"
    ).merge(
        dim_soil[["soil_id"]], on="soil_id", how="inner"
    )

    # Merge weather metrics
    fact_merged = fact_merged.merge(
        df_weather, on=["year", "season", "state", "district"], how="left"
    )

    # Merge market pricing
    fact_merged = fact_merged.merge(
        df_market, on=["year", "season", "state", "district", "crop"], how="left"
    )

    fact_merged["rainfall_mm"] = fact_merged["rainfall_mm"].fillna(fact_merged["rainfall_mm"].median())
    fact_merged["avg_temp_c"] = fact_merged["avg_temp_c"].fillna(fact_merged["avg_temp_c"].median())
    fact_merged["humidity_pct"] = fact_merged["humidity_pct"].fillna(fact_merged["humidity_pct"].median())
    fact_merged["market_price_per_quintal"] = fact_merged["market_price_per_quintal"].fillna(2500.0)

    fact_merged["fact_id"] = fact_merged.index + 1

    fact_agriculture = fact_merged[[
        "fact_id",
        "time_id",
        "location_id",
        "crop_id",
        "soil_id",
        "area_hectares",
        "production_tonnes",
        "yield_t_ha",
        "rainfall_mm",
        "avg_temp_c",
        "humidity_pct",
        "market_price_per_quintal"
    ]]

    # 5. Load into SQLite Data Warehouse
    print(f"Loading into database '{DB_PATH}'...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE dim_time (
        time_id INTEGER PRIMARY KEY,
        year INTEGER NOT NULL,
        season TEXT NOT NULL,
        quarter TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE dim_location (
        location_id INTEGER PRIMARY KEY,
        state TEXT NOT NULL,
        district TEXT NOT NULL,
        zone TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE dim_crop (
        crop_id INTEGER PRIMARY KEY,
        crop_name TEXT NOT NULL,
        category TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE dim_soil (
        soil_id INTEGER PRIMARY KEY,
        district TEXT NOT NULL,
        soil_type TEXT NOT NULL,
        nitrogen_n INTEGER NOT NULL,
        phosphorus_p INTEGER NOT NULL,
        potassium_k INTEGER NOT NULL,
        ph REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE fact_agriculture (
        fact_id INTEGER PRIMARY KEY,
        time_id INTEGER NOT NULL,
        location_id INTEGER NOT NULL,
        crop_id INTEGER NOT NULL,
        soil_id INTEGER NOT NULL,
        area_hectares REAL NOT NULL,
        production_tonnes REAL NOT NULL,
        yield_t_ha REAL NOT NULL,
        rainfall_mm REAL NOT NULL,
        avg_temp_c REAL NOT NULL,
        humidity_pct REAL NOT NULL,
        market_price_per_quintal REAL NOT NULL,
        FOREIGN KEY (time_id) REFERENCES dim_time(time_id),
        FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
        FOREIGN KEY (crop_id) REFERENCES dim_crop(crop_id),
        FOREIGN KEY (soil_id) REFERENCES dim_soil(soil_id)
    )
    """)

    dim_time.to_sql("dim_time", conn, if_exists="append", index=False)
    dim_location.to_sql("dim_location", conn, if_exists="append", index=False)
    dim_crop.to_sql("dim_crop", conn, if_exists="append", index=False)
    dim_soil.to_sql("dim_soil", conn, if_exists="append", index=False)
    fact_agriculture.to_sql("fact_agriculture", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print(f"ETL Complete! Successfully loaded into {DB_PATH}:")
    print(f" - dim_time: {len(dim_time)} rows")
    print(f" - dim_location: {len(dim_location)} rows")
    print(f" - dim_crop: {len(dim_crop)} rows")
    print(f" - dim_soil: {len(dim_soil)} rows")
    print(f" - fact_agriculture: {len(fact_agriculture)} rows")

if __name__ == "__main__":
    run_etl()
