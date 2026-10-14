import sqlite3
import pandas as pd

DB_PATH = "warehouse.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def olap_rollup():
    """Roll-up: Aggregate yield and production from District up to State and Zone."""
    query = """
    SELECT 
        l.zone,
        l.state,
        COUNT(DISTINCT l.district) AS district_count,
        ROUND(SUM(f.production_tonnes), 1) AS total_production_t,
        ROUND(AVG(f.yield_t_ha), 2) AS avg_yield_t_ha,
        ROUND(AVG(f.rainfall_mm), 1) AS avg_rainfall_mm
    FROM fact_agriculture f
    JOIN dim_location l ON f.location_id = l.location_id
    GROUP BY l.zone, l.state
    ORDER BY l.zone, total_production_t DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)

def olap_drilldown(selected_state="Punjab"):
    """Drill-down: Go from State level down to District and Crop level details."""
    query = """
    SELECT 
        l.state,
        l.district,
        c.crop_name,
        t.year,
        t.season,
        f.production_tonnes,
        f.yield_t_ha,
        f.market_price_per_quintal
    FROM fact_agriculture f
    JOIN dim_location l ON f.location_id = l.location_id
    JOIN dim_crop c ON f.crop_id = c.crop_id
    JOIN dim_time t ON f.time_id = t.time_id
    WHERE l.state = ?
    ORDER BY l.district, t.year DESC, f.yield_t_ha DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=[selected_state])

def olap_slice(crop_name="Rice"):
    """Slice: Filter specifically along 1 dimension (a single crop across all time and regions)."""
    query = """
    SELECT 
        t.year,
        t.season,
        l.state,
        c.crop_name,
        f.area_hectares,
        f.yield_t_ha,
        f.rainfall_mm,
        f.avg_temp_c
    FROM fact_agriculture f
    JOIN dim_crop c ON f.crop_id = c.crop_id
    JOIN dim_time t ON f.time_id = t.time_id
    JOIN dim_location l ON f.location_id = l.location_id
    WHERE c.crop_name = ?
    ORDER BY t.year DESC, f.yield_t_ha DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=[crop_name])

def olap_dice(crop_name="Wheat", season="Rabi", zone="North"):
    """Dice: Sub-cube filter across multiple dimensions (Crop, Season, and Geographical Zone)."""
    query = """
    SELECT 
        t.year,
        l.state,
        l.district,
        c.crop_name,
        f.area_hectares,
        f.production_tonnes,
        f.yield_t_ha,
        f.rainfall_mm,
        f.market_price_per_quintal
    FROM fact_agriculture f
    JOIN dim_crop c ON f.crop_id = c.crop_id
    JOIN dim_time t ON f.time_id = t.time_id
    JOIN dim_location l ON f.location_id = l.location_id
    WHERE c.crop_name = ? AND t.season = ? AND l.zone = ?
    ORDER BY t.year DESC, f.yield_t_ha DESC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=[crop_name, season, zone])

def olap_pivot():
    """Pivot: Cross-tabulation of Average Crop Yield across Years."""
    query = """
    SELECT 
        c.crop_name,
        t.year,
        ROUND(AVG(f.yield_t_ha), 2) AS avg_yield
    FROM fact_agriculture f
    JOIN dim_crop c ON f.crop_id = c.crop_id
    JOIN dim_time t ON f.time_id = t.time_id
    GROUP BY c.crop_name, t.year
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn)
    return df.pivot(index="crop_name", columns="year", values="avg_yield")

if __name__ == "__main__":
    print("\n========== 1. OLAP ROLL-UP (District -> State -> Zone) ==========")
    print(olap_rollup().head(10))

    print("\n========== 2. OLAP DRILL-DOWN (Punjab District & Crop Level) ==========")
    print(olap_drilldown("Punjab").head(8))

    print("\n========== 3. OLAP SLICE (Crop = 'Rice') ==========")
    print(olap_slice("Rice").head(8))

    print("\n========== 4. OLAP DICE (Wheat + Rabi + North Zone) ==========")
    print(olap_dice("Wheat", "Rabi", "North").head(8))

    print("\n========== 5. OLAP PIVOT (Crop vs Year Yield Matrix) ==========")
    print(olap_pivot())
