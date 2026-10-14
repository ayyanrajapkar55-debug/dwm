import os
import numpy as np
import pandas as pd

def generate_datasets():
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)

    # 1. Geography
    locations = [
        {"state": "Punjab", "district": "Ludhiana", "zone": "North"},
        {"state": "Punjab", "district": "Amritsar", "zone": "North"},
        {"state": "Punjab", "district": "Patiala", "zone": "North"},
        {"state": "Haryana", "district": "Karnal", "zone": "North"},
        {"state": "Haryana", "district": "Hisar", "zone": "North"},
        {"state": "Uttar Pradesh", "district": "Varanasi", "zone": "Central"},
        {"state": "Uttar Pradesh", "district": "Meerut", "zone": "Central"},
        {"state": "Uttar Pradesh", "district": "Agra", "zone": "Central"},
        {"state": "Maharashtra", "district": "Pune", "zone": "West"},
        {"state": "Maharashtra", "district": "Nagpur", "zone": "West"},
        {"state": "Maharashtra", "district": "Nashik", "zone": "West"},
        {"state": "Karnataka", "district": "Dharwad", "zone": "South"},
        {"state": "Karnataka", "district": "Bellary", "zone": "South"},
        {"state": "Andhra Pradesh", "district": "Guntur", "zone": "South"},
        {"state": "Andhra Pradesh", "district": "Krishna", "zone": "South"},
        {"state": "Madhya Pradesh", "district": "Indore", "zone": "Central"},
        {"state": "Madhya Pradesh", "district": "Bhopal", "zone": "Central"},
        {"state": "West Bengal", "district": "Burdwan", "zone": "East"},
        {"state": "West Bengal", "district": "Murshidabad", "zone": "East"},
        {"state": "Gujarat", "district": "Rajkot", "zone": "West"}
    ]

    # 2. Crops definition with biological requirements
    crops_info = [
        {"crop": "Rice", "category": "Cereal", "season": "Kharif", "soil_type": "Clayey Loam", "base_yield": 3.8, "n": 80, "p": 40, "k": 40, "ph": 6.2, "rain": 1150, "temp": 28, "price": 2200},
        {"crop": "Wheat", "category": "Cereal", "season": "Rabi", "soil_type": "Alluvial", "base_yield": 4.2, "n": 105, "p": 50, "k": 50, "ph": 6.8, "rain": 400, "temp": 20, "price": 2400},
        {"crop": "Maize", "category": "Cereal", "season": "Kharif", "soil_type": "Alluvial", "base_yield": 3.2, "n": 75, "p": 35, "k": 35, "ph": 6.5, "rain": 700, "temp": 25, "price": 1900},
        {"crop": "Cotton", "category": "Commercial", "season": "Kharif", "soil_type": "Black Soil", "base_yield": 2.1, "n": 95, "p": 45, "k": 45, "ph": 7.2, "rain": 850, "temp": 31, "price": 6800},
        {"crop": "Sugarcane", "category": "Commercial", "season": "Annual", "soil_type": "Alluvial", "base_yield": 70.0, "n": 150, "p": 65, "k": 70, "ph": 6.9, "rain": 1400, "temp": 29, "price": 350},
        {"crop": "Soybean", "category": "Oilseed", "season": "Kharif", "soil_type": "Black Soil", "base_yield": 2.3, "n": 40, "p": 60, "k": 40, "ph": 6.7, "rain": 900, "temp": 27, "price": 4500},
        {"crop": "Mustard", "category": "Oilseed", "season": "Rabi", "soil_type": "Sandy Loam", "base_yield": 1.7, "n": 60, "p": 30, "k": 30, "ph": 7.0, "rain": 350, "temp": 18, "price": 5400},
        {"crop": "Chickpea", "category": "Pulses", "season": "Rabi", "soil_type": "Sandy Loam", "base_yield": 1.5, "n": 25, "p": 55, "k": 30, "ph": 7.1, "rain": 300, "temp": 22, "price": 5100}
    ]

    years = [2020, 2021, 2022, 2023, 2024]
    
    prod_rows = []
    weather_rows = []
    soil_rows = []
    market_rows = []

    soil_id_counter = 1

    for year in years:
        for loc in locations:
            for c in crops_info:
                # Realistic zoning
                zone_fit = (loc["zone"] in ["North", "Central"] and c["crop"] in ["Wheat", "Mustard", "Sugarcane", "Rice"]) or \
                           (loc["zone"] in ["West", "Central"] and c["crop"] in ["Cotton", "Soybean", "Chickpea"]) or \
                           (loc["zone"] in ["East", "South"] and c["crop"] in ["Rice", "Maize", "Sugarcane"]) or \
                           (np.random.rand() > 0.45)
                
                if not zone_fit:
                    continue

                season = c["season"]

                # Weather with realistic variance
                rain_noise = np.random.normal(0, 35)
                temp_noise = np.random.normal(0, 1.0)
                rain = max(100.0, round(float(c["rain"] + rain_noise), 1))
                temp = round(float(c["temp"] + temp_noise), 1)
                humidity = round(float(np.clip(45 + (rain / 28), 30, 95)), 1)

                weather_rows.append({
                    "year": year,
                    "season": season,
                    "state": loc["state"],
                    "district": loc["district"],
                    "rainfall_mm": rain,
                    "avg_temp_c": temp,
                    "humidity_pct": humidity
                })

                # Soil nutrient measurements for this crop trial
                n_val = int(np.clip(np.random.normal(c["n"], 5), 15, 180))
                p_val = int(np.clip(np.random.normal(c["p"], 4), 10, 90))
                k_val = int(np.clip(np.random.normal(c["k"], 4), 10, 90))
                ph_val = round(float(np.clip(np.random.normal(c["ph"], 0.2), 5.5, 8.5)), 2)

                soil_record_id = soil_id_counter
                soil_id_counter += 1

                soil_rows.append({
                    "soil_id": soil_record_id,
                    "state": loc["state"],
                    "district": loc["district"],
                    "zone": loc["zone"],
                    "soil_type": c["soil_type"],
                    "nitrogen_n": n_val,
                    "phosphorus_p": p_val,
                    "potassium_k": k_val,
                    "ph": ph_val
                })

                # Area & Production
                area = round(float(np.random.uniform(500, 12000)), 1)
                
                rain_factor = 1.0 - min(abs(rain - c["rain"]) / (c["rain"] * 1.5), 0.3)
                temp_factor = 1.0 - min(abs(temp - c["temp"]) / 20.0, 0.2)
                soil_factor = 1.0 + (n_val - c["n"]) / 400.0
                
                calc_yield = c["base_yield"] * rain_factor * temp_factor * soil_factor * np.random.uniform(0.96, 1.04)
                calc_yield = round(float(max(0.2, calc_yield)), 2)
                production = round(float(area * calc_yield), 1)

                prod_rows.append({
                    "year": year,
                    "season": season,
                    "state": loc["state"],
                    "district": loc["district"],
                    "crop": c["crop"],
                    "category": c["category"],
                    "soil_id": soil_record_id,
                    "area_hectares": area,
                    "production_tonnes": production,
                    "yield_t_ha": calc_yield
                })

                # Market Price
                price_inflation = 1.0 + (year - 2020) * 0.05
                supply_factor = 1.0 - (calc_yield - c["base_yield"]) / (c["base_yield"] * 2.5)
                price = round(float(c["price"] * price_inflation * supply_factor * np.random.uniform(0.96, 1.04)), 2)
                
                market_rows.append({
                    "year": year,
                    "season": season,
                    "state": loc["state"],
                    "district": loc["district"],
                    "crop": c["crop"],
                    "market_price_per_quintal": price
                })

    df_prod = pd.DataFrame(prod_rows)
    df_weather = pd.DataFrame(weather_rows).drop_duplicates(subset=["year", "season", "state", "district"])
    df_soil = pd.DataFrame(soil_rows)
    df_market = pd.DataFrame(market_rows).drop_duplicates(subset=["year", "season", "state", "district", "crop"])

    df_prod.to_csv('data/raw_crop_production.csv', index=False)
    df_weather.to_csv('data/raw_weather.csv', index=False)
    df_soil.to_csv('data/raw_soil.csv', index=False)
    df_market.to_csv('data/raw_market.csv', index=False)

    print(f"Generated {len(df_prod)} production records, {len(df_weather)} weather records, {len(df_soil)} soil samples, {len(df_market)} market records.")

if __name__ == "__main__":
    generate_datasets()
