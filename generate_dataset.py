"""Generate a reproducible simulated dataset for demonstrating the project pipeline.
Replace with a documented real dataset before making real-world accuracy claims.
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1800
distance = np.round(rng.uniform(0.5, 25, n), 2)
traffic = rng.choice(["Low", "Moderate", "High", "Very High"], n, p=[.22,.40,.28,.10])
weather = rng.choice(["Clear", "Cloudy", "Rain", "Heavy Rain"], n, p=[.48,.25,.20,.07])
availability = rng.integers(0, 16, n)
prep = np.clip(rng.normal(14, 6, n), 3, 45).round().astype(int)
order_type = rng.choice(["Food", "Grocery", "Parcel", "Other"], n, p=[.55,.20,.18,.07])
traffic_effect = {"Low":0,"Moderate":5,"High":12,"Very High":20}
weather_effect = {"Clear":0,"Cloudy":2,"Rain":7,"Heavy Rain":14}
type_effect = {"Food":0,"Grocery":4,"Parcel":3,"Other":5}
noise = rng.normal(0, 4, n)
time = 7 + distance*3.1 + np.array([traffic_effect[x] for x in traffic]) + np.array([weather_effect[x] for x in weather]) + np.maximum(0, 7-availability)*1.0 + prep + np.array([type_effect[x] for x in order_type]) + noise
time = np.clip(time, 8, None).round(1)
df = pd.DataFrame({"distance_km":distance,"traffic":traffic,"weather":weather,"partner_availability":availability,"prep_time_min":prep,"order_type":order_type,"delivery_time_min":time})
out = Path(__file__).parent / "data" / "delivery_data.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} simulated rows to {out}")
