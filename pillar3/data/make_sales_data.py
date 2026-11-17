"""
Generates data/sales.csv for the data analysis agent (09).

Fake daily sales of an industrial vision equipment seller, Jan-Jun 2024.
A few things are planted in it so you can check whether the agent finds them.
They're listed at the bottom of README.md (spoiler), try the agent before reading.

    python data/make_sales_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

PRODUCTS = {  # name: (base daily units per region/channel, unit price in EUR)
    "camera_module": (6, 480.0),
    "lens_pack": (9, 120.0),
    "lighting_bar": (7, 210.0),
    "mount_kit": (14, 35.0),
    "sensor_kit": (4, 890.0),
}
REGIONS = {"North": 1.2, "South": 1.0, "East": 0.8, "West": 0.9}
CHANNELS = {"online": 0.6, "retail": 0.4}

dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
rows = []
for d in dates:
    month_idx = d.month - 1
    for region, r_mult in REGIONS.items():
        growth = 1.04 ** month_idx if region == "West" else 1.0
        for channel, c_mult in CHANNELS.items():
            weekend = 0.5 if (channel == "retail" and d.dayofweek >= 5) else 1.0
            for product, (base, price) in PRODUCTS.items():
                lam = base * r_mult * c_mult * growth * weekend
                if product == "camera_module" and region == "North" and \
                        pd.Timestamp("2024-03-04") <= d <= pd.Timestamp("2024-03-24"):
                    lam *= 0.05  # stockout at the North warehouse
                units = int(rng.poisson(lam))
                return_rate = 0.03
                if product == "lighting_bar" and d >= pd.Timestamp("2024-05-06"):
                    return_rate = 0.18  # defective batch
                returns = int(rng.binomial(units, return_rate)) if units else 0
                unit_price = round(price * rng.uniform(0.97, 1.03), 2)
                rows.append([d.date(), region, channel, product, units, unit_price, returns])

df = pd.DataFrame(rows, columns=["date", "region", "channel", "product", "units", "unit_price", "returns"])

# real data is never clean
df.loc[rng.choice(df.index, 40, replace=False), "unit_price"] = np.nan
df.loc[rng.choice(df.index, 1), "units"] = -12
df = pd.concat([df, df.sample(10, random_state=1)]).sort_values(["date", "region"]).reset_index(drop=True)
df["revenue"] = (df["units"] * df["unit_price"]).round(2)

out = Path(__file__).parent / "sales.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df)} rows to {out}")
