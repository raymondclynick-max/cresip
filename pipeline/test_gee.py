import ee
import pandas as pd

ee.Initialize(project="cresip-gee")
print("GEE ok")

df = pd.read_csv("data/coastal_reservoirs.csv").head(3)
print(df[["id","name","lat","lon","cap_m3"]].to_string())

for _, row in df.iterrows():
    point = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
    region = point.buffer(2000)
    jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory") \
            .filter(ee.Filter.calendarRange(2020, 2020, "year")) \
            .filter(ee.Filter.calendarRange(6, 6, "month"))
    monthly = jrc.first()
    water = monthly.select("water").eq(2)
    obs = monthly.select("water").gte(1)
    w = water.reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e6).getInfo()
    t = obs.reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e6).getInfo()
    frac = (w.get("water", 0) or 0) / max(t.get("water", 1), 1)
    print(f"id={row['id']} water={w} total={t} frac={frac:.3f}")
