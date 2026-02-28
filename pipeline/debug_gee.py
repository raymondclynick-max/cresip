"""
Debug GEE extraction for a single reservoir.
Run: python pipeline/debug_gee.py
"""
import ee
import pandas as pd

ee.Initialize(project="cresip-gee")
print("✓ GEE initialised")

# Load first reservoir
df = pd.read_csv("data/coastal_reservoirs.csv")
row = df.iloc[0]
print(f"\nTesting reservoir: {row.get('name', 'unnamed')} | id={row['id']}")
print(f"  lat={row['lat']}, lon={row['lon']}, cap_m3={row['cap_m3']}")

# Build point and buffer
point = ee.Geometry.Point([float(row['lon']), float(row['lat'])])
region = point.buffer(2000)  # 2km buffer

print("\n--- Test 1: JRC monthly water history ---")
try:
    jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
    img = jrc.filter(ee.Filter.calendarRange(2020, 2020, 'year')) \
              .filter(ee.Filter.calendarRange(6, 6, 'month')) \
              .first()
    
    if img is None:
        print("  No image returned")
    else:
        info = img.getInfo()
        print(f"  Image bands: {[b['id'] for b in info['bands']]}")
        
        # Sample water fraction
        result = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e6
        ).getInfo()
        print(f"  reduceRegion result: {result}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- Test 2: Check what collection version exists ---")
try:
    # Try different JRC versions
    for version in ["JRC/GSW1_4/MonthlyHistory", "JRC/GSW1_3/MonthlyHistory", "JRC/GSW1_2/MonthlyHistory"]:
        try:
            col = ee.ImageCollection(version)
            size = col.size().getInfo()
            print(f"  {version}: {size} images")
            break
        except Exception as e2:
            print(f"  {version}: ERROR - {e2}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- Test 3: Raw pixel values in region ---")
try:
    jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
    img = jrc.filter(ee.Filter.calendarRange(2020, 2020, 'year')) \
              .filter(ee.Filter.calendarRange(6, 6, 'month')) \
              .first()
    
    sample = img.sample(region=region, scale=30, numPixels=10).getInfo()
    print(f"  Sample features: {len(sample['features'])}")
    if sample['features']:
        print(f"  First feature props: {sample['features'][0]['properties']}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- Test 4: Check if region has any JRC data at all ---")
try:
    jrc_all = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory") \
                .filterDate("2020-01-01", "2020-12-31")
    count = jrc_all.size().getInfo()
    print(f"  JRC images in 2020: {count}")
    
    # Check first image over region
    first = jrc_all.first()
    val = first.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=30,
        maxPixels=1e6
    ).getInfo()
    print(f"  Value over reservoir region: {val}")
except Exception as e:
    print(f"  ERROR: {e}")
