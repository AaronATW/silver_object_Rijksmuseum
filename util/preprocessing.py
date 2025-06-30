import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "Rijksmuseum Ag objects - 20250623.csv"
OUTPUT_FILE = BASE_DIR / "data" / "final_xrf_metal_data.csv"

xrf_cols = [f"ch_{i:04d}" for i in range(1, 2049)]
metal_cols = ['Fe', 'Ni', 'Cu', 'Zn', 'Ag', 'Cd', 'Sn', 'Sb', 'Au', 'Hg', 'Pb', 'Bi']
meta_cols = ['ObjectID', 'Period', 'City', 'Country', 'Year', 'Artist']
df = pd.read_csv(DATA_FILE)
df = df[meta_cols + metal_cols + xrf_cols]

# XRF normalization
total_count = df[xrf_cols].sum(axis=1)
df[xrf_cols] = df[xrf_cols].div(total_count, axis=0)

# Add PeriodGroup column based on Year
def map_year_to_group(year):
    if pd.isna(year):
        return None
    if 1350 <= year <= 1699:
        return "Group1:1350-1699"
    elif 1700 <= year <= 1749:
        return "Group2:1700-1749"
    elif 1750 <= year <= 1799:
        return "Group3:1750-1799"
    elif 1800 <= year <= 1849:
        return "Group4:1800-1849"
    elif year >= 1850:
        return "Group5:1850-----"
df["PeriodGroup"] = df["Year"].apply(map_year_to_group)

# Add CityGroup column based on City
city_region_map = {
    # West-North (Noord-Holland)
    "Amsterdam": "West-North", "Amsterda": "West-North", "Amsterdan": "West-North",
    "Haarlem": "West-North", "Hoorn": "West-North", "Enkhuizen": "West-North",

    # West-South (Zuid-Holland & Utrecht)
    "Den Haag": "West-South", "De Haag": "West-South", "Rotterdam": "West-South",
    "Delft": "West-South", "Leiden": "West-South", "Dordrecht": "West-South",
    "Voorschoten": "West-South", "Schoonhoven": "West-South", "Utrecht": "West-South",
    "Zeist": "West-South",

    # North
    "Leeuwarden": "North", "Harlingen": "North", "Franeker": "North",
    "Sneek": "North", "Wommels": "North", "Dokkum": "North",
    "Winsum": "North", "Stavoren": "North", "Groningen": "North", "Bolsward": "North",

    # East
    "Zwolle": "East", "Nijmegen": "East", "Arnhem": "East", "Deventer": "East", "Zutphen": "East",

    # South
    "Maastricht": "South", "Oss": "South", "Boxmeer": "South",
    "s Hertogenbosch": "South", "s-Hertogenbosch": "South", "s- Hertogenbosch": "South",
    "Middelburg": "South", "Zierikzee": "South", "Bruinisse": "South",

    # Foreign or Unknown
    "Antwerpen": "Others", "Antwerp": "Others",
    "Kopenhagen": "Others", "Copenhagen": "Others",
    "Lingen": "Others", "Netherlands": "Others", "Unknown": "Others"
}

def map_city_to_region(city):
    return city_region_map.get(city, "ErrorGroup")
df["CityGroup"] = df["City"].apply(map_city_to_region)

# Add TopArtist column: keep artist names with >=40 samples, else assign 'Others'
artist_counts = df["Artist"].value_counts()
top_artists = artist_counts[(artist_counts >= 40) & (artist_counts.index != "Unknown")].index
df["TopArtist"] = df["Artist"].apply(lambda x: x if x in top_artists else "Others")

# Ensure the output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

def check_nan(df_subset, name):
    print(f"\nMissing values in {name}:")
    nan_cols = df_subset.columns[df_subset.isna().any()].tolist()
    if nan_cols:
        print(f"Columns: {nan_cols}")
    else:
        print("No missing values in columns")
    nan_rows = df_subset[df_subset.isna().any(axis=1)]
    print(f"Rows with NaN: {len(nan_rows)}")

check_nan(df[xrf_cols], "XRF channels")
check_nan(df[metal_cols], "Metal composition")
check_nan(df[meta_cols], "Metadata")

# Save metadata summary to file
summary_file = BASE_DIR / "data" / "metadata_summary.txt"
summary_file.parent.mkdir(parents=True, exist_ok=True)

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Metadata summary:\n\n")
    for col in ['Period', 'City', 'Country', 'Artist', 'PeriodGroup', 'CityGroup', 'TopArtist']:
        value_counts = df[col].value_counts(dropna=False)
        num_unique = value_counts.shape[0]
        f.write(f"- {col}: {num_unique} unique value(s)\n")
        for val, count in value_counts.items():
            val_str = str(val) if pd.notna(val) else "NaN"
            f.write(f"  {val_str}: {count}\n")
        f.write("\n")
