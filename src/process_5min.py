# Process each daily pems 5 minute file into smaller parquet file: 
# keeping mainline sensors on I-10, I-15, I-215, SR-60 and SR-91 timestamps 
# also added the per-station and per-day quality table 



import glob 
import os
import pandas as pd
import time 

RAW_DIR = "data/raw/station_5min"
OUT_DIR = "data/processed/5min"

QUALITY_FILE = "data/processed/station_quality.parquet"
FREEWAYS = [10, 15, 215, 60, 91]

# columns in the raw data file, we will use these to read the data and then filter out the columns we need for the analysis
NAMES = [
    "timestamp", "station", "district", "freeway", "direction", "lane_type",
    "station_length", "samples", "pct_observed", "total_flow", "avg_occupancy", "avg_speed",
]

# dataframe we are using for the processing, only keeping the columns we need for the analysis
KEEP = ["timestamp", "station", "freeway", "direction",
        "pct_observed", "total_flow", "avg_occupancy", "avg_speed"]


# category used for storing direction instead of using letter W etc. 
DTYPES = {
    "station": "int32",
    "freeway": "int16",
    "direction": "category",
    "pct_observed": "int8",
    "total_flow": "float32",
    "avg_occupancy": "float32",
    "avg_speed": "float32",
}


# processing the file 

def process_file(path):
    df = pd.read_csv(path, header = None, names = NAMES, usecols = range(len(NAMES)))
    df = df[(df["lane_type"] == "ML") & (df["freeway"].isin(FREEWAYS))].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S")
    return df[KEEP].astype(DTYPES)

# looping over all the files 

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "d08_text_station_5min_*")))
    print(f"{len(files)} raw files found")
    
    new_quality = []
    t_start = time.time()
    
    for i, path in enumerate(files, 1):
        day = os.path.basename(path).split("5min_")[1][:10]
        out_path = os.path.join(OUT_DIR, f"{day}.parquet")
        
        if os.path.exists(out_path):
            continue
        df = process_file(path)
        df.to_parquet(out_path, index=False)
        
        q = df.groupby("station", observed=True)["pct_observed"].mean().rename("pct_observed").reset_index()
        q["date"] = pd.Timestamp(day.replace("_", "-"))
        new_quality.append(q)

        n_dead = int((q["pct_observed"] == 0).sum())
        print(f"[{i}/{len(files)}] {day}: {len(df):,} rows, {df['station'].nunique()} stations, "
        f"{n_dead} fully imputed")

    if new_quality:
        quality = pd.concat(new_quality, ignore_index=True)
        if os.path.exists(QUALITY_FILE):
            quality = pd.concat([pd.read_parquet(QUALITY_FILE), quality], ignore_index=True)
            quality = quality.drop_duplicates(["station", "date"], keep="last")
        quality.to_parquet(QUALITY_FILE, index=False)

    print(f"Done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()