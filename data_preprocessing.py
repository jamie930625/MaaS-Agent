import os
import pandas as pd
import urllib.request

def download_data(url: str, filename: str) -> pd.DataFrame:
    """Download parquet file if not exists and load into DataFrame."""
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return pd.read_parquet(filename)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter outliers and invalid records based on business logic."""
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Time constraints
    df = df[(df['tpep_pickup_datetime'].dt.year == 2024) & 
            (df['tpep_pickup_datetime'].dt.month == 1) &
            (df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime'])]
    
    # Value constraints
    df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0)]
    df = df.dropna(subset=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime'])
    
    # Duration constraints
    df['duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 300)]
    
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate demand and generate temporal features."""
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')
    demand_df = df.groupby(['PULocationID', 'pickup_hour']).size().reset_index(name='demand')
    
    demand_df['hour'] = demand_df['pickup_hour'].dt.hour
    demand_df['day_of_week'] = demand_df['pickup_hour'].dt.dayofweek
    demand_df['is_weekend'] = (demand_df['day_of_week'] >= 5).astype(int)
    
    def is_rush_hour(row):
        if row['is_weekend'] == 0 and ((7 <= row['hour'] <= 9) or (17 <= row['hour'] <= 19)):
            return 1
        return 0
        
    demand_df['is_rush_hour'] = demand_df.apply(is_rush_hour, axis=1)
    return demand_df

if __name__ == "__main__":
    SOURCE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    RAW_FILE = "yellow_tripdata_2024_01.parquet"
    OUTPUT_FILE = "processed_demand_data.csv"
    
    raw_df = download_data(SOURCE_URL, RAW_FILE)
    cleaned_df = clean_data(raw_df)
    final_df = extract_features(cleaned_df)
    
    final_df.to_csv(OUTPUT_FILE, index=False)