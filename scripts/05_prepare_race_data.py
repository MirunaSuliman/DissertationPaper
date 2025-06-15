import pandas as pd
import fastf1 as ff1
import os
from tqdm import tqdm
import numpy as np

# Initialize FastF1 cache
ff1.Cache.enable_cache('./f1_cache')

def calculate_sector_balance(row):
    sectors = [row['Sector1Time'], row['Sector2Time'], row['Sector3Time']]
    first_half_std = np.std([sectors[0], sectors[1]])
    second_half_std = np.std([sectors[1], sectors[2]])
    return first_half_std - second_half_std

def create_race_level_dataset(cleaned_data_path='./output/f1_data_final_clean.csv',
                              clusters_path='./output/clustering/results/f1_data_with_clusters.csv',
                              output_path='./output/f1_race_level_data.csv'):
    target_variable = 'PointsFinish'

    # Load and preprocess data
    try:
        df = pd.read_csv(cleaned_data_path)
        time_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time', 'LapTime']
        for col in time_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
    except Exception as e:
        print(f"Error loading cleaned data: {str(e)}")
        return None

    try:
        clusters_df = pd.read_csv(clusters_path)
        df = df.merge(clusters_df[['Year', 'Circuit', 'DriverNumber', 'Cluster']], 
                      on=['Year', 'Circuit', 'DriverNumber'], how='left')
        df = df[df['Cluster'].notna()].copy()
        style_map = {0: 'Balanced', 1: 'Aggressive', 2: 'Cautious'}
        df['DrivingStyle'] = df['Cluster'].map(style_map)
    except Exception as e:
        print(f"Error loading or merging clusters: {str(e)}")
        return None

    # Feature engineering
    race_df = df[df['SessionType'] == 'Race'].copy()
    tqdm.pandas(desc="Calculating SectorBalance")
    race_df['SectorBalance'] = race_df.progress_apply(calculate_sector_balance, axis=1)

    features = race_df.groupby(
        ['Year', 'Circuit', 'DriverNumber', 'DrivingStyle', 'CircuitType']
    ).agg({
        'LapTime': [
            ('AvgPace', 'mean'),
            ('PaceConsistency', 'std'),
            ('PeakPace', lambda x: x.nsmallest(3).mean())
        ],
        'SectorBalance': 'mean'
    })

    features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in features.columns]
    features.reset_index(inplace=True)
    features.rename(columns={'SectorBalance_mean': 'SectorBalance'}, inplace=True)

    # Fetch race results for target variable
    print("Fetching race results...")
    results_cache = {}

    def get_race_result(year, circuit):
        cache_key = (int(year), circuit)
        if cache_key not in results_cache:
            try:
                session = ff1.get_session(year, circuit, 'R')
                session.load()
                results_cache[cache_key] = session.results
            except Exception as e:
                print(f"Error fetching {year} {circuit}: {e}")
                return None
        return results_cache[cache_key]

    tqdm.pandas(desc="Adding target variable")
    def add_outcome(row):
        results = get_race_result(row['Year'], row['Circuit'])
        if results is None:
            return np.nan
        try:
            driver_result = results[results['DriverNumber'] == str(int(row['DriverNumber']))].iloc[0]
            return 1 if driver_result['Points'] > 0 else 0
        except (IndexError, KeyError):
            return np.nan

    features[target_variable] = features.progress_apply(add_outcome, axis=1)

    # Final export
    features = features.dropna(subset=[target_variable])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print("Features:", list(features.columns))
    print("Target distribution:", features[target_variable].value_counts().to_dict())

    return features

if __name__ == "__main__":
    dataset = create_race_level_dataset()
