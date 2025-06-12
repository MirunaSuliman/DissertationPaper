import fastf1 as ff1
import pandas as pd
from tqdm import tqdm
import os

# Enable FastF1 cache
ff1.Cache.enable_cache('./f1_cache')

# --- CONFIGURATION ---
seasons = [2021, 2022, 2023]
circuits = {
    'Street': ['Monaco', 'Singapore'],
    'HighSpeed': ['Monza', 'Baku'],
    'Technical': ['Hungaroring', 'Barcelona']
}
sessions = ['Q', 'R']  # 'Q' = Qualifying, 'R' = Race

# --- DATA EXTRACTION FUNCTION ---
def extract_f1_data():
    all_laps = []

    for year in tqdm(seasons, desc="Processing Seasons"):
        for circuit_type, track_names in circuits.items():
            for track in track_names:
                for session_type in sessions:
                    try:
                        session = ff1.get_session(year, track, session_type)
                        session.load(telemetry=True)

                        laps = session.laps.copy()
                        laps['Year'] = year
                        laps['Circuit'] = track
                        laps['CircuitType'] = circuit_type
                        laps['SessionType'] = 'Qualifying' if session_type == 'Q' else 'Race'

                        for idx, lap in laps.iterrows():
                            try:
                                telemetry = lap.get_telemetry()
                                if telemetry is not None:
                                    laps.at[idx, 'SpeedAvg'] = telemetry['Speed'].mean()
                                    laps.at[idx, 'ThrottleAvg'] = telemetry['Throttle'].mean()
                                    laps.at[idx, 'BrakeAvg'] = telemetry['Brake'].mean()
                            except:
                                continue  # Skip invalid telemetry laps

                        # Keep only essential columns and drop invalid laps
                        cols_to_keep = [
                            'LapNumber', 'DriverNumber', 'Team', 'LapTime',
                            'Sector1Time', 'Sector2Time', 'Sector3Time',
                            'SpeedAvg', 'ThrottleAvg', 'BrakeAvg',
                            'Year', 'Circuit', 'CircuitType', 'SessionType'
                        ]
                        laps = laps[cols_to_keep].dropna(subset=['SpeedAvg'])
                        all_laps.append(laps)

                        print(f"{year} {track} {session_type} ({len(laps)} laps)")
                    except Exception as e:
                        print(f"Failed {year} {track} {session_type}: {str(e)}")
                        continue

    if not all_laps:
        raise ValueError("No data extracted! Check log for errors.")

    return pd.concat(all_laps).reset_index(drop=True)

# --- MAIN EXECUTION ---

def main():
    print("\n=== DATA EXTRACTION ===")
    df = extract_f1_data()
    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/f1_data.csv', index=False)
    print(f"Done! Saved {len(df)} laps to './data/f1_data.csv'")

if __name__ == "__main__":
    main()

