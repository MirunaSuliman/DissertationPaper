import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_PATH = './output/f1_data_clean.csv'
OUTPUT_PATH = './output/f1_data_final_clean.csv'
ANOMALY_THRESHOLD = 1.0  # in seconds

def main():
    print("\n=== ANOMALY EXCLUSION ===")

    # Load cleaned data
    clean_df = pd.read_csv(INPUT_PATH)

    # Calculate sector sum and discrepancy
    clean_df['ΣSectors'] = clean_df[['Sector1Time', 'Sector2Time', 'Sector3Time']].sum(axis=1)
    clean_df['Discrepancy'] = (clean_df['LapTime'] - clean_df['ΣSectors']).abs()

    # Identify anomalies
    anomalies = clean_df[clean_df['Discrepancy'] >= ANOMALY_THRESHOLD]
    print(f"Identified {len(anomalies)} anomalous laps:")
    print(anomalies[['LapNumber', 'DriverNumber', 'Circuit', 
                     'LapTime', 'ΣSectors', 'Discrepancy']].to_string(index=False))

    # Exclude anomalies
    final_df = clean_df[clean_df['Discrepancy'] < ANOMALY_THRESHOLD].drop(columns=['ΣSectors', 'Discrepancy'])

    # Save final cleaned dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nRemoved {len(anomalies)} anomalies. Final clean data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
