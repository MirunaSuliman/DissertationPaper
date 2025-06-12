import pandas as pd
import os


# --- DIAGNOSTICS ---
def analyze_missing_data(df):
    """Print missing data summary and data types."""
    print("\n=== Missing Data Analysis ===")
    print("Initial data shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    return df


# --- CLEANING PIPELINE ---
def clean_f1_data(df):
    """
    Clean telemetry data through:
    1. Exact arithmetic reconstruction (LapTime = ΣSectors)
    2. Driver-circuit median imputation (fallback)
    """
    time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
    
    for col in time_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds()

    initial_missing = df[time_cols].isnull().sum().to_dict()

    # Retain laps with either full LapTime or all sectors
    valid_mask = (
        df['LapTime'].notna() |
        (df['Sector1Time'].notna() & df['Sector2Time'].notna() & df['Sector3Time'].notna())
    )
    df = df[valid_mask].copy()

    # Reconstruct missing sectors from LapTime
    for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        others = [s for s in time_cols[1:] if s != sector]
        mask = df['LapTime'].notna() & df[sector].isna() & df[others].notna().all(axis=1)
        df.loc[mask, sector] = df.loc[mask, 'LapTime'] - df.loc[mask, others].sum(axis=1)

    # Reconstruct LapTime from all sectors
    mask = df['LapTime'].isna() & df[time_cols[1:]].notna().all(axis=1)
    df.loc[mask, 'LapTime'] = df.loc[mask, time_cols[1:]].sum(axis=1)

    # Median imputation for remaining gaps
    for sector in time_cols[1:]:
        if df[sector].isna().any():
            medians = df.groupby(['DriverNumber', 'Circuit'])[sector].median()
            df = df.merge(medians.rename(f'{sector}_median'), on=['DriverNumber', 'Circuit'], how='left')
            df[sector] = df[sector].fillna(df[f'{sector}_median'])
            df.drop(columns=f'{sector}_median', inplace=True)

    df.dropna(subset=time_cols, inplace=True)

    # Final diagnostics
    final_missing = df[time_cols].isnull().sum().to_dict()
    print("\n=== Cleaning Report ===")
    for col in time_cols[1:]:
        recovered = initial_missing[col] - final_missing[col]
        print(f"{col}: Recovered {recovered} values")

    print(f"\nFinal dataset shape: {df.shape}")
    return df.reset_index(drop=True)


# --- UTILITY ---
def save_clean_data(df, filename='f1_data_clean.csv'):
    """Save cleaned dataset to output folder."""
    os.makedirs('./output', exist_ok=True)
    output_path = f'./output/{filename}'
    df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    return output_path


# --- EXECUTION ---
def main():
    raw_df = pd.read_csv('./data/f1_data.csv')
    analyze_missing_data(raw_df)
    clean_df = clean_f1_data(raw_df)
    save_clean_data(clean_df)

    # Lap time reconstruction validation
    print("\nVerifying reconstruction accuracy...")
    time_diff = clean_df['LapTime'] - clean_df[['Sector1Time', 'Sector2Time', 'Sector3Time']].sum(axis=1)
    mismatches = (time_diff.abs() >= 0.001).sum()

    if mismatches == 0:
        print(f"All {len(clean_df)} lap times valid (±0.001s tolerance)")
    else:
        print(f"Found {mismatches} invalid lap reconstructions")

if __name__ == "__main__":
    main()
