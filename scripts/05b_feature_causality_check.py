import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('output/modeling/f1_race_level_data.csv')

# Define features and target
features = ['LapTime_AvgPace', 'LapTime_PaceConsistency', 'LapTime_PeakPace', 'SectorBalance']
target = 'PointsFinish'

# Function to compute R² in both directions
def compute_bidirectional_r2(X, y):
    model1 = LinearRegression().fit(X, y)
    r2_forward = model1.score(X, y)

    model2 = LinearRegression().fit(y, X)
    r2_reverse = model2.score(y, X)

    return r2_forward, r2_reverse

# Run for all features
print("Bidirectional R² values (feature <-> PointsFinish):\n")
for feature in features:
    X = df[[feature]].values
    y = df[[target]].values
    r2_fwd, r2_rev = compute_bidirectional_r2(X, y)
    print(f"{feature}:")
    print(f"  R² ({target} ~ {feature}): {r2_fwd:.4f}")
    print(f"  R² ({feature} ~ {target}): {r2_rev:.4f}\n")
