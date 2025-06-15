import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def run_svm_by_style(data_path='output/f1_race_level_data.csv', target='PointsFinish'):
    df = pd.read_csv(data_path)
    features = ['LapTime_AvgPace', 'LapTime_PaceConsistency', 'LapTime_PeakPace', 'SectorBalance']

    # Ensure output directories exist
    figures_dir = 'output/modeling/figures_svm'
    results_dir = 'output/modeling/results_svm'
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    summary = []

    for style in df['DrivingStyle'].unique():
        print(f"\nTraining SVM model for driving style: {style}")
        df_style = df[df['DrivingStyle'] == style]

        if len(df_style) < 20:
            print(f"Skipping {style} (too few samples: {len(df_style)})")
            continue

        X = df_style[features]
        y = df_style[target]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(5), scoring='accuracy')
        cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        print(f"Cross-validated Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Purples')
        plt.title(f'Confusion Matrix (SVM - {style})')
        plt.savefig(f'{figures_dir}/confusion_matrix_svm_{style}.png')
        plt.close()

        # Summary row
        summary.append({
            'DrivingStyle': style,
            'Accuracy': acc,
            'F1': f1,
            'CV_Mean_Acc': cv_mean,
            'CV_Std_Acc': cv_std
        })

    # Save summary metrics
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{results_dir}/svm_summary_metrics.csv', index=False)
    print("\nSVM summary metrics saved.")
    print(summary_df)

    return summary_df

if __name__ == "__main__":
    run_svm_by_style(target='PointsFinish')
