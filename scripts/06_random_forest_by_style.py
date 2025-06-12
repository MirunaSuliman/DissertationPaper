import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def run_random_forest_by_style(data_path='output/modeling/results/f1_race_level_data.csv', target='PointsFinish'):
    df = pd.read_csv(data_path)
    features = ['LapTime_AvgPace', 'LapTime_PaceConsistency', 'LapTime_PeakPace', 'SectorBalance']

    # Ensure output directories exist
    figures_dir = 'output/modeling/figures'
    results_dir = 'output/modeling/results'
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    summary = []
    all_importances = []

    for style in df['DrivingStyle'].unique():
        print(f"\nTraining model for driving style: {style}")
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
        model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(5), scoring='accuracy')
        cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

        print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        print(f"Cross-validated Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        # Feature importances
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'DrivingStyle': style
        })
        all_importances.append(importance_df)

        # Plot feature importances
        importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance_df_sorted, palette='viridis')
        plt.title(f'Feature Importances ({style})')
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/feature_importances_plot_{style}.png')
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix ({style})')
        plt.savefig(f'{figures_dir}/confusion_matrix_{style}.png')
        plt.close()

        # Summary row
        summary.append({
            'DrivingStyle': style,
            'Accuracy': acc,
            'F1': f1,
            'CV_Mean_Acc': cv_mean,
            'CV_Std_Acc': cv_std,
            'Top_Feature': importance_df_sorted.iloc[0]['Feature']
        })

    # Combine all feature importances
    combined_importances = pd.concat(all_importances)

    # Grouped bar chart
    plt.figure(figsize=(12, 7))
    sns.barplot(data=combined_importances, x='Feature', y='Importance', hue='DrivingStyle', palette='Set2')
    plt.title('Feature Importance Comparison Across Driving Styles')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.legend(title='Driving Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/feature_importance_comparison_across_styles.png')
    plt.close()

    # Heatmap
    heatmap_data = combined_importances.pivot(index='Feature', columns='DrivingStyle', values='Importance')
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Feature Importance Heatmap by Driving Style')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/feature_importance_heatmap.png')
    plt.close()

    # Save summary metrics
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{results_dir}/summary_metrics.csv', index=False)
    print("\nSummary metrics saved.")
    print(summary_df)

    return summary_df

if __name__ == "__main__":
    run_random_forest_by_style(target='PointsFinish')
