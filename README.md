# Estimating Formula 1 Car Performance Through Predictive Analytics and Data Mining

This repository accompanies the dissertation project:

**“Estimating Formula 1 Driver Performance Using Telemetry-Based Behavioral Clustering and Predictive Modeling”**  
*Department of Software Development and Business Information Systems*  
*Faculty of Economics and Business Administration*  
*Alexandru Ioan Cuza University, Iași, Romania*

---

## Project Summary

This study explores whether Formula 1 drivers exhibit identifiable driving styles based on qualifying telemetry, and how these styles influence race outcomes. Drawing on telemetry from six Grand Prix circuits across the 2021–2023 seasons, the analysis proceeds in two stages:

### 1. Behavioral Clustering (Hypothesis H₁)
Standardized lap telemetry—average throttle, brake, and speed—is reduced via Principal Component Analysis (PCA), followed by K-Means clustering to group laps into different driving styles. Hierarchical clustering is used for structural validation. Three distinct styles consistently emerge:
- **Aggressive**
- **Balanced**
- **Cautious**

### 2. Predictive Modeling (Hypothesis H₂)
Each behavioral group is modeled independently using Random Forest classifiers to predict whether a driver scores points in a race. Aggregated performance features (e.g., average pace, pace consistency, peak pace, and sector balance) are computed at the race level. Model performance and feature importances vary across styles, highlighting the role of driving behavior in shaping competitive outcomes.

---

## Repository Structure

### **Data**  
- `data/`  
  - `f1_data.csv` *(Raw dataset)*  

### **Output**  
- `output/`  
  - **Clustering**  
    - `figures/` *(Visualizations: PCA plots, dendrograms, etc.)*  
    - `results/` *(Cluster labels, metrics, etc.)*  
  - **Modeling**  
    - `figures/` *(Model plots: feature importance, confusion matrices etc.)*  
    - `results/` *(Race level data, smmary metrics etc.)*  
  - `f1_data_clean.csv` *(Intermediate cleaned data)*  
  - `f1_data_final_clean.csv` *(Final cleaned data after anomaly removal)*  

### **Scripts**  
- `scripts/` *(Pipeline in numbered order)*  
  - `01_data_extraction.py`  
  - `02_data_cleaning.py`  
  - `03_exclude_anomalies.py`  
  - `04_driver_clustering.py`  
  - `05_prepare_race_data.py`  
  - `06_random_forest_by_style.py`  

### **Config**  
- `.gitattributes` *(Git settings)*  

---

## Key Contributions

- Demonstrates that unsupervised learning can extract meaningful behavioral patterns from raw telemetry.
- Shows that the predictive value of race features is conditioned by driving style.
- Supports a behavior-aware modeling framework for race performance analytics in elite motorsport.

---

## Author

**Miruna Suliman**  
Email: [miruna.suliman@yahoo.com](mailto:miruna.suliman@yahoo.com)  
GitHub: [@MirunaSuliman](https://github.com/MirunaSuliman)

---
