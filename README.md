# Bengaluru Real Estate Valuation & Investment Engine

## 🔗 Professional Profiles & Artifacts
* **LinkedIn:** [Your Name / LinkedIn Profile URL](https://linkedin.com/in/varshitha-gurram-1797252aa)
* **GitHub Repository:** [Project Repository URL](https://github.com/varshithagurram/Real-estate-prediction-model)
* **Tableau Public Dashboard:** [Interactive Market Dashboard URL](https://public.tableau.com/app/profile/varshitha.g2368/viz/BengaluruRealEstate/Dashboard1?publish=yes)

An end-to-end data product that cleans messy residential housing data in Bengaluru, trains a predictive machine learning model to estimate property prices, and visualizes market trends through an interactive dashboard.

## Project Architecture & Workspace
* **data/**: Contains raw and cleaned property listing datasets.
* **models/**: Houses serialized model weights (`.pkl`) and performance logs (`model_metrics.json`).
* **reports/**: Contains text-based summaries covering business strategies, exploratory data analysis (EDA), and project wrap-ups.
* **Notebooks**: All-in-one Jupyter Notebooks handling data cleaning, model training, evaluation, and time-series plotting.

## Tech Stack & Libraries
* **Language:** Python
* **Data & ML:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Tableau Public, Matplotlib, Seaborn

## Core Features & Achievements
1. **Data Gating Pipeline:** Cleans erratic text ranges (e.g., "2100-2850 sqft") and filters out extreme data-entry human errors to ensure downstream data quality.
2. **Predictive Machine Learning:** Trains a Random Forest Regressor hitting an **R² score of ~0.84**, maintaining an average prediction error (MAE) of just ₹4.25 Lakhs.
3. **Tableau Business Intelligence:** A fully connected cross-filtering dashboard tracking micro-market spatial valuation metrics and structural layout price volatility.
4. **5-Year Investment Horizon:** Plots time-series growth trajectories across high-density IT corridors like Indiranagar, Whitefield, and Electronic City.

## Quick Start
To install the required dependencies and explore the notebooks locally, run:
```bash
pip install -r requirements.txt

