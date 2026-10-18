# Customer Spending Prediction

## Project overview
This project uses supervised machine learning to estimate a customer's `spending` value from available demographic and purchase-behaviour columns.

## Dataset
- Source file supplied for this project: `customer_data(1).csv`
- Rows in source file: 1000
- Columns: name, age, gender, education, income, country, purchase_frequency, spending
- Prediction target: `spending`
- `name` is intentionally excluded from training because a person's name is an identifier and should not be used as a predictive feature.

## Folder structure
```text
Customer_Spending_Prediction_Project/
├── app.py
├── train_model.py
├── eda.py
├── requirements.txt
├── README.md
├── data/
│   ├── customer_data.csv
│   └── customer_data_cleaned.csv
├── models/
│   └── best_model.joblib
└── reports/
    ├── project_summary.json
    ├── model_comparison.csv
    └── figures/
```

## Run on macOS (MacBook Air M1)
1. Install Python 3.10 or newer if needed.
2. Open Terminal and move into this folder.
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Retrain and compare models:
   ```bash
   python train_model.py
   ```
6. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
7. The terminal will show a local URL (usually `http://localhost:8501`). Open it in your browser.

## Generate EDA charts
```bash
python eda.py
```

## Model evaluation
The dataset is split into 80% training and 20% testing with a fixed random seed. Models are compared using MAE, RMSE and R². The lowest-RMSE model is saved as `models/best_model.joblib`.

## Important limitations
- This is a learning/demo project, not a production decision system.
- Results depend on the quality and representativeness of the supplied dataset.
- `purchase_frequency` appears numeric and is used as supplied; check the original data documentation for its unit/meaning.
- Correlation does not establish causation.
- Do not use names as model features or publish identifiable personal information.
