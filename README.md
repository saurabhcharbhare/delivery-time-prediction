# Delivery Time Prediction Using Regression

**Student:** Saurabh Charbhare  
**Program:** B.Tech Computer Science & Engineering  
**University:** DBATU Lonere  
**Academic year:** 2026–2027

A Streamlit application that estimates delivery duration from distance, traffic, weather, nearby delivery-partner availability, order preparation time and order type. It trains and compares Linear Regression and Random Forest Regression and reports MAE, RMSE and R² on a held-out test split.

> **Important data note:** `data/delivery_data.csv` is simulated educational data generated with a documented formula. It is suitable for demonstrating the pipeline, not for claiming real-world delivery accuracy. For final college submission, find and cite a reliable real dataset (or clearly justify a self-collected dataset), retrain, record the results, and update this README/report.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`
6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

To regenerate the simulated dataset: `python generate_dataset.py`.

## ML pipeline

- **Task:** supervised regression.
- **Target:** `delivery_time_min`.
- **Features:** `distance_km`, `traffic`, `weather`, `partner_availability`, `prep_time_min`, `order_type`.
- **Split:** 80% train / 20% test, `random_state=42`.
- **Numeric preprocessing:** median imputation and standard scaling.
- **Categorical preprocessing:** most-frequent imputation and one-hot encoding.
- **Algorithms:** Linear Regression and Random Forest Regressor.
- **Metrics:** MAE, RMSE and R².

## Deploy to a public URL

1. Create a GitHub repository and upload this folder's contents.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Choose **Create app**, select the repository and branch, set the main file path to `app.py`, then deploy.
4. Test the deployed app on mobile and desktop. Use the resulting public URL in your resume only after confirming it works.

## Before submission

- Replace the simulated dataset with a documented dataset where possible; include source, license/terms, feature definitions and limitations.
- Run at least one model-improvement experiment and record before/after metrics.
- Add screenshots and verified metrics to the report.
- Understand the code and be able to explain preprocessing, split, algorithms and evaluation in viva.
- Prediction is an estimate, not a guaranteed delivery time.
