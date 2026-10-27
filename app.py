import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Delivery Time Prediction", page_icon="📦", layout="wide")

# Compact product-style styling; no external assets required.
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f4f7fb; color:#172338;}
.block-container {padding-top:1.7rem; max-width:1200px;}
.hero {background:linear-gradient(120deg,#10233f,#17466a);padding:28px 30px;border-radius:18px;color:white;margin-bottom:20px;}
.hero h1 {color:white;font-size:2rem;margin:0 0 8px 0;letter-spacing:-.6px;}
.hero p {color:#d5e5f2;margin:0;font-size:1rem;}
.kpi {background:white;border:1px solid #e3eaf2;border-radius:14px;padding:17px 19px;min-height:100px;}
.kpi-label {color:#66768c;font-size:.82rem;margin-bottom:8px;}
.kpi-value {color:#172338;font-size:1.55rem;font-weight:700;}
.section {font-size:1.15rem;font-weight:700;margin:10px 0 12px 0;color:#172338;}
.note {background:#edf5ff;border-left:4px solid #3984c6;padding:12px 14px;border-radius:8px;color:#314a67;font-size:.9rem;}
footer {visibility:hidden;}
</style>""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "delivery_data.csv"
FEATURES = ["distance_km", "traffic", "weather", "partner_availability", "prep_time_min", "order_type"]
TARGET = "delivery_time_min"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_models(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric = ["distance_km", "partner_availability", "prep_time_min"]
    categorical = ["traffic", "weather", "order_type"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    models = {
        "Linear Regression": Pipeline([("prep", pre), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=180, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1))])
    }
    metrics, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics.append({"Algorithm": name, "MAE (min)": mean_absolute_error(y_test, pred), "RMSE (min)": mean_squared_error(y_test, pred) ** .5, "R²": r2_score(y_test, pred)})
        fitted[name] = model
    return fitted, pd.DataFrame(metrics), X_test, y_test

if not DATA_PATH.exists():
    st.error("Dataset file not found. Please ensure data/delivery_data.csv is included in the project.")
    st.stop()

df = load_data()
models, metrics, X_test, y_test = train_models(df)

st.markdown("""<div class="hero"><h1>Delivery Time Prediction</h1><p>Machine learning for smarter delivery estimates · Python + Regression</p></div>""", unsafe_allow_html=True)

page = st.sidebar.radio("NAVIGATION", ["Prediction", "Model Performance", "Dataset Explorer", "About Project"])
st.sidebar.markdown("---")
st.sidebar.caption("B.Tech CSE · Machine Learning Mini Project")
st.sidebar.caption("Developed by: Saurabh Charbhare")

if page == "Prediction":
    a,b,c = st.columns(3)
    for col, label, value in [(a,"DATASET RECORDS",f"{len(df):,}"),(b,"MODELS COMPARED","02"),(c,"TEST SPLIT","20%")]:
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section">Estimate a delivery</div>', unsafe_allow_html=True)
    st.write("Enter the order and delivery conditions below. The selected trained regression model will return an estimated time.")
    with st.form("prediction_form"):
        left,right = st.columns(2)
        with left:
            distance = st.number_input("Delivery distance (km)", min_value=0.1, max_value=80.0, value=4.5, step=0.5)
            traffic = st.selectbox("Traffic conditions", ["Low", "Moderate", "High", "Very High"], index=1)
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Heavy Rain"], index=0)
        with right:
            availability = st.slider("Available delivery partners nearby", 0, 20, 5)
            prep = st.number_input("Order preparation time (minutes)", min_value=1, max_value=180, value=12, step=1)
            order_type = st.selectbox("Order type", ["Food", "Grocery", "Parcel", "Other"])
        model_choice = st.selectbox("Prediction model", list(models.keys()), index=1)
        submitted = st.form_submit_button("Calculate estimated time", type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([{"distance_km":distance,"traffic":traffic,"weather":weather,"partner_availability":availability,"prep_time_min":prep,"order_type":order_type}])
        prediction = max(1, float(models[model_choice].predict(sample)[0]))
        st.markdown("---")
        x,y = st.columns([1,2])
        with x:
            st.markdown(f'<div class="kpi"><div class="kpi-label">ESTIMATED DELIVERY TIME</div><div class="kpi-value">{prediction:.0f} min</div></div>', unsafe_allow_html=True)
        with y:
            st.markdown(f'<div class="note"><b>Model used:</b> {model_choice}<br>Estimate based on the entered order details and the project dataset. Actual delivery can vary with route, restaurant workload and live conditions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Academic prototype: the included dataset is simulated for demonstration. Replace it with a documented real dataset and retrain before claiming real-world accuracy.</div>', unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown('<div class="section">Regression model comparison</div>', unsafe_allow_html=True)
    st.caption("Metrics are calculated on the held-out test set (20%, random_state=42). Lower MAE/RMSE and higher R² generally indicate better test performance.")
    st.dataframe(metrics.style.format({"MAE (min)":"{:.2f}","RMSE (min)":"{:.2f}","R²":"{:.3f}"}), use_container_width=True, hide_index=True)
    best = metrics.sort_values("MAE (min)").iloc[0]
    st.markdown(f'<div class="note"><b>Current selection by MAE:</b> {best["Algorithm"]} (MAE {best["MAE (min)" ]:.2f} min). Final model selection should consider all metrics and validate on a reliable real dataset.</div>', unsafe_allow_html=True)
    selected = st.selectbox("View actual vs predicted", list(models.keys()))
    pred = models[selected].predict(X_test)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(y_test, pred, alpha=.55)
    lo, hi = min(float(y_test.min()), float(pred.min())), max(float(y_test.max()), float(pred.max()))
    ax.plot([lo,hi],[lo,hi], linestyle="--")
    ax.set_xlabel("Actual delivery time (minutes)"); ax.set_ylabel("Predicted delivery time (minutes)"); ax.set_title(selected)
    ax.grid(alpha=.2); st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif page == "Dataset Explorer":
    st.markdown('<div class="section">Dataset overview</div>', unsafe_allow_html=True)
    st.write("The included CSV is a simulated teaching dataset, not measured delivery-company data.")
    x,y,z = st.columns(3)
    x.metric("Rows", f"{len(df):,}"); y.metric("Input features", len(FEATURES)); z.metric("Target", "Minutes")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.markdown("**Target distribution**")
    fig, ax = plt.subplots(figsize=(7,3)); ax.hist(df[TARGET], bins=25); ax.set_xlabel("Delivery time (minutes)"); ax.set_ylabel("Orders"); ax.grid(axis="y",alpha=.2); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.download_button("Download dataset CSV", df.to_csv(index=False), "delivery_data.csv", "text/csv")

else:
    st.markdown('<div class="section">Project overview</div>', unsafe_allow_html=True)
    st.write("**Problem statement:** Estimate delivery duration using order and delivery conditions. This is a supervised regression problem because the output is a continuous numeric value.")
    st.write("**Input features:** distance, traffic, weather, nearby partner availability, preparation time and order type.")
    st.write("**Target variable:** delivery_time_min.")
    st.write("**Pipeline:** Dataset → preprocessing → 80/20 train-test split → Linear Regression and Random Forest → MAE/RMSE/R² → prediction interface.")
    st.write("**Preprocessing:** median imputation and standard scaling for numeric inputs; most-frequent imputation and one-hot encoding for categorical inputs.")
    st.write("**Developer:** Saurabh Charbhare · B.Tech Computer Science & Engineering · DBATU Lonere · Academic Year 2026–2027")
    st.markdown('<div class="note">Before submission, document the true source of any replacement dataset, run and record experiments, verify metrics, and be ready to explain every step during viva.</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Delivery Time Prediction · Academic prototype · Prediction is an estimate, not a guaranteed delivery time.")
