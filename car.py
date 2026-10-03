import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE CONFIG & STYLING (UI/UX DESIGN)
# ==========================================
st.set_page_config(
    page_title="AutoValuate | Used Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI/UX polish
st.markdown("""
<style>
    /* Global Styling & Variables */
    :root {
        --primary-color: #2563eb;
        --card-bg: #1e293b;
        --border-color: #334155;
        --text-muted: #94a3b8;
    }
    
    /* Hide Default Streamlit Menu Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header Container */
    .header-container {
        padding: 1rem 0rem 1rem 0rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: var(--text-muted);
        font-size: 1rem;
    }

    /* Metric Card Styling */
    .metric-card {
        background: #0f172a;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Summary Tags */
    .tag-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .tag-pill {
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Custom Primary Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATASET & ML MODEL ENGINE
# ==========================================
@st.cache_data
def generate_market_data():
    np.random.seed(42)
    n_samples = 1500

    brands = {
        'Maruti': ['Swift', 'Baleno', 'Dzire'], 
        'Hyundai': ['i20', 'Creta', 'Verna'], 
        'Honda': ['City', 'Civic', 'Amaze'],
        'Toyota': ['Fortuner', 'Innova', 'Glanza']
    }
    
    brand_list, model_list = [], []
    for _ in range(n_samples):
        b = np.random.choice(list(brands.keys()))
        m = np.random.choice(brands[b])
        brand_list.append(b)
        model_list.append(m)

    years = np.random.randint(2013, 2027, n_samples)
    kms_driven = np.random.randint(5000, 160000, n_samples)
    fuel_types = np.random.choice(['Petrol', 'Diesel', 'CNG'], n_samples)
    transmissions = np.random.choice(['Manual', 'Automatic'], n_samples)
    engine_size = np.random.choice([1.0, 1.2, 1.5, 2.0, 2.8], n_samples)

    prices = (years - 2000) * 90000 + (engine_size * 200000) - (kms_driven * 1.1)
    prices += np.where(fuel_types == 'Diesel', 70000, 0) + np.where(transmissions == 'Automatic', 90000, 0)
    prices += np.random.normal(0, 35000, n_samples)
    prices = np.clip(prices, 180000, 6000000)

    return pd.DataFrame({
        'Brand': brand_list,
        'Model': model_list,
        'Year': years,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_types,
        'Transmission': transmissions,
        'Engine_Size': engine_size,
        'Price': prices
    })

df = generate_market_data()

# Data Preprocessing & Model Training
df_encoded = pd.get_dummies(df.drop(columns=['Model']), columns=['Brand', 'Fuel_Type', 'Transmission'], drop_first=False)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 3. SIDEBAR CONTROLS (INPUT ARCHITECTURE)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car.png", width=60)
    st.title("Vehicle Configurator")
    st.caption("Select parameters to run prediction model")
    st.markdown("---")

    selected_brand = st.selectbox("Manufacturer", df['Brand'].unique())
    brand_models = df[df['Brand'] == selected_brand]['Model'].unique()
    selected_model = st.selectbox("Model Series", brand_models)

    col_side1, col_side2 = st.columns(2)
    with col_side1:
        selected_year = st.slider("Model Year", 2013, 2026, 2020)
    with col_side2:
        selected_engine = st.select_slider("Engine (L)", options=[1.0, 1.2, 1.5, 2.0, 2.8], value=1.5)

    selected_kms = st.number_input("Odometer (KM)", min_value=1000, max_value=300000, value=35000, step=5000)
    selected_fuel = st.radio("Fuel System", ['Petrol', 'Diesel', 'CNG'], horizontal=True)
    selected_transmission = st.radio("Transmission", ['Manual', 'Automatic'], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("✨ Calculate Price", use_container_width=True)

# ==========================================
# 4. MAIN CANVAS (DASHBOARD LAYOUT)
# ==========================================

# Top Header Area
st.markdown("""
<div class="header-container">
    <div class="header-title">AutoValuate Market Intelligence</div>
    <div class="header-subtitle">Real-time valuation dashboard powered by Random Forest Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Process user input for prediction
user_input = pd.DataFrame({
    'Year': [selected_year],
    'Kms_Driven': [selected_kms],
    'Engine_Size': [selected_engine],
    'Brand_Honda': [1 if selected_brand == 'Honda' else 0],
    'Brand_Hyundai': [1 if selected_brand == 'Hyundai' else 0],
    'Brand_Maruti': [1 if selected_brand == 'Maruti' else 0],
    'Brand_Toyota': [1 if selected_brand == 'Toyota' else 0],
    'Fuel_Type_CNG': [1 if selected_fuel == 'CNG' else 0],
    'Fuel_Type_Diesel': [1 if selected_fuel == 'Diesel' else 0],
    'Fuel_Type_Petrol': [1 if selected_fuel == 'Petrol' else 0],
    'Transmission_Automatic': [1 if selected_transmission == 'Automatic' else 0],
    'Transmission_Manual': [1 if selected_transmission == 'Manual' else 0]
}).reindex(columns=X.columns, fill_value=0)

predicted_price = model.predict(user_input)[0]

# Dashboard Split Columns
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.subheader("💡 Estimated Valuation")
    
    # Valuation Display Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #3b82f6; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.2);">
        <div style="color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Market Value</div>
        <div style="color: #60a5fa; font-size: 3rem; font-weight: 800; margin: 0.4rem 0;">₹{predicted_price:,.2f}</div>
        <div style="color: #cbd5e1; font-size: 0.85rem;">Estimated for <b>{selected_year} {selected_brand} {selected_model}</b></div>
        <div class="tag-container">
            <span class="tag-pill">⚙️ {selected_engine}L Engine</span>
            <span class="tag-pill">⛽ {selected_fuel}</span>
            <span class="tag-pill">🕹️ {selected_transmission}</span>
            <span class="tag-pill">🛣️ {selected_kms:,} KM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Floor / Ceiling Metrics
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Suggested Floor Price</div>
            <div class="metric-value" style="color: #34d399;">₹{predicted_price * 0.94:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Suggested Ceiling Price</div>
            <div class="metric-value" style="color: #f43f5e;">₹{predicted_price * 1.06:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.subheader("📈 Analytics & Market Insights")
    
    tab_depreciation, tab_importance = st.tabs(["📉 Price vs Mileage Trend", "🎯 Key Pricing Drivers"])
    
    with tab_depreciation:
        filtered_df = df[df['Brand'] == selected_brand]
        fig = px.scatter(
            filtered_df, 
            x="Kms_Driven", 
            y="Price", 
            color="Fuel_Type",
            hover_data=['Year', 'Model'],
            labels={"Kms_Driven": "Odometer (km)", "Price": "Price (₹)"},
            template="plotly_dark"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_importance:
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(6)
        fig_imp = px.bar(
            x=importances.values, 
            y=importances.index, 
            orientation='h',
            labels={'x': 'Relative Impact Weight', 'y': 'Feature'},
            template="plotly_dark",
            color_discrete_sequence=['#3b82f6']
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)