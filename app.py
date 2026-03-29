import os
import re
import base64
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Gujarat House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_gujarat_house_data.csv")

model = load_model()
data = load_data()

cities = sorted(data["city"].dropna().unique().tolist())
property_types = sorted(data["property_type"].dropna().unique().tolist())
furnishing_options = sorted(data["furnishing"].dropna().unique().tolist())
parking_options = sorted(data["parking"].dropna().unique().tolist())

def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = name.replace("/", "_")
    return name

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_background_image(city_name, locality_name=None):
    # 1. locality image
    if locality_name:
        locality_file = os.path.join("assets", f"{safe_filename(locality_name)}.jpg")
        if os.path.exists(locality_file):
            return locality_file

    # 2. city image
    city_file = os.path.join("assets", f"{safe_filename(city_name)}.jpg")
    if os.path.exists(city_file):
        return city_file

    # 3. default image
    return os.path.join("assets", "default.jpg")

def apply_background(image_b64):
    if image_b64 is None:
        return

    st.markdown(f"""
    <style>
    .stApp {{
        background:
            linear-gradient(rgba(4, 10, 20, 0.72), rgba(4, 10, 20, 0.82)),
            url("data:image/jpg;base64,{image_b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        z-index: -1;
    }}

    .block-container {{
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .hero-title {{
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }}

    .hero-subtitle {{
        color: #d1d5db;
        font-size: 1.08rem;
        margin-bottom: 1.4rem;
    }}

    .glass-card {{
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.28);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        margin-bottom: 1rem;
    }}

    .metric-green {{
        background: linear-gradient(135deg, rgba(34,197,94,0.90), rgba(21,128,61,0.58));
        border-radius: 22px;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}

    .metric-blue {{
        background: linear-gradient(135deg, rgba(59,130,246,0.90), rgba(29,78,216,0.58));
        border-radius: 22px;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}

    .metric-purple {{
        background: linear-gradient(135deg, rgba(168,85,247,0.92), rgba(107,33,168,0.58));
        border-radius: 22px;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}

    .metric-label {{
        font-size: 0.95rem;
        opacity: 0.9;
    }}

    .metric-value {{
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# Default render
default_city = cities[0] if cities else "Ahmedabad"
default_localities = sorted(
    data.loc[data["city"] == default_city, "locality"].dropna().unique().tolist()
)
default_locality = default_localities[0] if default_localities else None
default_bg = get_background_image(default_city, default_locality)
apply_background(get_base64_image(default_bg))

st.markdown('<div class="hero-title">🏠 Gujarat House Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Professional city-wise and locality-wise property prediction with premium glass UI</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.05, 0.95])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    city = st.selectbox("Select City", cities)

    localities = sorted(
        data.loc[data["city"] == city, "locality"].dropna().unique().tolist()
    )
    locality = st.selectbox("Select Locality", localities)

    # Apply changed background after selection
    current_bg = get_background_image(city, locality)
    apply_background(get_base64_image(current_bg))

    property_type = st.selectbox("Property Type", property_types)
    bhk = st.slider("BHK", 1, 10, 2)
    area = st.number_input("Area (SqFt)", min_value=300, max_value=10000, value=1200, step=50)
    furnishing = st.selectbox("Furnishing", furnishing_options)
    age = st.slider("Property Age (Years)", 0, 50, 5)
    parking = st.selectbox("Parking", parking_options)

    predict_btn = st.button("Predict Price", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("About this app")
    st.write(
        "This premium ML app predicts Gujarat property prices using city, locality, "
        "property type, BHK, area, furnishing, age, and parking."
    )
    st.write(
        "It compares the same property across cities, generates downloadable reports, "
        "and stores prediction history."
    )
    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "city": city,
        "locality": locality,
        "property_type": property_type,
        "bhk": bhk,
        "area": area,
        "furnishing": furnishing,
        "age": age,
        "parking": parking
    }])

    prediction = max(model.predict(input_data)[0], 0)
    price_crore = prediction / 100
    price_per_sqft = (prediction * 100000) / area

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-green">
            <div class="metric-label">Estimated Price</div>
            <div class="metric-value">₹ {prediction:.2f} Lakhs</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-blue">
            <div class="metric-label">Price in Crores</div>
            <div class="metric-value">₹ {price_crore:.2f} Cr</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-purple">
            <div class="metric-label">Price per SqFt</div>
            <div class="metric-value">₹ {price_per_sqft:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Entered Details")
    st.dataframe(input_data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    comparison_data = []
    for c in cities:
        city_localities = sorted(
            data.loc[data["city"] == c, "locality"].dropna().unique().tolist()
        )
        locality_for_city = locality if locality in city_localities else city_localities[0]

        temp_input = input_data.copy()
        temp_input["city"] = c
        temp_input["locality"] = locality_for_city

        city_price = max(model.predict(temp_input)[0], 0)

        comparison_data.append({
            "City": c,
            "Locality Used": locality_for_city,
            "Predicted Price (Lakhs)": round(city_price, 2),
            "Predicted Price (Cr)": round(city_price / 100, 2)
        })

    df_compare = pd.DataFrame(comparison_data).sort_values(
        by="Predicted Price (Lakhs)", ascending=True
    ).reset_index(drop=True)

    g1, g2 = st.columns([1.2, 0.8])

    with g1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Price Comparison Across Cities")
        st.dataframe(df_compare, use_container_width=True)
        st.subheader("📈 Price Comparison Graph")
        st.bar_chart(df_compare.set_index("City")[["Predicted Price (Lakhs)"]])
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        best_city = df_compare.loc[df_compare["Predicted Price (Lakhs)"].idxmin(), "City"]
        highest_city = df_compare.loc[df_compare["Predicted Price (Lakhs)"].idxmax(), "City"]

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Insights")
        st.success(f"💡 Best City for Lower Budget Investment: {best_city}")
        st.warning(f"🏙️ Highest Predicted Price City: {highest_city}")
        st.markdown('</div>', unsafe_allow_html=True)

    history_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": city,
        "locality": locality,
        "property_type": property_type,
        "bhk": bhk,
        "area": area,
        "furnishing": furnishing,
        "age": age,
        "parking": parking,
        "predicted_price_lakhs": round(prediction, 2),
        "predicted_price_crore": round(price_crore, 2),
        "price_per_sqft": round(price_per_sqft, 2)
    }])

    history_file = "prediction_history.csv"
    file_exists = os.path.exists(history_file)
    history_row.to_csv(history_file, mode="a", header=not file_exists, index=False)

    report_text = f"""
GUJARAT HOUSE PRICE PREDICTION REPORT
-------------------------------------

Generated On: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

INPUT DETAILS
-------------
City: {city}
Locality: {locality}
Property Type: {property_type}
BHK: {bhk}
Area (SqFt): {area}
Furnishing: {furnishing}
Property Age (Years): {age}
Parking: {parking}

PREDICTION
----------
Estimated Price: ₹ {prediction:.2f} Lakhs
Estimated Price in Crores: ₹ {price_crore:.2f} Cr
Price per SqFt: ₹ {price_per_sqft:.2f}

CITY COMPARISON
---------------
{df_compare.to_string(index=False)}
"""

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📄 Download Prediction Report")
    st.download_button(
        label="⬇️ Download Report (.txt)",
        data=report_text,
        file_name="house_price_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    if os.path.exists(history_file):
        with open(history_file, "rb") as f:
            st.download_button(
                label="⬇️ Download Prediction History (.csv)",
                data=f,
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn | Premium Gujarat Property Prediction App")