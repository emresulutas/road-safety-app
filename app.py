import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

# Lovable'deki tasarımın geliştirilmiş hali
st.markdown("""
<style>
    /* Arkaplan gradient - Lovable'deki gibi */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Ana konteyner */
    .main-container {
        background: white;
        border-radius: 24px;
        padding: 40px;
        margin: 30px auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        max-width: 1200px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Header - BEYAZ YAPILDI */
    .main-header {
        text-align: center;
        color: white !important;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .sub-header {
        text-align: center;
        color: white !important;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 40px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Başlık beyaz yapıldı */
    .section-title {
        color: white !important;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .section-subtitle {
        color: white !important;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }

    /* Yol kartları - Geliştirilmiş */
    .road-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .road-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        border-color: #667eea;
    }

    .road-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
    }

    .road-title {
        color: #2d3748;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .road-feature {
        display: flex;
        align-items: center;
        margin: 12px 0;
        color: #4a5568;
        font-size: 1rem;
    }

    .road-feature-icon {
        margin-right: 12px;
        font-size: 1.2rem;
    }

    /* Buton stilleri - Geliştirilmiş */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 16px 40px;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }

    /* Radio butonları - DÜZELTİLDİ */
    div[data-testid="stRadio"] > label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Skor göstergesi */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 30px 0;
        color: white;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }

    .risk-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        margin: 10px 0;
    }

    /* Selectbox stilleri */
    .stSelectbox label {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
</style>
""", unsafe_allow_html=True)


# Model yükleme fonksiyonu - basit fallback ile
@st.cache_resource
def load_model():
    try:
        model_path = "C:\\Users\\ASUS\\Desktop\\intuition-vs-ai\\model\\accident_model.pkl"

        # Özel sınıf tanımı olmadan yüklemeyi dene
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Kaggle modeli başarıyla yüklendi!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Model yüklenemedi, basit model kullanılıyor: {e}")
        return None


# Geliştirilmiş basit model (daha gerçekçi risk hesaplama)
def calculate_risk_simple(lighting, weather, curvature, speed_limit, accidents):
    lighting_map = {"🌙 Night": 0.8, "🌆 Dusk": 0.5, "☀️ Day": 0.2}
    weather_map = {"🌧️ Rainy": 0.7, "☁️ Cloudy": 0.4, "🌈 Clear": 0.1}

    # Daha gerçekçi ağırlıklar
    risk = (
            lighting_map[lighting] * 0.3 +
            weather_map[weather] * 0.25 +
            curvature * 0.25 +
            (speed_limit / 120) * 0.15 +
            (accidents / 10) * 0.05
    )
    return min(max(risk, 0), 1)


# Risk hesaplama fonksiyonu
def calculate_risk(lighting, weather, curvature, speed_limit, accidents):
    model = load_model()

    if model is None:
        return calculate_risk_simple(lighting, weather, curvature, speed_limit, accidents)

    try:
        # Model için feature hazırlama
        lighting_map = {"🌙 Night": 0, "🌆 Dusk": 1, "☀️ Day": 2}
        weather_map = {"🌧️ Rainy": 1, "☁️ Cloudy": 2, "🌈 Clear": 0}

        features = np.array([[
            lighting_map[lighting],
            weather_map[weather],
            curvature,
            speed_limit,
            accidents
        ]])

        prediction = model.predict(features)[0]
        # Tahmini 0-1 aralığına normalize et
        normalized_risk = min(max(prediction, 0), 1)
        return normalized_risk

    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
        return calculate_risk_simple(lighting, weather, curvature, speed_limit, accidents)


# Başlık - BEYAZ YAPILDI
st.markdown('<div class="main-header">🚗 Road Safety Challenge</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Test your intuition against AI predictions</div>', unsafe_allow_html=True)

# Ana içerik
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Oyun başlığı - BEYAZ YAPILDI
    st.markdown('<div class="section-title">🎯 Which road is safer?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Select the road you think has lower accident risk</div>',
                unsafe_allow_html=True)

    # İki yol seçeneği
    col1, col2 = st.columns(2)

    with col1:
        # Yol A için interaktif seçimler
        st.markdown('<div class="road-card">', unsafe_allow_html=True)
        st.markdown('<div class="road-title">🛣️ Road A</div>', unsafe_allow_html=True)

        a_lighting = st.selectbox("Lighting", ["🌙 Night", "🌆 Dusk", "☀️ Day"], key="a_light")
        a_weather = st.selectbox("Weather", ["🌧️ Rainy", "☁️ Cloudy", "🌈 Clear"], key="a_weather")
        a_curvature = st.slider("Curvature", 0.1, 1.0, 0.3, key="a_curve")
        a_speed = st.slider("Speed Limit", 30, 120, 90, key="a_speed")
        a_accidents = st.slider("Accident History", 1, 10, 5, key="a_accidents")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Yol B için interaktif seçimler
        st.markdown('<div class="road-card">', unsafe_allow_html=True)
        st.markdown('<div class="road-title">🛣️ Road B</div>', unsafe_allow_html=True)

        b_lighting = st.selectbox("Lighting", ["🌙 Night", "🌆 Dusk", "☀️ Day"], key="b_light")
        b_weather = st.selectbox("Weather", ["🌧️ Rainy", "☁️ Cloudy", "🌈 Clear"], key="b_weather")
        b_curvature = st.slider("Curvature", 0.1, 1.0, 0.7, key="b_curve")
        b_speed = st.slider("Speed Limit", 30, 120, 60, key="b_speed")
        b_accidents = st.slider("Accident History", 1, 10, 3, key="b_accidents")

        st.markdown("</div>", unsafe_allow_html=True)

    # Seçim - DÜZELTİLDİ
    st.markdown("---")
    st.markdown("**Select the SAFER road:**")

    # Radio butonları düzgün gösterilsin diye
    choice = st.radio(
        "",  # Boş label
        ["Road A", "Road B"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Gönder butonu
    if st.button("🚀 Analyze with AI", use_container_width=True):
        # Risk hesaplamaları
        risk_a = calculate_risk(a_lighting, a_weather, a_curvature, a_speed, a_accidents)
        risk_b = calculate_risk(b_lighting, b_weather, b_curvature, b_speed, b_accidents)

        # Sonuç belirleme
        safer_road = "A" if risk_a < risk_b else "B"
        user_correct = (choice == f"Road {safer_road}")

        # Sonuç gösterimi
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if user_correct:
            st.success("🎉 Correct! Your intuition matches AI prediction")
        else:
            st.error("🤖 AI suggests a different safer road")

        st.markdown("### 📊 Risk Analysis")

        col3, col4 = st.columns(2)
        with col3:
            st.metric("🛣️ Road A Risk", f"{risk_a * 100:.1f}%")
        with col4:
            st.metric("🛣️ Road B Risk", f"{risk_b * 100:.1f}%")

        st.markdown(f"**🎯 Safer Road: Road {safer_road}**")

        # Risk seviyesi
        max_risk = max(risk_a, risk_b)
        if max_risk > 0.7:
            risk_level = "🔴 High Risk"
        elif max_risk > 0.4:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🟢 Low Risk"

        st.markdown(f'<div class="risk-badge">{risk_level}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Açıklama
        with st.expander("🔍 How does AI calculate risk?"):
            st.markdown("""
            **Machine Learning Model Analysis:**
            - **Lighting Conditions**: Night → Higher risk
            - **Weather**: Rainy → Higher risk  
            - **Road Curvature**: More curves → Higher risk
            - **Speed Limit**: Higher speed → Higher risk
            - **Accident History**: More accidents → Higher risk

            The model analyzes these 5 key factors using trained algorithms to predict accident probability.
            """)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: white;'>"
    "Built with ❤️ using Streamlit | Machine Learning Powered"
    "</div>",
    unsafe_allow_html=True
)