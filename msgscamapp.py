import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- CYBER THEME ----------------
def set_cyber_theme():
    st.markdown("""
    <style>
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #ff0000);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    .cyber-title {
        text-align: center;
        font-size: 52px;
        font-weight: bold;
        color: #ff0000;
        text-shadow:
            0 0 5px #ff0000,
            0 0 15px #ff0000,
            0 0 30px #00ffab,
            0 0 60px #00ffab;
        animation: flicker 1.2s infinite alternate;
    }

    @keyframes flicker {
        0% {opacity: 1;}
        50% {opacity: 0.85;}
        100% {opacity: 1;}
    }

    textarea {
        background: rgba(0,0,0,0.7) !important;
        color: #00ffab !important;
        border: 2px solid #00ffab !important;
        border-radius: 15px !important;
        font-size: 18px !important;
        box-shadow: 0 0 15px #00ffab;
    }

    textarea::placeholder {
        color: #00ffab !important;
        font-weight: bold;
    }

    button {
        background-color: black !important;
        color: #00ffab !important;
        border: 2px solid #00ffab !important;
        border-radius: 15px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        box-shadow: 0 0 15px #00ffab;
        transition: 0.3s;
    }

    button:hover {
        background-color: #00ffab !important;
        color: black !important;
        box-shadow: 0 0 30px #00ffab;
        transform: scale(1.05);
    }

    label, p {
        color: #00ffab !important;
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_cyber_theme()

# ---------------- TITLE ----------------
st.markdown(
    '<div class="cyber-title">🚨 ONLINE SCAM MESSAGE DETECTOR 🚨</div>',
    unsafe_allow_html=True
)

# ---------------- WELCOME TEXT ----------------
st.markdown(
    "<p style='text-align:center; font-size:22px; font-weight:bold; color:#00ffab;'>"
    "Welcome To Online Scam Message Detection App 🔐</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("scam_message.csv")  # columns: message , label

df = load_data()

# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(df):
    X = df["message"]
    y = df["label"]
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_vec, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# ---------------- INPUT ----------------
user_msg = st.text_area(
    "Enter message to check:",
    height=200,
    placeholder="Paste the SMS / WhatsApp / Email message here..."
)

# ---------------- BUTTON ----------------
if st.button("CHECK MESSAGE"):
    if user_msg.strip() == "":
        st.warning("⚠️ Message empty da!")
    else:
        user_vec = vectorizer.transform([user_msg])
        pred = model.predict(user_vec)[0]
        prob = model.predict_proba(user_vec)[0][pred] * 100

        if pred == 1:
            st.error(f"🚨 SCAM DETECTED! ({prob:.2f}%)")
        else:
            st.success(f"✅ SAFE MESSAGE ({prob:.2f}%)")

        # ---------------- THANK YOU MESSAGE ----------------
        st.markdown(
            "<hr style='border:1px solid #00ffab'>"
            "<p style='text-align:center; font-size:20px; font-weight:bold; color:#00ffab;'>"
            "🙏 Thank You for using Online Scam Message Detection App 🙏</p>",
            unsafe_allow_html=True
        )