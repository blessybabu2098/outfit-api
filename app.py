import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 1) LOAD DATA + TRAIN MODEL (one time)
@st.cache_data
def load_model():
    # read your CSV (must be in the same folder as app.py)
    data = pd.read_csv("outfit_data.csv")

    # encode text columns to numbers
    weather_encoder = LabelEncoder()
    outfit_encoder = LabelEncoder()

    data["Weather_encoded"] = weather_encoder.fit_transform(data["Weather"])
    data["Outfit_encoded"] = outfit_encoder.fit_transform(data["Outfit"])

    # features (inputs) and target (output)
    X = data[["Temperature", "Weather_encoded"]]
    y = data["Outfit_encoded"]

    # train a simple decision tree
    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model, weather_encoder, outfit_encoder, data

model, weather_encoder, outfit_encoder, data = load_model()

# 2) STREAMLIT UI

st.set_page_config(page_title="Outfit Recommender", page_icon="👗")

st.title("👗 Outfit Recommender")
st.write("Get a simple outfit suggestion based on temperature and weather.")

# Show the dataset (optional)
#with st.expander("See training data"):
    #st.dataframe(data)

# Inputs from user
temp = st.number_input("🌡️ Enter Temperature (°C)", min_value=-10.0, max_value=50.0, value=28.0, step=0.5)

weather = st.selectbox(
    "☀️ Select Weather Condition",
    ["Sunny", "Cloudy", "Rainy", "Cold"]
)

# When button is clicked, predict
if st.button("✨ Get Outfit"):
    # clean weather text
    weather_clean = weather.strip().title()

    # encode weather
    weather_num = weather_encoder.transform([weather_clean])[0]

    # predict encoded outfit
    pred_code = model.predict([[temp, weather_num]])[0]

    # decode outfit back to text
    outfit = outfit_encoder.inverse_transform([pred_code])[0]

    st.success(f"Recommended Outfit: **{outfit}**")