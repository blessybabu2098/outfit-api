# outfit_api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------- 1. CREATE THE APP ----------
app = FastAPI()

# ---------- 2. LOAD DATA + TRAIN MODEL ONCE ----------
data = pd.read_csv("outfit_data.csv")

weather_encoder = LabelEncoder()
outfit_encoder = LabelEncoder()

data["Weather_encoded"] = weather_encoder.fit_transform(data["Weather"])
data["Outfit_encoded"] = outfit_encoder.fit_transform(data["Outfit"])

X = data[["Temperature", "Weather_encoded"]]
y = data["Outfit_encoded"]

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------- 3. REQUEST + RESPONSE SHAPES ----------

class OutfitRequest(BaseModel):
    temp: float
    weather: str

class OutfitResponse(BaseModel):
    outfit: str

# ---------- 4. PREDICT ENDPOINT ----------

@app.post("/predict", response_model=OutfitResponse)
def predict_outfit(request: OutfitRequest):
    # clean weather text like " sunny " -> "Sunny"
    weather_clean = request.weather.strip().title()

    # turn weather into number
    weather_num = weather_encoder.transform([weather_clean])[0]

    # predict encoded outfit
    pred_code = model.predict([[request.temp, weather_num]])[0]

    # turn number back into outfit name
    outfit = outfit_encoder.inverse_transform([pred_code])[0]

    return OutfitResponse(outfit=outfit)
