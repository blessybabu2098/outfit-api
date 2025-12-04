# 1) IMPORT NECESSARY TOOLS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 2) LOAD THE CSV DATA
data = pd.read_csv("outfit_data.csv")

print("Here is the dataset:")
print(data)

# 3) ENCODE TEXT LABELS (Sunny, Cloudy etc.) INTO NUMBERS
weather_encoder = LabelEncoder()
outfit_encoder = LabelEncoder()

data["Weather_encoded"] = weather_encoder.fit_transform(data["Weather"])
data["Outfit_encoded"] = outfit_encoder.fit_transform(data["Outfit"])

# 4) DEFINE INPUT (X) AND OUTPUT (y)
X = data[["Temperature", "Weather_encoded"]]   # inputs
y = data["Outfit_encoded"]                     # output

# 5) CREATE THE MODEL
model = DecisionTreeClassifier()

# 6) TRAIN THE MODEL
model.fit(X, y)

# 7) TAKE USER INPUT: EXAMPLE
temp = float(input("Enter temperature: "))
weather = input("Enter weather (Sunny/Cloudy/Rainy/Cold): ")

# Convert weather input to number
weather_num = weather_encoder.transform([weather])[0]

# 8) PREDICT OUTFIT
predicted_encoded = model.predict([[temp, weather_num]])[0]

predicted_outfit = outfit_encoder.inverse_transform([predicted_encoded])[0]

print("\nRecommended Outfit:")
print(predicted_outfit)
