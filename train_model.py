import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("gujarat_house_data.csv")

# Required columns
required_columns = [
    "City",
    "Locality",
    "Property Type",
    "BHK",
    "Area (SqFt)",
    "Furnishing",
    "Property Age (Years)",
    "Parking",
    "Price (Lakhs)"
]

data = data[required_columns].copy()

# Rename columns
data.columns = [
    "city",
    "locality",
    "property_type",
    "bhk",
    "area",
    "furnishing",
    "age",
    "parking",
    "price"
]

# Clean text columns
text_cols = ["city", "locality", "property_type", "furnishing", "parking"]
for col in text_cols:
    data[col] = data[col].astype(str).str.strip()

# Expanded Gujarat locality mapping
locality_mapping = {
    "Ahmedabad": {
        "Locality_100": "Satellite",
        "Locality_101": "Bopal",
        "Locality_102": "South Bopal",
        "Locality_103": "Chandkheda",
        "Locality_104": "Gota",
        "Locality_105": "Navrangpura",
        "Locality_106": "Maninagar",
        "Locality_107": "SG Highway",
        "Locality_108": "Prahlad Nagar",
        "Locality_109": "Naranpura"
    },
    "Surat": {
        "Locality_100": "Vesu",
        "Locality_101": "Adajan",
        "Locality_102": "Pal",
        "Locality_103": "Piplod",
        "Locality_104": "Althan",
        "Locality_105": "Dumas Road",
        "Locality_106": "Citylight",
        "Locality_107": "Athwa",
        "Locality_108": "Katargam",
        "Locality_109": "Udhna"
    },
    "Vadodara": {
        "Locality_100": "Alkapuri",
        "Locality_101": "Gotri",
        "Locality_102": "Akota",
        "Locality_103": "Vasna Road",
        "Locality_104": "Manjalpur",
        "Locality_105": "Karelibaug",
        "Locality_106": "Harni",
        "Locality_107": "Sama"
    },
    "Rajkot": {
        "Locality_100": "Kalawad Road",
        "Locality_101": "150 Feet Ring Road",
        "Locality_102": "Raiya Road",
        "Locality_103": "Amin Marg",
        "Locality_104": "University Road",
        "Locality_105": "Mavdi",
        "Locality_106": "Nana Mava"
    },
    "Gandhinagar": {
        "Locality_100": "Sector 21",
        "Locality_101": "Sector 22",
        "Locality_102": "Sector 23",
        "Locality_103": "Kudasan",
        "Locality_104": "Raysan",
        "Locality_105": "Sargasan",
        "Locality_106": "Infocity"
    },
    "Bhavnagar": {
        "Locality_100": "Waghawadi Road",
        "Locality_101": "Nilambaug",
        "Locality_102": "Kaliyabid",
        "Locality_103": "Subhashnagar",
        "Locality_104": "Chitra"
    },
    "Jamnagar": {
        "Locality_100": "Patel Colony",
        "Locality_101": "Ranjit Nagar",
        "Locality_102": "Indira Marg",
        "Locality_103": "Gulabnagar",
        "Locality_104": "Digjam Circle"
    },
    "Junagadh": {
        "Locality_100": "Moti Baug",
        "Locality_101": "Joshipura",
        "Locality_102": "Zanzarda Road",
        "Locality_103": "Kalwa Chowk"
    }
}

def fix_locality(row):
    city = row["city"]
    locality = row["locality"]
    if city in locality_mapping and locality in locality_mapping[city]:
        return locality_mapping[city][locality]
    return locality

data["locality"] = data.apply(fix_locality, axis=1)

# Convert numeric columns
num_cols = ["bhk", "area", "age", "price"]
for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove invalid data
data.dropna(inplace=True)
data = data[(data["price"] > 0) & (data["price"] < 200)]

# Save cleaned dataset
data.to_csv("cleaned_gujarat_house_data.csv", index=False)

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

categorical_features = ["city", "locality", "property_type", "furnishing", "parking"]
numeric_features = ["bhk", "area", "age"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2
        ))
    ]
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully.")
print(f"Mean Absolute Error: {mae:.2f} Lakhs")
print(f"R2 Score: {r2:.4f}")

# Save model
joblib.dump(model, "house_model.pkl")
print("Model saved as house_model.pkl")
print("Cleaned dataset saved as cleaned_gujarat_house_data.csv")