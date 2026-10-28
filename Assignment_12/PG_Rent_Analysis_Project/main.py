"""
Student PG Rent Analysis and Prediction
Experiment 12: Real-world problem-solving mini project.

Dataset note:
The included CSV is synthetic demonstration data designed for coursework.
Replace it with verified, locally collected PG listings before making real decisions.

Run:
    pip install pandas numpy matplotlib scikit-learn
    python main.py

The script reads pg_rent_dataset.csv from the same folder, prints descriptive
analysis, trains regression/classification models, saves three graphs in graphs/,
and accepts an optional rent prediction input.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "pg_rent_dataset.csv"
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "Distance_km", "Sharing_Capacity", "Current_Residents",
    "WiFi", "Laundry", "Meals_Included", "AC", "Security"
]
CATEGORICAL_FEATURES = ["Accommodation_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Monthly_Rent_Rs"

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET, "PG_ID", "Location"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    return data

def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])

def save_graphs(data, median_rent):
    # 1. Rent distribution
    plt.figure(figsize=(8, 5))
    plt.hist(data[TARGET], bins=12, edgecolor="black")
    plt.axvline(median_rent, linestyle="--", label=f"Median = Rs. {median_rent:,.0f}")
    plt.title("Monthly PG Rent Distribution")
    plt.xlabel("Monthly rent per person (Rs.)")
    plt.ylabel("Number of PG listings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "01_rent_distribution.png", dpi=160)
    plt.close()

    # 2. Rent vs distance by accommodation type
    plt.figure(figsize=(8, 5))
    for accommodation in sorted(data["Accommodation_Type"].dropna().unique()):
        subset = data[data["Accommodation_Type"] == accommodation]
        plt.scatter(subset["Distance_km"], subset[TARGET], label=accommodation, alpha=0.8)
    plt.title("PG Rent vs Distance from College")
    plt.xlabel("Distance from college (km)")
    plt.ylabel("Monthly rent per person (Rs.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "02_rent_vs_distance.png", dpi=160)
    plt.close()

    # 3. Mean rent by sharing capacity
    average = data.groupby("Sharing_Capacity")[TARGET].mean().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(average.index.astype(str), average.values, edgecolor="black")
    plt.title("Average Rent by Room-Sharing Capacity")
    plt.xlabel("Number of people sharing")
    plt.ylabel("Average monthly rent (Rs.)")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "03_rent_by_sharing.png", dpi=160)
    plt.close()

def main():
    data = load_data()
    median_rent = float(data[TARGET].median())
    data["Rent_Category"] = np.where(data[TARGET] <= median_rent, "Affordable", "Expensive")

    print("=" * 62)
    print("STUDENT PG RENT ANALYSIS AND PREDICTION")
    print("=" * 62)
    print(f"Number of listings: {len(data)}")
    print(f"Median monthly rent: Rs. {median_rent:,.0f}")
    print(f"Mean monthly rent: Rs. {data[TARGET].mean():,.2f}")
    print(f"Minimum / maximum rent: Rs. {data[TARGET].min():,.0f} / Rs. {data[TARGET].max():,.0f}")
    print("\nAccommodation counts:")
    print(data["Accommodation_Type"].value_counts().to_string())
    print("\nAverage rent by accommodation type:")
    print(data.groupby("Accommodation_Type")[TARGET].mean().round(2).to_string())
    print("\nAverage rent by sharing capacity:")
    print(data.groupby("Sharing_Capacity")[TARGET].mean().round(2).to_string())
    print("\nAffordable/expensive counts:")
    print(data["Rent_Category"].value_counts().to_string())

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    regression = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=250, random_state=42))
    ])
    regression.fit(X_train, y_train)
    predictions = regression.predict(X_test)
    print("\n--- Rent prediction model (Random Forest Regression) ---")
    print(f"MAE:  Rs. {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"RMSE: Rs. {np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
    print(f"R-squared: {r2_score(y_test, predictions):.3f}")

    # Classify affordability using the median rent from the full demo dataset.
    y_class = data["Rent_Category"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.25, random_state=42, stratify=y_class
    )
    classifier = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    classifier.fit(Xc_train, yc_train)
    class_predictions = classifier.predict(Xc_test)
    print("\n--- Affordability classification (Logistic Regression) ---")
    print(f"Accuracy: {accuracy_score(yc_test, class_predictions):.3f}")
    print(classification_report(yc_test, class_predictions, zero_division=0))

    save_graphs(data, median_rent)
    print(f"Graphs saved to: {GRAPH_DIR}")

    print("\n--- Try a rent prediction ---")
    print("Enter values when prompted, or press Enter to skip.")
    try:
        distance = input("Distance from college in km [e.g. 1.2]: ").strip()
        if distance:
            sharing = int(input("Sharing capacity [1-5]: ").strip())
            residents = int(input("Current residents: ").strip())
            accommodation = input("Accommodation type [Boys/Girls/Co-ed]: ").strip().title()
            if accommodation not in {"Boys", "Girls", "Co-Ed"}:
                raise ValueError("Accommodation type must be Boys, Girls, or Co-ed.")
            values = {
                "Distance_km": float(distance),
                "Sharing_Capacity": sharing,
                "Current_Residents": residents,
                "WiFi": int(input("WiFi available? 1=yes, 0=no: ").strip()),
                "Laundry": int(input("Laundry available? 1=yes, 0=no: ").strip()),
                "Meals_Included": int(input("Meals included? 1=yes, 0=no: ").strip()),
                "AC": int(input("AC available? 1=yes, 0=no: ").strip()),
                "Security": int(input("Security available? 1=yes, 0=no: ").strip()),
                "Accommodation_Type": "Co-ed" if accommodation == "Co-Ed" else accommodation
            }
            sample = pd.DataFrame([values], columns=FEATURES)
            estimated_rent = float(regression.predict(sample)[0])
            category = "Affordable" if estimated_rent <= median_rent else "Expensive"
            print(f"\nEstimated monthly rent per person: Rs. {estimated_rent:,.0f}")
            print(f"Dataset-relative category: {category} (median threshold Rs. {median_rent:,.0f})")
    except (ValueError, EOFError) as error:
        print(f"Prediction skipped: {error}")

if __name__ == "__main__":
    main()
