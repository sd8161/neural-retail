import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df):
    # Features & target
    X = df[['Recency', 'Frequency', 'Monetary']]
    y = df['Churn']

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, "models/churn_model.pkl")

    return model



if __name__ == "__main__":
    df = pd.read_csv("data/processed/rfm_segmented.csv")
    train_model(df)