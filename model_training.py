import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

def train_model(data_path: str, model_path: str):
    """Train XGBoost regressor and save the model artifact."""
    df = pd.read_csv(data_path)
    features = ['PULocationID', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
    
    X = df[features]
    y = df['demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Validation MAE: {mae:.2f}")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    INPUT_DATA = "processed_demand_data.csv"
    OUTPUT_MODEL = "taxi_demand_model.pkl"
    train_model(INPUT_DATA, OUTPUT_MODEL)