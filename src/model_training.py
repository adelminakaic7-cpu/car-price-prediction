import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from data_cleaning import clean_car_data
from feature_engineering import add_features
from data_preprocessing import prepare_data, build_preprocessing_pipeline
from model_evaluation import evaluate_predictions, print_metrics, get_sample_table

def train_and_save():
    df = pd.read_csv('data/cars_features.csv')
    
    
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(df)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # Pobjednički model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    
    print("Treniranje finalnog modela...")
    pipeline.fit(X_train, y_train)
    
    # Evaluacija
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    print_metrics("Finalni Random Forest Model", metrics)
    
    print("\nPrimeri predviđanja:")
    print(get_sample_table(y_test, y_pred, n=5).to_string(index=False))
    
    # Čuvanje modela
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/car_price_model.joblib')
    print("\nModel je uspešno sačuvan u: models/car_price_model.joblib")

if __name__ == '__main__':
    train_and_save()