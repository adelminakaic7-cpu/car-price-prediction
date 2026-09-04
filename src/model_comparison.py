import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_cleaning import clean_car_data
from feature_engineering import add_features
from data_preprocessing import prepare_data, build_preprocessing_pipeline
from model_evaluation import evaluate_predictions, print_metrics

def compare():
    # 1. Učitavanje i obrada podataka
    df = pd.read_csv('data/cars_features.csv')
    
    
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(df)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    
    # 2. Modeli koje upoređujemo
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=12),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # 3. Treniranje i evaluacija svakog modela
    results = []
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('algo', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        m = evaluate_predictions(y_test, y_pred)
        m['Model'] = name
        results.append(m)
        print_metrics(name, m)
        
    summary = pd.DataFrame(results)[['Model', 'MAE', 'RMSE', 'R2']].sort_values(by='MAE')
    print("\n=== KONAČNO POREĐENJE MODELA ===")
    print(summary.to_string(index=False))

if __name__ == '__main__':
    compare()