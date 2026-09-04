import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def print_metrics(model_name, metrics):
    print(f"\n--- Evaluacija: {model_name} ---")
    print(f"MAE  (Prosječna greška u $):  ${metrics['MAE']:.2f}")
    print(f"RMSE (Kvadratna greška u $): ${metrics['RMSE']:.2f}")
    print(f"R²   (Tačnost modela 0-1):    {metrics['R2']:.4f}")

def get_sample_table(y_true, y_pred, n=5):
    # Pravi tabelu za poređenje stvarnih i predviđenih cijena
    df_res = pd.DataFrame({
        'Stvarna cena ($)': y_true.values[:n],
        'Predviđena cena ($)': y_pred[:n]
    })
    df_res['Razlika ($)'] = (df_res['Stvarna cena ($)'] - df_res['Predviđena cena ($)']).abs()
    return df_res