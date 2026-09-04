import pandas as pd

def add_features(df):
    df = df.copy()
    
    # 1. Starost automobila
    if 'year' in df.columns:
        df['car_age'] = 2026 - df['year']
        # Ako je auto iz 2026. godine, stavljamo da je starost 1 godina da ne bismo dijelili sa nulom
        df['car_age'] = df['car_age'].replace(0, 1)
        
    # 2. Prosečna kilometraža po godini
    if 'mileage' in df.columns and 'car_age' in df.columns:
        df['mileage_per_year'] = df['mileage'] / df['car_age']
        
    # 3. Zapremina motora u litrima 
    if 'volume' in df.columns:
        df['engine_volume_liters'] = df['volume'] / 1000.0
        
    return df

# Testiranje skripte u terminalu
if __name__ == '__main__':
    from data_cleaning import clean_car_data
    
    df_raw = pd.read_csv('data/cars_cleaned.csv')
    df_clean = clean_car_data(df_raw)
    df_featured = add_features(df_clean)
    
    print("Nove kolone uspješno dodane!")
    print("Kolone u tabeli:", list(df_featured.columns))