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
        
    # 3. Zapremina motora u litrima (npr. 2000 cm3 -> 2.0L)
    if 'volume' in df.columns:
        df['engine_volume_liters'] = df['volume'] / 1000.0
        

    return df

if __name__ == '__main__':
    # 1. Učitavanje očišćenog fajla
    df_clean = pd.read_csv('data/cars_cleaned.csv')
    
    # 2. Dodavanje novih kolona
    df_features = add_features(df_clean)
    
    # 3. Snimanje u novi CSV fajl
    output_path = 'data/cars_features.csv'
    df_features.to_csv(output_path, index=False)
    
    print("Inženjering karakteristika je uspješno završen!")
    print(f"Novi fajl sačuvan u: {output_path}")
    print(f"Nove kolone u skupu: {[c for c in df_features.columns if c not in df_clean.columns]}")