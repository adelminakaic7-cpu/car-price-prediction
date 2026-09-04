import os
import pandas as pd

def clean_car_data(df):
    df = df.copy()
    
    # 1. Sređivanje naziva kolona
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace(' ', '_', regex=False)
    )
    
    df = df.rename(columns={
        'mileagekilometers': 'mileage',
        'volumecm3': 'volume'
    })
    
    # 2. Uklanjanje dupliranih redova
    df = df.drop_duplicates()
    
    # 3. Uklanjanje redova bez cijene 
    df = df.dropna(subset=['priceusd'])
    
    # 4. Detaljno filtriranje nerealnih vrijednosti (Outliers)
    df = df[(df['priceusd'] >= 300) & (df['priceusd'] <= 120000)]
    
    if 'year' in df.columns:
        df = df[(df['year'] >= 1980) & (df['year'] <= 2026)]
        
    if 'mileage' in df.columns:
        df = df[(df['mileage'] >= 500) & (df['mileage'] <= 800000)]
        
    if 'volume' in df.columns:
        df = df[(df['volume'] >= 600) & (df['volume'] <= 7000)]
        
    # 5. Sređivanje tekstualnih/kategorijskih kolona
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].replace({'nan': 'unknown', 'none': 'unknown'})
    
    df = df.dropna()
    return df

if __name__ == '__main__':
    RAW_PATH = 'data/cars.csv'
    CLEAN_PATH = 'data/cars_cleaned.csv'
    
    if os.path.exists(RAW_PATH):
        df_raw = pd.read_csv(RAW_PATH)
        df_clean = clean_car_data(df_raw)
        
        # Čuvanje novog očišćenog CSV fajla
        df_clean.to_csv(CLEAN_PATH, index=False)
        
        print(f"Čišćenje završeno!")
        print(f"Sirovi podaci: {len(df_raw)} redova")
        print(f"Očišćeni podaci: {len(df_clean)} redova")
        print(f"Novi fajl je sačuvan na: {CLEAN_PATH}")
    else:
        print(f"Greška: Fajl {RAW_PATH} ne postoji.")