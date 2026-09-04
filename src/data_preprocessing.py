import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(num_cols, cat_cols):
    # Obrada numeričkih podataka (popunjavanje praznina + skaliranje)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Obrada tekstualnih/kategorijskih podataka (popunjavanje + OneHotEncoding)
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Spajanje u jedan pretprocesor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    return preprocessor

def prepare_data(df, target_col='priceusd'):
    # Razdvajanje ulaznih karakteristika (X) i ciljne kolone koju predviđamo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Prepoznavanje tipova kolona
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Podjela na trening i test skup
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, num_cols, cat_cols

if __name__ == '__main__':
    import pandas as pd
    
    # Testno učitavanje i provjera funkcija
    df = pd.read_csv('data/cars_features.csv')
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(df)
    
    print("=== TEST PRETPROCESIRANJA ===")
    print(f"Numeričke kolone ({len(num_cols)}):", num_cols)
    print(f"Kategorijske kolone ({len(cat_cols)}):", cat_cols)
    print(f"Dimenzije X_train skupa: {X_train.shape}")
    print(f"Dimenzije X_test skupa: {X_test.shape}")