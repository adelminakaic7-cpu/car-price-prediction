# 🚗 Predikcija Cijena Polovnih Automobila (Car Price Prediction)

Ovaj projekat predstavlja cjelokupan **Machine Learning pipeline** za predikciju cijena polovnih automobila na osnovu njihovih karakteristika (godište, pređena kilometraža, vrsta goriva, mjenjač, zapremina motora, itd.). 

Projekat je strukturiran po modularnom principu — od eksplorativne analize podataka (EDA) do automatskog pretprocesiranja, treniranja i evaluacije više regresionih modela.

💾 Preuzimanje Sačuvanog Modela (Model Release)
Zbog veličine fajla od 304 MB, sačuvani Random Forest model nije direktno u samom repozitorijumu, već je dostupan za preuzimanje u zvaničnom GitHub Release izdanju:

📥 **[Preuzmi car_price_model.joblib](https://github.com/adelminakaic7-cpu/car-price-prediction/releases/download/v1.0.0/car_price_model.joblib)**

Nakon preuzimanja, stavite .joblib fajl u models/ direktorijum unutar projekta, ili ga regenerišite pokretanjem skripte python src/model_training.py.

---

## 📊 Rezultati i Izbor Modela

Testirano je više regresionih algoritama na testnom skupu podataka. Performanse su mjerene pomoću tri standardne metrike: **MAE** (prosječna apsolutna greška), **RMSE** (kvadratna greška) i **$R^2$** (koeficijent determinacije).

| Model | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** 🏆 | **978.02** | **2108.49** | **0.9222** |
| Gradient Boosting Regressor | 1375.21 | 2519.46 | 0.8889 |
| Decision Tree Regressor | 1297.02 | 2776.90 | 0.8651 |
| Linear Regression | 2030.18 | 3695.41 | 0.7611 |

**Pobjednički model:** **Random Forest Regressor** je izabran kao finalni model jer objašnjava preko **92% varijanse** u cijenama automobila sa prosječnom greškom manjom od **$1.000**.

---

## 📁 Struktura Projekta

```text
car-price-prediction/
│
├── data/
│   ├── cars.csv               # Originalni sirovi podaci
│   ├── cars_cleaned.csv       # Očišćeni podaci
│   └── cars_featured.csv      # Podaci sa novim kreiranim kolonama
│
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis (Jupyter Notebook)
│
├── src/
│   ├── data_cleaning.py       # Čišćenje i filtriranje autlajera
│   ├── feature_engineering.py # Kreiranje novih varijabli (car_age, mileage_per_year, ...)
│   ├── data_preprocessing.py  # Sklearn ColumnTransformer & Pipeline
│   ├── model_evaluation.py   # Pomoćne funkcije za evaluaciju i metrike
│   ├── model_comparison.py   # Skripta za poređenje više modela
│   └── model_training.py     # Treniranje i generisanje modela
│
├── requirements.txt           # Potrebne biblioteke za pokretanje
└── README.md                  # Dokumentacija projekta 


⚙️ Instalacija i Pokretanje

Prije pokretanja skripti, potrebno je instalirati sve potrebne biblioteke navedene u `requirements.txt` fajlu:


pip install -r requirements.txt
```
---

# 🛠️ Pokretanje Cijevovoda (Pipeline)
Skripte se pokreću po redoslijedu kroz terminal:

## Čišćenje podataka:
python src/data_cleaning.py

## Inženjering karakteristika:
python src/feature_engineering.py

## Testiranje pripreme i podjele podataka:
python src/data_preprocessing.py

## Modul za evaluaciju (Pomoćni modul):
'#' src/model_evaluation.py 
* Sadrži funkcije za izračunavanje metrika (MAE, RMSE, R²) i ne pokreće se samostalno, već ga uvoze skripte za poredak i treniranje.

## Poređenje modela:
python src/model_comparison.py

## Treniranje finalnog modela:
python src/model_training.py
