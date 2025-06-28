import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


#besttenadne
pad_train  = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test   = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"
pad_output = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\Section 5\ARDL\Xit+1 forecasts\arma_hetero_top3_per_land.xlsx"

train_df = pd.read_excel(pad_train)
test_df  = pd.read_excel(pad_test)

verkorte_variabelen = ['eco', 'food', 'water', 'health', 'infra', 'habi']
volledige_namen = {
    'eco':   'Ecosystem',
    'food':  'Food',
    'water': 'Water',
    'health':'Health',
    'infra': 'Infrastructure',
    'habi':  'Habitat'
}

def maak_lange_tabel(df: pd.DataFrame) -> pd.DataFrame:
    lange = pd.wide_to_long(
        df,
        stubnames=verkorte_variabelen,
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    lange['Year'] = pd.to_datetime(lange['Year'], format='%Y')
    return lange

lange_train = maak_lange_tabel(train_df)
lange_test  = maak_lange_tabel(test_df)

#heteroarma per land
resultaten = []
landen = lange_train['ISO3'].unique()

for var in verkorte_variabelen:
    for land in landen:
        #training en test
        serie_train = (lange_train
            .loc[lange_train['ISO3'] == land]
            .sort_values('Year')[var]
            .dropna()
        )
        serie_test = (lange_test
            .loc[lange_test['ISO3'] == land]
            .sort_values('Year')[var]
            .dropna()
            .values[:2] 
        )
        for p in range(3):
            for q in range(3):
                if p == 0 and q == 0:
                    continue  
                model_naam = f"ARMA({p},{q})"
                try:
                    model = ARIMA(serie_train, order=(p, 0, q)).fit()
                    voorspelling = model.forecast(steps=2)
                    # Controleer lengte
                    if len(serie_test) == len(voorspelling):
                        mae  = mean_absolute_error(serie_test, voorspelling)
                        rmse = np.sqrt(mean_squared_error(serie_test, voorspelling))
                        resultaten.append({
                            'Land':             land,
                            'Variabele':        var,
                            'Variabele_volledig': volledige_namen[var],
                            'Model':            model_naam,
                            'MAE':              mae,
                            'RMSE':             rmse,
                            'AIC':              model.aic,
                            'BIC':              model.bic
                        })
                except Exception as e:
                    print(f"[Fout] {land} - {var} - {model_naam}: {e}")

#alleen top drie landen
df_res = pd.DataFrame(resultaten)
df_top3 = (
    df_res
    .sort_values(['Land', 'Variabele_volledig', 'RMSE'])
    .groupby(['Land', 'Variabele_volledig'])
    .head(3)
    .reset_index(drop=True)
)

#excel
df_top3.to_excel(pad_output, index=False)
print(f"\nTop 3 ARMA-modellen per land en variabele opgeslagen in: {pad_output}")
