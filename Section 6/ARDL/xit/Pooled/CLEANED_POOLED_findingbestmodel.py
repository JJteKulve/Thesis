import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

#files
pad_train = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test  = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"
pad_output = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\Section 5\ARDL\Xit+1 forecasts\arma_xit_forecasts_pooled_only.xlsx"

#data
data_train = pd.read_excel(pad_train)
data_test  = pd.read_excel(pad_test)

#variables
verkort_naam = ['eco', 'food', 'water', 'health', 'infra', 'habi']
volledige_naam = {
    'eco':   'Ecosystem',
    'food':  'Food',
    'water': 'Water',
    'health':'Health',
    'infra': 'Infrastructure',
    'habi':  'Habitat'
}

#longg
def maak_lange_tabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zet brede dataset om naar lange vorm:
    kolommen: ISO3, Name, Year, en iedere variabele uit verkort_naam
    """
    lange = pd.wide_to_long(
        df,
        stubnames=verkort_naam,
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    lange['Year'] = pd.to_datetime(lange['Year'], format='%Y')
    return lange

lange_train = maak_lange_tabel(data_train)
lange_test  = maak_lange_tabel(data_test)

#pooled en forecast maken
resultaten = []

def evalueer_pooled_arma(variable: str):
    series_train = lange_train.groupby('Year')[variable].mean().sort_index()
    series_test  = lange_test.groupby('Year')[variable].mean().sort_index().iloc[:2].values

    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue  
            naam_model = f"ARMA({p},{q})"
            try:
                model = ARIMA(series_train, order=(p, 0, q)).fit()
                voorspelling = model.forecast(steps=2)

                resultaten.append({
                    'Variable':        variable,
                    'Variable_full':   volledige_naam[variable],
                    'Model':           naam_model,
                    'Type':            'Pooled',
                    'MAE':             mean_absolute_error(series_test, voorspelling),
                    'RMSE':            np.sqrt(mean_squared_error(series_test, voorspelling)),
                    'AIC':             model.aic,
                    'BIC':             model.bic,
                    'Model_index':     f"{variable}_P_{p}{q}"    
                })
            except Exception as foutmelding:
                print(f"[Pooled] Fout bij {variable} {naam_model}: {foutmelding}")

for var in verkort_naam:
    print(f"Start modelselectie voor: {var.upper()}")
    evalueer_pooled_arma(var)

#resultaten overzichtelijk weergeven met beste model bovenaan 
df_resultaten = pd.DataFrame(resultaten)
df_resultaten = df_resultaten.sort_values(by=['Variable_full', 'RMSE'])

#excel voor controle
df_resultaten.to_excel(pad_output, index=False)
print(f"Resultaten opgeslagen naar: {pad_output}")

#latex overleaf
def maak_latex_tabel(df: pd.DataFrame, caption: str, label: str) -> str:
    kolommen = ['Variable_full', 'Model', 'MAE', 'RMSE', 'AIC', 'BIC']
    return df[kolommen].to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        column_format='llrrrr'
    )

latex_pooled = maak_latex_tabel(
    df_resultaten,
    caption="Forecast evaluatie van gepoolde ARMA-modellen voor verklarende variabelen.",
    label="tab:arma_pooled"
)

print("\nLaTeX-tabel POOLED MODELLEN:\n", latex_pooled)
