import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.gridspec as gridspec

#inladen en naar long
pad_train = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"

train_data = pd.read_excel(pad_train)
test_data = pd.read_excel(pad_test)

#long voor vulnerbaility variable
def naar_lange_vorm(df):
    df_lang = pd.wide_to_long(
        df, stubnames=['vulner'], i=['ISO3', 'Name'], j='Jaar', sep='_', suffix=r'\d{4}'
    ).reset_index()
    df_lang['Jaar'] = pd.to_datetime(df_lang['Jaar'], format='%Y')
    return df_lang

train_lang = naar_lange_vorm(train_data)
test_lang = naar_lange_vorm(test_data)

#arma p,q tm 2.2
model_ordes = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]
model_namen = ['AR(1)', 'MA(1)', 'ARMA(1,1)', 'ARMA(2,1)', 'ARMA(1,2)', 'ARMA(2,2)']
landen = test_data['ISO3'].unique()
resultaten = []

figuur = plt.figure(figsize=(16, 12))
layout = gridspec.GridSpec(4, 2)

#loop all countries
for idx, land in enumerate(landen):
    train_land = train_lang[train_lang['ISO3'] == land].copy()
    test_land = test_lang[test_lang['ISO3'] == land].copy()

    y_train = train_land.groupby('Jaar')['vulner'].mean().sort_index()
    y_test = test_land.groupby('Jaar')['vulner'].mean().sort_index()

    beste_mae = np.inf
    beste_forecast = None
    beste_model = ''
    beste_rmse = None
    beste_aic = None
    beste_bic = None

    for orde, naam in zip(model_ordes, model_namen):
        try:
            model = ARIMA(y_train, order=orde)
            fit = model.fit()
            voorspelling = fit.forecast(steps=len(y_test))
            voorspelling.index = y_test.index

            rmse = np.sqrt(mean_squared_error(y_test, voorspelling))
            mae = mean_absolute_error(y_test, voorspelling)

            resultaten.append({
                'ISO3': land,
                'Model': naam,
                'AIC': fit.aic,
                'BIC': fit.bic,
                'RMSE': rmse,
                'MAE': mae
            })

            if mae < beste_mae:
                beste_mae = mae
                beste_rmse = rmse
                beste_forecast = voorspelling
                beste_model = naam
                beste_aic = fit.aic
                beste_bic = fit.bic

        except Exception as fout:
            resultaten.append({
                'ISO3': land,
                'Model': naam,
                'AIC': np.nan,
                'BIC': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'Foutmelding': str(fout)
            })

    #plotten en checken 
    asje = figuur.add_subplot(layout[idx])
    asje.plot(y_test.index, y_test.values, label='Werkelijk', color='black')
    asje.plot(beste_forecast.index, beste_forecast.values, label='Voorspelling', linestyle='--', color='red')
    asje.set_title(f'{land} | {beste_model} | RMSE={beste_rmse:.3f} MAE={beste_mae:.3f}')
    asje.set_xlabel('Jaar')
    asje.set_ylabel('Kwetsbaarheid')
    asje.legend()

plt.suptitle('Heterogene ARMA-modellen per land', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#latex table
resultaten_df = pd.DataFrame(resultaten)

latex = resultaten_df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Vergelijking van ARMA(p,q) modellen per land",
    label="tab:arma_heterogeen"
)

print("\nLaTeX-tabel:\n")
print(latex)
