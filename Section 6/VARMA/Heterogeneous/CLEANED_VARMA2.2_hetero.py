import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

#laden
pad_train = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"

train_data = pd.read_excel(pad_train)
test_data = pd.read_excel(pad_test)

#long van alle x variables
variabelen = ['eco', 'food', 'water', 'health', 'infra', 'habi']

def naar_lange_vorm(df):
    df_lang = pd.wide_to_long(df, stubnames=variabelen, i=['ISO3', 'Name'], j='Jaar', sep='_', suffix=r'\d{4}').reset_index()
    df_lang['Jaar'] = pd.to_datetime(df_lang['Jaar'], format='%Y')
    return df_lang

train_lang = naar_lange_vorm(train_data)
test_lang = naar_lange_vorm(test_data)

#varma xit
model_ordes = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]
landen = test_data['ISO3'].unique()
resultaten = []

figuur = plt.figure(figsize=(16, 12))
layout = gridspec.GridSpec(4, 2)

#loop over landen en ordes
for idx, land in enumerate(landen):
    train_land = train_lang[train_lang['ISO3'] == land].copy()
    test_land = test_lang[test_lang['ISO3'] == land].copy()

    y_train = train_land.pivot(index='Jaar', columns='ISO3')[variabelen].droplevel(1, axis=1)
    y_test = test_land.pivot(index='Jaar', columns='ISO3')[variabelen].droplevel(1, axis=1)

    beste_mae = np.inf
    beste_voorspelling = None
    beste_model = ''
    beste_rmse = None
    beste_aic = None
    beste_bic = None

    for orde in model_ordes:
        try:
            model = VARMAX(y_train, order=orde, enforce_stationarity=True)
            fit = model.fit(disp=False)
            voorspelling = fit.forecast(steps=len(y_test))
            voorspelling.index = y_test.index

            rmse = np.sqrt(mean_squared_error(y_test.values, voorspelling.values))
            mae = mean_absolute_error(y_test.values, voorspelling.values)

            resultaten.append({
                'ISO3': land,
                'Model': f"VARMA{orde}",
                'AIC': fit.aic,
                'BIC': fit.bic,
                'RMSE': rmse,
                'MAE': mae
            })

            if mae < beste_mae:
                beste_mae = mae
                beste_rmse = rmse
                beste_voorspelling = voorspelling
                beste_model = f"VARMA{orde}"
                beste_aic = fit.aic
                beste_bic = fit.bic

        except Exception as fout:
            resultaten.append({
                'ISO3': land,
                'Model': f"VARMA{orde}",
                'AIC': np.nan,
                'BIC': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'Foutmelding': str(fout)
            })

    #plotten mean en actual forecasts
    asje = figuur.add_subplot(layout[idx])
    asje.plot(y_test.index, y_test.mean(axis=1), label='Werkelijk (gemiddelde)', color='black')
    asje.plot(beste_voorspelling.index, beste_voorspelling.mean(axis=1), label='Voorspelling (gemiddelde)', linestyle='--', color='red')
    asje.set_title(f'{land} | {beste_model}\nRMSE={beste_rmse:.3f} MAE={beste_mae:.3f}')
    asje.set_xlabel('Jaar')
    asje.set_ylabel('Gemiddelde van verklarende variabelen')
    asje.legend()

plt.suptitle('Heterogene VARMA-modellen per land (X_it)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#resultaten en overleaf code
resultaten_df = pd.DataFrame(resultaten)

latex = resultaten_df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Vergelijking van VARMA(p,q) modellen per land",
    label="tab:varma_heterogeen"
)

print("\nLaTeX-tabel:\n")
print(latex)
