import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL
from sklearn.metrics import mean_squared_error, mean_absolute_error


pad_train = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"
pad_voorspelling = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\Section 5\ARDL\Xit+1 forecasts\xit_forecast_pooled_2021_2025.xlsx"

train_data = pd.read_excel(pad_train)
test_data = pd.read_excel(pad_test)
voorspelling_data = pd.read_excel(pad_voorspelling)

variabelen = ['vulner', 'food', 'water', 'health', 'eco', 'infra', 'habi']
exogene_vars = ['food', 'water', 'health', 'eco', 'infra', 'habi']

#long maken
def maak_lange_vorm(df):
    df_lang = pd.wide_to_long(df, stubnames=variabelen, i=['ISO3', 'Name'], j='Jaar', sep='_', suffix=r'\d{4}').reset_index()
    df_lang['Jaar'] = pd.to_datetime(df_lang['Jaar'].astype(str) + '-12-31')
    return df_lang

train_lang = maak_lange_vorm(train_data)
test_lang = maak_lange_vorm(test_data)
voorspelling_data['Jaar'] = pd.to_datetime(voorspelling_data['Year'].astype(str) + '-12-31')

#alle orders
model_ordes = [(1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (2,2)]
landen = train_lang['ISO3'].unique()
resultaten = []

#loop voor landen en alle orders
for p, q in model_ordes:
    fig, assen = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    assen_flat = assen.flatten()
    fig.suptitle(f'Heterogeen ARDL model — ARDL({p},{q})', fontsize=16)

    for idx, land in enumerate(landen):
        asje = assen_flat[idx]
        land_train = train_lang[train_lang['ISO3'] == land].set_index('Jaar').sort_index()
        y_train = land_train['vulner']
        X_train = land_train[exogene_vars]

        try:
            model = ARDL(endog=y_train, lags=p, exog=X_train, order=q)
            resultaat = model.fit()
            params = resultaat.params
            const = params.get('const', 0.0)

            phi_1 = params.get('L1.vulner', params.get('vulner.L1', 0.0)) if p > 0 else 0.0
            phi_2 = params.get('L2.vulner', params.get('vulner.L2', 0.0)) if p > 1 else 0.0

            betas_0, betas_1, betas_2 = [], [], []
            for var in exogene_vars:
                betas_0.append(params.get(f'{var}.L0', 0.0))
                betas_1.append(params.get(f'{var}.L1', 0.0) if q > 0 else 0.0)
                betas_2.append(params.get(f'{var}.L2', 0.0) if q > 1 else 0.0)

            y_2020 = y_train.loc['2020-12-31']
            y_2019 = y_train.loc['2019-12-31'] if p > 1 else 0.0

            x_2020 = land_train.loc['2020-12-31', exogene_vars].values
            x_2019 = land_train.loc['2019-12-31', exogene_vars].values if q > 1 else np.zeros(len(exogene_vars))
            x_2021 = voorspelling_data[(voorspelling_data['ISO3'] == land) & (voorspelling_data['Jaar'] == '2021-12-31')][exogene_vars].values.flatten()
            x_2022 = voorspelling_data[(voorspelling_data['ISO3'] == land) & (voorspelling_data['Jaar'] == '2022-12-31')][exogene_vars].values.flatten()

            if any(len(x) == 0 for x in [x_2020, x_2021, x_2022]):
                raise ValueError("Voorspellingsdata ontbreekt")

            y_hat_2021 = const + phi_1 * y_2020 + phi_2 * y_2019 + np.dot(betas_0, x_2021) + np.dot(betas_1, x_2020) + np.dot(betas_2, x_2019)
            y_hat_2022 = const + phi_1 * y_hat_2021 + phi_2 * y_2020 + np.dot(betas_0, x_2022) + np.dot(betas_1, x_2021) + np.dot(betas_2, x_2020)

            y_voorspeld = pd.Series([y_hat_2021, y_hat_2022], index=pd.to_datetime(['2021-12-31', '2022-12-31']))
            y_werkelijk = test_lang[test_lang['ISO3'] == land].set_index('Jaar')['vulner'].loc['2021-12-31':'2022-12-31']

            if len(y_werkelijk) != 2:
                raise ValueError("y_test bevat geen twee observaties")

            rmse = np.sqrt(mean_squared_error(y_werkelijk, y_voorspeld))
            mae = mean_absolute_error(y_werkelijk, y_voorspeld)

            resultaten.append({
                'ISO3': land,
                'Model': f'ARDL({p},{q})',
                'AIC': resultaat.aic,
                'BIC': resultaat.bic,
                'RMSE': rmse,
                'MAE': mae
            })

            asje.plot(y_werkelijk.index, y_werkelijk.values, label='Werkelijk', marker='o', color='black')
            asje.plot(y_voorspeld.index, y_voorspeld.values, label='Voorspelling', linestyle='--', marker='x', color='red')
            asje.set_title(f'{land} | RMSE={rmse:.3f}, MAE={mae:.3f}')
            asje.set_ylabel('Kwetsbaarheid')
            if idx == 0:
                asje.legend()

        except Exception as fout:
            asje.set_title(f"{land} (Fout)")
            asje.text(0.5, 0.5, str(fout), ha='center', va='center')
            asje.axis('off')

    for j in range(len(landen), len(assen_flat)):
        assen_flat[j].axis('off')

    plt.show()

#resultaten
resultaten_df = pd.DataFrame(resultaten)
print(resultaten_df)

latex = resultaten_df.to_latex(
    index=False,
    float_format="%.3f",
    caption="ARDL per land met handmatige forecast: in-sample AIC/BIC en out-of-sample RMSE/MAE (2021–2022)",
    label="tab:ardl_hetero_handmatig",
    position='H'
).replace(r"\\begin{table}", r"\\begin{table}[H]")

print("\nLaTeX-tabel:\n")
print(latex)
