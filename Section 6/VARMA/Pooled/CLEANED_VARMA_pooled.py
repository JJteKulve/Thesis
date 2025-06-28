import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.gridspec as gridspec


pad_train = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"

train_data = pd.read_excel(pad_train)
test_data = pd.read_excel(pad_test)

#pooled means per jaar
variabelen = ['eco', 'food', 'water', 'health', 'infra', 'habi']

def bereken_paneelgemiddelde(df):
    df_lang = pd.wide_to_long(
        df, stubnames=variabelen, i=['ISO3', 'Name'], j='Jaar', sep='_', suffix=r'\d{4}'
    ).reset_index()
    gemiddeldes = df_lang.groupby('Jaar')[variabelen].mean().sort_index()
    gemiddeldes.index = pd.to_datetime(gemiddeldes.index, format="%Y")
    return gemiddeldes, df_lang

paneel_train, df_train_lang = bereken_paneelgemiddelde(train_data)
paneel_test, df_test_lang = bereken_paneelgemiddelde(test_data)

#varma tm 1.1
model_orden = [(1, 0), (0, 1), (1, 1)]
resultaten = []

for orde in model_orden:
    model_naam = f'VARMA{orde}'
    try:
        #model schatten voor pooled
        model = VARMAX(paneel_train, order=orde, enforce_stationarity=True)
        fit = model.fit(disp=False)

        #forecasts voor test periode
        stappen = len(paneel_test)
        voorspelling = fit.get_forecast(steps=stappen).predicted_mean
        voorspelling.index = paneel_test.index

        landen = test_data['ISO3'].unique()
        fouten_per_land = []
        figuur = plt.figure(figsize=(16, 10))
        layout = gridspec.GridSpec(4, 2)

        for idx, land in enumerate(landen):
            land_df = df_test_lang[df_test_lang['ISO3'] == land].copy()
            land_df = land_df.sort_values('Jaar')
            land_df['Jaar'] = pd.to_datetime(land_df['Jaar'], format="%Y")
            land_df = land_df.set_index('Jaar')

            werkelijk = land_df[variabelen].mean(axis=1)
            voorspeld = voorspelling[variabelen].mean(axis=1)

            werkelijk_21_22 = werkelijk.loc["2021":"2022"]
            voorspeld_21_22 = voorspeld.loc["2021":"2022"]

            rmse = np.sqrt(mean_squared_error(werkelijk_21_22, voorspeld_21_22))
            mae = mean_absolute_error(werkelijk_21_22, voorspeld_21_22)
            fouten_per_land.append({'ISO3': land, 'RMSE': rmse, 'MAE': mae})

            #plotten
            asje = figuur.add_subplot(layout[idx])
            asje.plot(werkelijk_21_22.index, werkelijk_21_22.values, label='Werkelijk', marker='o')
            asje.plot(voorspeld_21_22.index, voorspeld_21_22.values, label='Voorspelling', marker='x')
            asje.set_title(f'{land} | RMSE={rmse:.3f} MAE={mae:.3f}')
            asje.set_xlabel('Jaar')
            asje.set_ylabel('Kwetsbaarheid')
            asje.legend()

        plt.suptitle(f'Voorspellingen versus Werkelijkheid per land (Model: {model_naam})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

#fout berekenen
        gemiddelde_rmse = np.mean([x['RMSE'] for x in fouten_per_land])
        gemiddelde_mae = np.mean([x['MAE'] for x in fouten_per_land])

        resultaten.append({
            'Model': model_naam,
            'AIC': fit.aic,
            'BIC': fit.bic,
            'Gem. RMSE': gemiddelde_rmse,
            'Gem. MAE': gemiddelde_mae
        })

#checken of er geenf outen in zitten
    except Exception as fout:
        resultaten.append({
            'Model': model_naam,
            'AIC': np.nan,
            'BIC': np.nan,
            'Gem. RMSE': np.nan,
            'Gem. MAE': np.nan,
            'Fout': str(fout)
        })

#resultaten en latex
resultaten_df = pd.DataFrame(resultaten)

print("\nResultaten per model:\n")
print(resultaten_df)

resultaten_df.to_excel("VARMA_model_resultaten.xlsx", index=False)

latex = resultaten_df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Vergelijking van gepoolde VARMA-modellen: AIC, BIC en out-of-sample fouten",
    label="tab:varma_pooled_models"
)

print("\nLaTeX-tabel:\n")
print(latex)
