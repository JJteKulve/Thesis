import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize

#data
pad_trainbestand = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_testbestand  = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"

data_train = pd.read_excel(pad_trainbestand)
data_test  = pd.read_excel(pad_testbestand)

#naarlong
def extracteer_lange_formaat(dataframe: pd.DataFrame) -> pd.DataFrame:
    lange_df = pd.wide_to_long(
        dataframe,
        stubnames=["vulner"],
        i=["ISO3", "Name"],
        j="Year",
        sep="_",
        suffix=r"\d{4}"
    ).reset_index()
    # Jaar naar datetime
    lange_df['Year'] = pd.to_datetime(lange_df['Year'], format='%Y')
    return lange_df

lange_train = extracteer_lange_formaat(data_train)
lange_test  = extracteer_lange_formaat(data_test)
landcodes   = lange_test['ISO3'].unique()

#def

def neg_logwaarschijnlijkheid_ar1(params, df_lange):
    phi = params[0]
    sigma = 0.1
    if not (-1 < phi < 1):
        return 1e6
    totale_ll = 0.0
    for land in df_lange['ISO3'].unique():
        y = df_lange[df_lange['ISO3'] == land].sort_values('Year')['vulner'].values
        residuen = y[1:] - phi * y[:-1]
        totale_ll += -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + residuen**2 / sigma**2)
    return -totale_ll


def neg_logwaarschijnlijkheid_ma1(params, df_lange):
    theta = params[0]
    sigma = 0.1
    if not (-1 < theta < 1):
        return 1e6
    totale_ll = 0.0
    for land in df_lange['ISO3'].unique():
        y = df_lange[df_lange['ISO3'] == land].sort_values('Year')['vulner'].values
        fouten = np.zeros_like(y)
        for t in range(1, len(y)):
            fouten[t] = y[t] - theta * fouten[t-1]
        totale_ll += -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + fouten**2 / sigma**2)
    return -totale_ll


def neg_logwaarschijnlijkheid_arma11(params, df_lange):
    phi, theta = params
    sigma = 0.1
    if not (-1 < phi < 1) or not (-1 < theta < 1):
        return 1e6
    totale_ll = 0.0
    for land in df_lange['ISO3'].unique():
        y = df_lange[df_lange['ISO3'] == land].sort_values('Year')['vulner'].values
        fouten = np.zeros_like(y)
        for t in range(1, len(y)):
            voorspeld = phi * y[t-1] + theta * fouten[t-1]
            fouten[t] = y[t] - voorspeld
        totale_ll += -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + fouten**2 / sigma**2)
    return -totale_ll

#model fit en forecasten

def pas_model_toe_en_voorspel(model_naam, df_train, df_test):
    y_train = df_train.groupby('Year')['vulner'].mean().sort_index()
    y_test  = df_test.groupby('Year')['vulner'].mean().sort_index()
    geschiedenis = list(y_train.values)
    voorspellingen = []

    if model_naam == 'AR(1)':
        res = minimize(neg_logwaarschijnlijkheid_ar1, x0=[0.5], args=(df_train,), bounds=[(-0.99, 0.99)])
        phi = res.x[0]
        loglik = -res.fun
        for _ in range(len(y_test)):
            volgende = phi * geschiedenis[-1]
            voorspellingen.append(volgende)
            geschiedenis.append(volgende)
        return voorspellingen, loglik, 1

    if model_naam == 'MA(1)':
        res = minimize(neg_logwaarschijnlijkheid_ma1, x0=[0.5], args=(df_train,), bounds=[(-0.99, 0.99)])
        theta = res.x[0]
        loglik = -res.fun
        vorige_fout = 0.0
        for _ in range(len(y_test)):
            volgende = theta * vorige_fout
            voorspellingen.append(volgende)
            geschiedenis.append(volgende)
            vorige_fout = 0.0
        return voorspellingen, loglik, 1

    if model_naam == 'ARMA(1,1)':
        res = minimize(neg_logwaarschijnlijkheid_arma11, x0=[0.5, 0.5], args=(df_train,), bounds=[(-0.99, 0.99)]*2)
        phi, theta = res.x
        loglik = -res.fun
        vorige_fout = 0.0
        for _ in range(len(y_test)):
            volgende = phi * geschiedenis[-1] + theta * vorige_fout
            voorspellingen.append(volgende)
            geschiedenis.append(volgende)
            vorige_fout = 0.0
        return voorspellingen, loglik, 2

#results
resultaten = []
for model in ['AR(1)', 'MA(1)', 'ARMA(1,1)']:
    vals, loglik, k = pas_model_toe_en_voorspel(model, lange_train, lange_test)
    # Verzamel werkelijke waarden per land
    actual_all = []
    for land in landcodes:
        df_land = lange_test[lange_test['ISO3'] == land].set_index('Year').sort_index()
        actual = df_land['vulner'].reindex(lange_test['Year'].unique()).values
        actual_all.extend(actual)
    # Herhaal voorspelling voor elk land
    forecast_repeated = np.tile(vals, len(landcodes))
    # Bereken foutmaten
    rmse = np.sqrt(mean_squared_error(actual_all, forecast_repeated))
    mae  = mean_absolute_error(actual_all, forecast_repeated)
    n    = len(actual_all)
    aic  = 2 * k - 2 * loglik
    bic  = np.log(n) * k - 2 * loglik
    resultaten.append({'Model': model, 'AIC': aic, 'BIC': bic, 'RMSE': rmse, 'MAE': mae})

#latex tabel
resultaten_df = pd.DataFrame(resultaten)
latex_tabel = resultaten_df.to_latex(index=False, float_format="%.3f",
    caption="Vergelijking van AR(1), MA(1) en ARMA(1,1) modellen met iteratieve forecasting en pooled evaluatie",
    label="tab:arma_pooled_iteratief")
print("\nLaTeX Table:\n", latex_tabel)