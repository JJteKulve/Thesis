import pandas as pd
import numpy as np
import warnings
from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_squared_error, mean_absolute_error


pad_train    = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_train.xlsx"
pad_test     = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit_test.xlsx"
pad_forecast = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\xit_forecast_heterogeneous_2023_2026.xlsx"

df_train    = pd.read_excel(pad_train)
df_test     = pd.read_excel(pad_test)
df_forecast = pd.read_excel(pad_forecast)


stubnames = ['vulner','food','water','health','eco','infra','habi']
x_vars    = ['food','water','health','eco','infra','habi']


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    df_long = pd.wide_to_long(
        df,
        stubnames=stubnames,
        i=['ISO3','Name'],
        j='Year',
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    df_long['Year'] = pd.to_datetime(df_long['Year'].astype(str) + '-12-31')
    return df_long

#train en test data naar long format
df_long       = to_long(df_train).set_index(['ISO3','Year']).sort_index()
df_test_long  = to_long(df_test).set_index(['ISO3','Year']).sort_index()
df_forecast['Year'] = pd.to_datetime(df_forecast['Year'].astype(str) + '-12-31')
countries     = df_long.index.get_level_values(0).unique()

#ardl modellen
orders     = [(1,0),(0,1),(1,1)]
resultaten = []

for p,q in orders:
    modelnaam = f"ARDL({p},{q})"

    df_model = df_long.reset_index().sort_values(['ISO3','Year']).copy()

    # lag
    if p>0:
        df_model['vulner_l1'] = df_model.groupby('ISO3')['vulner'].shift(1)

    # lags exo variables
    for var in x_vars:
        kolom = f"{var}_l1" if q>0 else f"{var}_l0"
        df_model[kolom] = df_model.groupby('ISO3')[var].shift(1 if q>0 else 0)

    df_model = df_model.dropna().set_index(['ISO3','Year']).sort_index()

    #formule
    exog = []
    if p>0:
        exog.append('vulner_l1')
    exog += [f"{var}_l1" if q>0 else f"{var}_l0" for var in x_vars]

    missend = [v for v in exog if v not in df_model.columns]
    if missend:
        print(f"  Overslaan {modelnaam}, ontbreekt: {missend}")
        continue

    #fixed effects met panel ols
    formule = "vulner ~ 1 + " + " + ".join(exog) + " + EntityEffects"
    try:
        mod = PanelOLS.from_formula(formule, data=df_model)
        res = mod.fit()
    except Exception as err:
        print(f"  Fout schatten {modelnaam}: {err}")
        resultaten.append({'Model':modelnaam,'AIC':np.nan,'BIC':np.nan,'Mean MAE':np.nan,'Mean RMSE':np.nan})
        continue

 #resultaten
    beta_full  = res.params
    n_obs      = res.nobs
    k_params   = len(beta_full)
    mse        = (res.resids**2).mean()
    aic        = n_obs * np.log(mse) + 2 * k_params
    bic        = n_obs * np.log(mse) + k_params * np.log(n_obs)

    #SPLIT THE PANEL BIJ P>0 ALLEEN
    if p > 0:
        jaren   = sorted(df_model.index.get_level_values('Year').unique())
        mid     = len(jaren) // 2
        jaren1  = jaren[:mid]
        jaren2  = jaren[mid:]
        deel1   = df_model[df_model.index.get_level_values('Year').isin(jaren1)]
        deel2   = df_model[df_model.index.get_level_values('Year').isin(jaren2)]
        sub     = []
        for deel in (deel1, deel2):
            try:
                r = PanelOLS.from_formula(formule, data=deel).fit()
                sub.append(r.params)
            except:
                pass
        if len(sub) == 2:
            theta_bar     = (sub[0] + sub[1]) / 2
            beta_corrected = 2 * beta_full - theta_bar
        else:
            beta_corrected = beta_full
    else:
        beta_corrected = beta_full

#forecasting
    rmse_lijst, mae_lijst = [],[]
    for iso in countries:
        try:
            y2020 = df_long.loc[(iso,pd.Timestamp('2020-12-31')), 'vulner']
            x2021 = df_forecast.loc[(df_forecast['ISO3']==iso)&(df_forecast['Year']==pd.Timestamp('2021-12-31')), x_vars].values.flatten()
            x2022 = df_forecast.loc[(df_forecast['ISO3']==iso)&(df_forecast['Year']==pd.Timestamp('2022-12-31')), x_vars].values.flatten()

            phi    = beta_corrected.get('vulner_l1',0)
            beta_x = np.array([beta_corrected.get(col,0) for col in exog if col!='vulner_l1'])

            # pred
            if p>0:
                y1 = phi*y2020 + beta_x.dot(x2021)
                y2 = phi*y1   + beta_x.dot(x2022)
            else:
                y1 = beta_x.dot(x2021)
                y2 = beta_x.dot(x2022)

            idx    = [pd.Timestamp('2021-12-31'), pd.Timestamp('2022-12-31')]
            y_act  = df_test_long.loc[(iso, idx), 'vulner']
            y_pred = pd.Series([y1, y2], index=idx)

            rmse_lijst.append(np.sqrt(mean_squared_error(y_act, y_pred)))
            mae_lijst.append(mean_absolute_error(y_act, y_pred))
        except Exception as err:
            print(f"  Fout forecast {iso} [{modelnaam}]: {err}")

    resultaten.append({
        'Model':    modelnaam,
        'AIC':      aic,
        'BIC':      bic,
        'Mean MAE': np.nanmean(mae_lijst),
        'Mean RMSE':np.nanmean(rmse_lijst)
    })

#output
df_res = pd.DataFrame(resultaten)
print(df_res[['Model','AIC','BIC','Mean MAE','Mean RMSE']])


latex = df_res[['Model','AIC','BIC','Mean MAE','Mean RMSE']]\
    .to_latex(index=False, float_format='%.3f',
              caption='ARDL-modellen met fixed effects en split-panel jackknife biascorrectie (alleen bij p > 0)',
              label='tab:splitjackknife_ardl_fe')\
    .replace(r'\\begin\{table\}',r'\\begin{table}[H]')
print("\nLaTeX tabel:\n", latex)
