import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#data
pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test  = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"

df_train = pd.read_excel(pad_train)
df_test  = pd.read_excel(pad_test)

#lang
def naar_lang(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=['vulner'],      
        i=['ISO3', 'Name'],        
        j='Jaar',                
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    df_lang.rename(columns={'vulner': 'kwetsbaarheid'}, inplace=True)
    return df_lang

df_train_lang = naar_lang(df_train)
df_test_lang  = naar_lang(df_test)

df_train_lang['land_idx'] = df_train_lang['ISO3'].astype('category').cat.codes
land_codes = df_train_lang['ISO3'].astype('category').cat.categories
aantal_landen = df_train_lang['land_idx'].nunique()

# lags
df_train_lang = df_train_lang.sort_values(['ISO3', 'Jaar'])
df_train_lang['kwetsbaarheid_lag1'] = df_train_lang.groupby('ISO3')['kwetsbaarheid'].shift(1)

(df_train_lang['residu_lag1']
    ) = df_train_lang.groupby('ISO3')['kwetsbaarheid'].diff() - df_train_lang['kwetsbaarheid_lag1'].diff()

df_train_lang = df_train_lang.dropna().copy()

#ma1
with pm.Model() as model:
    mu_theta    = pm.Normal('mu_theta', mu=0.0, sigma=0.2)
    sigma_theta = pm.Exponential('sigma_theta', 5.0)
    theta       = pm.Normal('theta', mu=mu_theta, sigma=sigma_theta, shape=aantal_landen)

    sigma = pm.Exponential('sigma', lam=1/0.03)


    idx = df_train_lang['land_idx'].values

    mu = theta[idx] * df_train_lang['residu_lag1'].values


    y_obs = pm.Normal(
        'y_obs',
        mu=mu,
        sigma=sigma,
        observed=df_train_lang['kwetsbaarheid'].values
    )

    approx = pm.fit(method="advi", n=15000)
    trace  = approx.sample(1000)

#insample
theta_gemiddeld = trace.posterior['theta'].mean(('chain', 'draw')).values
theta_vec = theta_gemiddeld[df_train_lang['land_idx'].values]


y_pred_in = theta_vec * df_train_lang['residu_lag1'].values
y_true_in = df_train_lang['kwetsbaarheid'].values


mse_in = mean_squared_error(y_true_in, y_pred_in)
mae_in = mean_absolute_error(y_true_in, y_pred_in)
n = len(y_true_in)
k = 1  
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k

# forecasts
forecast_df = df_test_lang.sort_values(['ISO3', 'Jaar']).copy()
laatste = (
    df_train_lang.sort_values(['ISO3', 'Jaar'])
    .groupby('ISO3')
    .tail(1)
    .reset_index(drop=True)
    .copy()
)

forecast_records = []
land_idx_map = dict(zip(land_codes, range(aantal_landen)))

for jaar in [2021, 2022]:
    nieuw_laag = []
    for iso3 in land_codes:
        rij = laatste[laatste['ISO3'] == iso3]
        if len(rij) < 1:
            print(f"Te weinig data voor {iso3} in {jaar}.")
            continue

        idx_i = land_idx_map[iso3]
        theta_i = theta_gemiddeld[idx_i]

        residu_nieuw = rij['kwetsbaarheid'].iloc[-1] - theta_i * rij['residu_lag1'].iloc[-1]


        y_hat = theta_i * residu_nieuw

        y_true = forecast_df.loc[
            (forecast_df['ISO3'] == iso3) &
            (forecast_df['Jaar'] == jaar), 'kwetsbaarheid'
        ].values[0]

        forecast_records.append({
            'ISO3': iso3,
            'Jaar': jaar,
            'voorspelling': y_hat,
            'werkelijk': y_true
        })


        nieuw_laag.append({
            'ISO3': iso3,
            'kwetsbaarheid': y_hat,
            'residu_lag1': rij['kwetsbaarheid'].iloc[-1] - y_hat,
            'Jaar': jaar
        })

    laatste = pd.concat([laatste, pd.DataFrame(nieuw_laag)], ignore_index=True)
    laatste = (
        laatste.sort_values(['ISO3', 'Jaar'])
        .groupby('ISO3')
        .tail(1)
        .reset_index(drop=True)
    )

forecast_result = pd.DataFrame(forecast_records)

#OUTOFSAMPLE
mae_out = mean_absolute_error(
    forecast_result['werkelijk'],
    forecast_result['voorspelling']
)
rmse_out = np.sqrt(mean_squared_error(
    forecast_result['werkelijk'],
    forecast_result['voorspelling']
))

#PLOT EN LATEDX
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i >= len(land_codes):
        ax.axis('off')
        continue
    iso3 = land_codes[i]
    data = forecast_result[forecast_result['ISO3'] == iso3]
    ax.plot(data['Jaar'], data['werkelijk'], label='Waargenomen', color='black')
    ax.plot(data['Jaar'], data['voorspelling'], '--', color='red', label='Voorspelling')
    ax.set_title(f'{iso3}')
    ax.legend()

plt.tight_layout()
plt.show()


latex = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\
\\midrule
Bayesiaans MA(1) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestatie: Bayesiaans MA(1) met in-sample fit (AIC/BIC) en out-of-sample forecastfouten voor 2021--2022.}}
\\end{{table}}
"""
print(latex)
