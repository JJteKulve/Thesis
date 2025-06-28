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

#lagss
df_train_lang = df_train_lang.sort_values(['ISO3', 'Jaar'])
# lag-1 van kwetsbaarheid
df_train_lang['kwetsbaarheid_lag1'] = df_train_lang.groupby('ISO3')['kwetsbaarheid'].shift(1)
# residu-1: verschil werkelijke verandering minus vorige lag
df_train_lang['residu_lag1'] = (
    df_train_lang.groupby('ISO3')['kwetsbaarheid'].diff()
    - df_train_lang['kwetsbaarheid_lag1'].diff()
)

df_train_lang = df_train_lang.dropna().copy()

#arma1.1
with pm.Model() as model:
    mu_phi    = pm.Normal('mu_phi', mu=0.85, sigma=0.1)
    sigma_phi = pm.Exponential('sigma_phi', 5.0)
    phi       = pm.Normal('phi', mu=mu_phi, sigma=sigma_phi, shape=aantal_landen)

    mu_theta    = pm.Normal('mu_theta', mu=0.0, sigma=0.2)
    sigma_theta = pm.Exponential('sigma_theta', 5.0)
    theta       = pm.Normal('theta', mu=mu_theta, sigma=sigma_theta, shape=aantal_landen)

    sigma = pm.Exponential('sigma', lam=1/0.03)

    dx = df_train_lang['land_idx'].values

    mu = (
        phi[dx] * df_train_lang['kwetsbaarheid_lag1'].values +
        theta[dx] * df_train_lang['residu_lag1'].values
    )

    y_obs = pm.Normal(
        'y_obs',
        mu=mu,
        sigma=sigma,
        observed=df_train_lang['kwetsbaarheid'].values
    )

    approx = pm.fit(method='advi', n=15000)
    trace  = approx.sample(1000)

#INSAMPLE
phi_gemiddeld   = trace.posterior['phi'].mean(('chain','draw')).values
theta_gemiddeld = trace.posterior['theta'].mean(('chain','draw')).values

dx_all = df_train_lang['land_idx'].values
y_pred_in = (
    phi_gemiddeld[dx_all] * df_train_lang['kwetsbaarheid_lag1'].values +
    theta_gemiddeld[dx_all] * df_train_lang['residu_lag1'].values
)
y_true_in = df_train_lang['kwetsbaarheid'].values


i_mse = mean_squared_error(y_true_in, y_pred_in)
i_mae = mean_absolute_error(y_true_in, y_pred_in)
n = len(y_true_in)
k = 2  

aic = n * np.log(i_mse) + 2 * k
bic = n * np.log(i_mse) + np.log(n) * k

# FORECASTING VOOR 2 JAAR
forecast_df = df_test_lang.sort_values(['ISO3','Jaar']).copy()

laatste_kwets = (
    df_train_lang.sort_values(['ISO3','Jaar'])
    .groupby('ISO3')['kwetsbaarheid']
    .last()
    .copy()
)
laatste_residu = (
    df_train_lang.sort_values(['ISO3','Jaar'])
    .groupby('ISO3')
    .apply(lambda x: x['kwetsbaarheid'].iloc[-1]
          - (phi_gemiddeld[x['land_idx'].iloc[-1]] * x['kwetsbaarheid_lag1'].iloc[-1]
             + theta_gemiddeld[x['land_idx'].iloc[-1]] * x['residu_lag1'].iloc[-1]))
)

records = []
idx_map = dict(zip(land_codes, range(aantal_landen)))
for jaar in [2021, 2022]:
    for iso3 in land_codes:
        idx_i = idx_map[iso3]
        φ_i    = phi_gemiddeld[idx_i]
        θ_i    = theta_gemiddeld[idx_i]
        ε_i    = laatste_residu[iso3]
        y_prev = laatste_kwets[iso3]

        y_hat = φ_i * y_prev + θ_i * ε_i
        y_true = forecast_df.loc[
            (forecast_df['ISO3']==iso3)&(forecast_df['Jaar']==jaar),'kwetsbaarheid'
        ].values[0]

        records.append({
            'ISO3': iso3,
            'Jaar': jaar,
            'voorspelling': y_hat,
            'werkelijk': y_true
        })

        laatste_residu[iso3] = y_true - y_hat
        laatste_kwets[iso3]  = y_hat

forecast_result = pd.DataFrame(records)

# OOS
o_mae  = mean_absolute_error(forecast_result['werkelijk'], forecast_result['voorspelling'])
o_rmse = np.sqrt(mean_squared_error(forecast_result['werkelijk'], forecast_result['voorspelling']))

# PLOTS
fig, axes = plt.subplots(4,2,figsize=(14,10),sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i>=len(land_codes): ax.axis('off'); continue
    iso3 = land_codes[i]
    data = forecast_result[forecast_result['ISO3']==iso3]
    ax.plot(data['Jaar'], data['werkelijk'], label='Waargenomen', color='black')
    ax.plot(data['Jaar'], data['voorspelling'], '--', label='Voorspelling', color='red')
    ax.set_title(iso3)
    ax.legend()
plt.tight_layout()
plt.show()

#LATEX
latex = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\
\\midrule
Bayesiaans ARMA(1,1) & {aic:.2f} & {bic:.2f} & {o_mae:.4f} & {o_rmse:.4f} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestatie: ARMA(1,1) met Bayesiaanse schatting, in-sample fit (AIC/BIC) en out-of-sample forecastfouten voor 2021--2022.}}
\\end{{table}}
"""
print(latex)