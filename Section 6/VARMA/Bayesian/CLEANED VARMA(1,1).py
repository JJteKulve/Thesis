import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"  
pad_test  = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"  
variabelen = ['food', 'water', 'health', 'eco', 'infra', 'habi']

def lees_data(pad):  
    return pd.read_excel(pad)

df_train = lees_data(pad_train)
df_test  = lees_data(pad_test)

#lang
def naar_long(df):  
    df_long = pd.wide_to_long(
        df,
        stubnames=variabelen + ['vulner'],
        i=['ISO3','Name'],
        j='Year',
        sep='_',
        suffix='\\d{4}'
    ).reset_index()
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long

train_long = naar_long(df_train)
test_long  = naar_long(df_test)

#landen
train_long['land_idx'] = train_long['ISO3'].astype('category').cat.codes
land_namen = train_long['ISO3'].astype('category').cat.categories
n_landen  = train_long['land_idx'].nunique()

#lags
for v in variabelen:
    train_long[f'{v}_lag1']   = train_long.groupby('ISO3')[v].shift(1)
    train_long[f'{v}_verschil'] = train_long.groupby('ISO3')[v].diff()
# Verwijder NA
train_long = train_long.dropna().reset_index(drop=True)

#varma met priors 1,1
with pm.Model() as model:
    #phi
    mu_phi    = pm.Normal('mu_phi', mu=0.2, sigma=0.3)
    sigma_phi = pm.Exponential('sigma_phi', 2.0)
    #theta
    mu_theta    = pm.Normal('mu_theta', mu=0.0, sigma=0.2)
    sigma_theta = pm.Exponential('sigma_theta', 5.0)
    # observatie error
    sigma_y = pm.Exponential('sigma_y', lam=1/0.03)

    #
    phi   = {v: pm.Normal(f'phi_{v}', mu=mu_phi, sigma=sigma_phi, shape=n_landen)
             for v in variabelen}
    theta = {v: pm.Normal(f'theta_{v}', mu=mu_theta, sigma=sigma_theta, shape=n_landen)
             for v in variabelen}

    idx = train_long['land_idx'].values
    mu  = sum(
        phi[v][idx] * train_long[f'{v}_lag1'].values +
        theta[v][idx] * train_long[f'{v}_verschil'].values
        for v in variabelen
    ) / len(variabelen)

    #obs model
    pm.Normal('y_obs', mu=mu, sigma=sigma_y, observed=train_long['vulner'].values)

    #sim
    approx = pm.fit(method='advi', n=20000)
    trace  = approx.sample(1000)

#insample eva 
x_lag   = np.stack([train_long[f'{v}_lag1'].values for v in variabelen], axis=1)
x_diff  = np.stack([train_long[f'{v}_verschil'].values for v in variabelen], axis=1)

phi_mean = np.stack([
    trace.posterior[f'phi_{v}'].mean(('chain','draw')).values[train_long['land_idx'].values]
    for v in variabelen
], axis=1)
theta_mean = np.stack([
    trace.posterior[f'theta_{v}'].mean(('chain','draw')).values[train_long['land_idx'].values]
    for v in variabelen
], axis=1)

voorsp_in = (phi_mean * x_lag + theta_mean * x_diff).mean(axis=1)
waar_in   = train_long['vulner'].values

mse_in = mean_squared_error(waar_in, voorsp_in)
mae_in = mean_absolute_error(waar_in, voorsp_in)
n = len(waar_in)
k = 2 * len(variabelen)
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k

# forecasting
laatste_X     = train_long.sort_values('Year').groupby('ISO3')[variabelen].last()
vorig_verschil = train_long.sort_values('Year').groupby('ISO3')[variabelen].diff().groupby(train_long['ISO3']).last()

# Posterior 
gelijk_phi   = {v: trace.posterior[f'phi_{v}'].mean(('chain','draw')).values for v in variabelen}
gelijk_theta = {v: trace.posterior[f'theta_{v}'].mean(('chain','draw')).values for v in variabelen}

forecast_lijst = []
for jaar in [2021, 2022]:
    temp = test_long[test_long['Year'] == jaar].copy()
    temp['land_idx'] = temp['ISO3'].astype('category').cat.codes
    for v in variabelen:
        temp[f'{v}_lag1']     = temp['ISO3'].map(laatste_X[v])
        temp[f'{v}_verschil'] = temp['ISO3'].map(vorig_verschil[v])
        # Bereken voorspelde X
        idx = temp['land_idx'].values
        temp[v] = (
            gelijk_phi[v][idx] * temp[f'{v}_lag1'].values +
            gelijk_theta[v][idx] * temp[f'{v}_verschil'].values
        )
    temp['vulner_forecast'] = temp[variabelen].mean(axis=1)
    for v in variabelen:
        vorig_verschil[v] = temp[v].values - temp[f'{v}_lag1'].values
        laatste_X[v]      = temp[v].values
    forecast_lijst.append(temp)

forecast_final = pd.concat(forecast_lijst).sort_values(['ISO3','Year']).reset_index(drop=True)

#OOs
y_true = forecast_final['vulner'].values
y_pred = forecast_final['vulner_forecast'].values
mae_out = mean_absolute_error(y_true, y_pred)
rmse_out = np.sqrt(mean_squared_error(y_true, y_pred))

#plotss
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i >= len(land_namen):
        ax.axis('off')
        continue
    land = land_namen[i]
    data_land = forecast_final[forecast_final['ISO3'] == land]
    ax.plot(data_land['Year'], data_land['vulner'], label='Echt')
    ax.plot(data_land['Year'], data_land['vulner_forecast'], '--', label='Forecast')
    ax.set_title(land)
    ax.legend()
plt.tight_layout()
plt.show()

#latex
latex = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\\
\\midrule
Bayesiaans VARMA(1,1) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestaties: in-sample fit (AIC, BIC) en out-of-sample fouten (2021–2022).}}
\\end{{table}}
"""
print(latex)
