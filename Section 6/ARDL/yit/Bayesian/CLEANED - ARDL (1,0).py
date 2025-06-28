import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"
pad_voorspelling = r"/Users/macbookair/Desktop/xit_forecast_pooled_2021_2025.xlsx"

data_train = pd.read_excel(pad_train)
data_test = pd.read_excel(pad_test)
data_voorspelling = pd.read_excel(pad_voorspelling)

#lang formaat
variabelen_kort = ['food', 'water', 'health', 'eco', 'infra', 'habi']
def naar_lang_formaat(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen_kort + ['vulner'],
        i=['ISO3', 'Name'],
        j='Jaar',
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    df_lang.rename(columns={'vulner': 'kwetsbaarheid'}, inplace=True)
    df_lang['Jaar'] = pd.to_datetime(df_lang['Jaar'].astype(str) + '-12-31')
    return df_lang

train_lang = naar_lang_formaat(data_train)
test_lang = naar_lang_formaat(data_test)


data_voorspelling.rename(columns={'Year': 'Jaar'}, inplace=True)
data_voorspelling['Jaar'] = pd.to_datetime(data_voorspelling['Jaar'].astype(str) + '-12-31')


train_lang['land_index'] = train_lang['ISO3'].astype('category').cat.codes
data_voorspelling['land_index'] = data_voorspelling['ISO3'].astype('category').cat.codes
lijst_landen = train_lang['ISO3'].astype('category').cat.categories
n_landen = train_lang['land_index'].nunique()

# lag
train_lang['kwetsbaarheid_lag1'] = train_lang.groupby('ISO3')['kwetsbaarheid'].shift(1)
train_lang = train_lang.dropna().copy()

# ardl1.0 check
with pm.Model() as model:
    mu_phi = pm.Normal('mu_phi', mu=0.8, sigma=0.2)
    sigma_phi = pm.Exponential('sigma_phi', 2.0)
    phi_per_land = pm.Normal('phi_per_land', mu=mu_phi, sigma=sigma_phi, shape=n_landen)

    mu_beta = pm.Normal('mu_beta', mu=0.7, sigma=0.3)
    sigma_beta = pm.Exponential('sigma_beta', 2.0)
    beta_per_variabele = {
        v: pm.Normal(f'beta_{v}', mu=mu_beta, sigma=sigma_beta, shape=n_landen)
        for v in variabelen_kort
    }

    sigma = pm.Exponential('sigma', lam=1/0.03)

    idx = train_lang['land_index'].values
    mu = (
        phi_per_land[idx] * train_lang['kwetsbaarheid_lag1'].values
        + sum(beta_per_variabele[v][idx] * train_lang[v].values for v in variabelen_kort)
    )

    pm.Normal('observaties', mu=mu, sigma=sigma, observed=train_lang['kwetsbaarheid'].values)

    approx = pm.fit(method="advi", n=20000)
    trace = approx.sample(1000)

#in sample done
phi_mean = trace.posterior['phi_per_land'].mean(('chain','draw')).values
beta_mean = {v: trace.posterior[f'beta_{v}'].mean(('chain','draw')).values for v in variabelen_kort}

idx = train_lang['land_index'].values
X_train = np.stack([train_lang[v].values for v in variabelen_kort], axis=1)
beta_mat = np.stack([beta_mean[v][idx] for v in variabelen_kort], axis=1)

pred_in = phi_mean[idx] * train_lang['kwetsbaarheid_lag1'].values + (beta_mat * X_train).sum(axis=1)
true_in = train_lang['kwetsbaarheid'].values

mse_in = mean_squared_error(true_in, pred_in)
mae_in = mean_absolute_error(true_in, pred_in)
n = len(true_in)
k = 1 + len(variabelen_kort)
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k

# voorspellen
laatste = train_lang.sort_values('Jaar').groupby('ISO3')['kwetsbaarheid'].last().copy()
voorsp_records = []
for jaar in [2021, 2022]:
    df_temp = data_voorspelling[data_voorspelling['Jaar'].dt.year == jaar].copy()
    df_temp['kwetsbaarheid_lag1'] = df_temp['ISO3'].map(laatste)

    idx = df_temp['land_index'].values
    X_now = np.stack([df_temp[v].values for v in variabelen_kort], axis=1)
    beta_vals = np.stack([beta_mean[v][idx] for v in variabelen_kort], axis=1)

    y_hat = phi_mean[idx] * df_temp['kwetsbaarheid_lag1'].values + (beta_vals * X_now).sum(axis=1)
    for i, iso in enumerate(df_temp['ISO3']):
        laatste[iso] = y_hat[i]
    df_temp['kwetsbaarheid_voorspelling'] = y_hat
    true_vals = test_lang[test_lang['Jaar'].dt.year == jaar].set_index('ISO3')['kwetsbaarheid']
    df_temp['kwetsbaarheid_echt'] = df_temp['ISO3'].map(true_vals)
    voorsp_records.append(df_temp)

forecast_result = pd.concat(voorsp_records).sort_values(['ISO3','Jaar']).reset_index(drop=True)

#outofsample
mae_out = mean_absolute_error(forecast_result['kwetsbaarheid_echt'], forecast_result['kwetsbaarheid_voorspelling'])
rmse_out = np.sqrt(mean_squared_error(forecast_result['kwetsbaarheid_echt'], forecast_result['kwetsbaarheid_voorspelling']))

#resultaten
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i >= len(lijst_landen): ax.axis('off'); continue
    iso = lijst_landen[i]
    data = forecast_result[forecast_result['ISO3'] == iso]
    ax.plot(data['Jaar'].dt.year, data['kwetsbaarheid_echt'], label='Echt')
    ax.plot(data['Jaar'].dt.year, data['kwetsbaarheid_voorspelling'], '--', label='Voorspelling')
    ax.set_title(f'{iso}')
    ax.set_xticks([2021, 2022])
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.show()

#latex
latex = f'''
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\
\\midrule
Bayesiaanse ARDL(1,0) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{ARDL(1,0): In-sample fit en out-of-sample forecast fouten (2021--2022).}}
\\end{{table}}
'''
print(latex)
