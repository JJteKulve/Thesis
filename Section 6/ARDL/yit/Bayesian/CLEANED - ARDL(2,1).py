import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# data
pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"
pad_voorspelling = r"/Users/macbookair/Desktop/xit_forecast_pooled_2021_2025.xlsx"

gegevens_train = pd.read_excel(pad_train)
gegevens_test = pd.read_excel(pad_test)
gegevens_voorspelling = pd.read_excel(pad_voorspelling)

#omzetten
variabelen_kort = ['food', 'water', 'health', 'eco', 'infra', 'habi']
def naar_lang(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen_kort + ['vulner'],
        i=['ISO3', 'Name'], j='Jaar', sep='_', suffix=r'\d{4}'
    ).reset_index()
    df_lang.rename(columns={'vulner': 'kwetsbaarheid'}, inplace=True)
    df_lang['Jaar'] = pd.to_datetime(df_lang['Jaar'].astype(str) + '-12-31')
    return df_lang

train_lang = naar_lang(gegevens_train)
test_lang = naar_lang(gegevens_test)

data_voorspelling = gegevens_voorspelling.copy()
data_voorspelling.rename(columns={'Year': 'Jaar'}, inplace=True)
data_voorspelling['Jaar'] = pd.to_datetime(data_voorspelling['Jaar'].astype(str) + '-12-31')

train_lang['land_idx'] = train_lang['ISO3'].astype('category').cat.codes
data_voorspelling['land_idx'] = data_voorspelling['ISO3'].map(
    dict(zip(train_lang['ISO3'].astype('category').cat.categories, range(train_lang['ISO3'].nunique())))
)
landen = train_lang['ISO3'].astype('category').cat.categories
n_landen = len(landen)

# lagss
for var in ['kwetsbaarheid'] + variabelen_kort:
    train_lang[f"{var}_lag1"] = train_lang.groupby('ISO3')[var].shift(1)
    train_lang[f"{var}_lag2"] = train_lang.groupby('ISO3')[var].shift(2)
train_lang = train_lang.dropna().copy()

#ardl 2.1
with pm.Model() as ardl21_model:
    mu_phi = pm.Normal('mu_phi', mu=0.8, sigma=0.2)
    sigma_phi = pm.Exponential('sigma_phi', 2.0)
    phi_y1 = pm.Normal('phi_y1', mu=mu_phi, sigma=sigma_phi, shape=n_landen)
    phi_y2 = pm.Normal('phi_y2', mu=mu_phi, sigma=sigma_phi, shape=n_landen)

    mu_beta = pm.Normal('mu_beta', mu=0.7, sigma=0.3)
    sigma_beta = pm.Exponential('sigma_beta', 2.0)
    phi_x_lag = {
        var: pm.Normal(f'phi_{var}_lag', mu=mu_beta, sigma=sigma_beta, shape=n_landen)
        for var in variabelen_kort
    }

    sigma = pm.Exponential('sigma', lam=1/0.03)
    idx = train_lang['land_idx'].values

    mu = (
        phi_y1[idx] * train_lang['kwetsbaarheid_lag1'].values +
        phi_y2[idx] * train_lang['kwetsbaarheid_lag2'].values +
        sum(
            phi_x_lag[var][idx] * train_lang[f'{var}_lag1'].values
            for var in variabelen_kort
        )
    )

    pm.Normal('waarnemingen', mu=mu, sigma=sigma, observed=train_lang['kwetsbaarheid'].values)
    approx = pm.fit(method='advi', n=15000)
    trace = approx.sample(1000)

phi_y1_mean = trace.posterior['phi_y1'].mean(('chain', 'draw')).values
phi_y2_mean = trace.posterior['phi_y2'].mean(('chain', 'draw')).values
phi_x_mean = {var: trace.posterior[f'phi_{var}_lag'].mean(('chain', 'draw')).values for var in variabelen_kort}

#insample
idx = train_lang['land_idx'].values
y_pred_in = (
    phi_y1_mean[idx] * train_lang['kwetsbaarheid_lag1'].values +
    phi_y2_mean[idx] * train_lang['kwetsbaarheid_lag2'].values +
    sum(phi_x_mean[var][idx] * train_lang[f'{var}_lag1'].values for var in variabelen_kort)
)
y_true_in = train_lang['kwetsbaarheid'].values
mse_in = mean_squared_error(y_true_in, y_pred_in)
mae_in = mean_absolute_error(y_true_in, y_pred_in)
n = len(y_true_in)
k = 2 + len(variabelen_kort)
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k
print(f"In-sample AIC: {aic:.2f}, BIC: {bic:.2f}, MAE: {mae_in:.4f}")

#voorspellen
last_vals = (
    train_lang
    .sort_values(['ISO3','Jaar'])
    .groupby('ISO3')
    .tail(2)
    .reset_index(drop=True)
)
forecast_records = []
for jaar in [2021, 2022]:
    temp = data_voorspelling[data_voorspelling['Jaar'].dt.year == jaar].copy()
    temp = temp[temp['ISO3'].isin(landen)].copy()
    temp['land_idx'] = temp['ISO3'].map({c: i for i, c in enumerate(landen)})

    new_rows = []
    for iso in landen:
        sub = temp[temp['ISO3'] == iso]
        if sub.empty:
            continue
        i = sub.index[0]
        ci = sub.at[i, 'land_idx']

        lv = last_vals[last_vals['ISO3'] == iso].sort_values('Jaar')
        y1 = lv['kwetsbaarheid'].iloc[-1]
        y2 = lv['kwetsbaarheid'].iloc[-2]
        x1 = {var: lv[var].iloc[-1] for var in variabelen_kort}

        mu_fc = phi_y1_mean[ci] * y1 + phi_y2_mean[ci] * y2
        for var in variabelen_kort:
            mu_fc += phi_x_mean[var][ci] * x1[var]

        temp.at[i, 'kwetsbaarheid_forecast'] = mu_fc

        nieuw = {'ISO3': iso, 'Jaar': pd.to_datetime(f"{jaar}-12-31"), 'kwetsbaarheid': mu_fc}
        for var in variabelen_kort:
            nieuw[var] = sub.at[i, var]
        new_rows.append(nieuw)

    last_vals = pd.concat([last_vals, pd.DataFrame(new_rows)], ignore_index=True)
    last_vals = (
        last_vals
        .sort_values(['ISO3','Jaar'])
        .groupby('ISO3')
        .tail(2)
        .reset_index(drop=True)
    )
    forecast_records.append(temp)

forecast_df = pd.concat(forecast_records).sort_values(['ISO3','Jaar']).reset_index(drop=True)

#outofsampleing
y_true_out = test_lang[test_lang['Jaar'].dt.year.isin([2021, 2022])].sort_values(['ISO3','Jaar'])['kwetsbaarheid'].values
y_pred_out = forecast_df['kwetsbaarheid_forecast'].values
mae_out = mean_absolute_error(y_true_out, y_pred_out)
rmse_out = np.sqrt(mean_squared_error(y_true_out, y_pred_out))
print(f"Out-of-sample MAE (2021–22): {mae_out:.4f}, RMSE: {rmse_out:.4f}")

# plotten
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
jaren = [2021, 2022]
for ax, iso in zip(axes.flatten(), landen):
    echt = test_lang[(test_lang['ISO3'] == iso) & (test_lang['Jaar'].dt.year.isin(jaren))]
    fc = forecast_df[forecast_df['ISO3'] == iso]
    ax.plot(echt['Jaar'].dt.year, echt['kwetsbaarheid'], 'o-', label='Echt')
    ax.plot(fc['Jaar'].dt.year, fc['kwetsbaarheid_forecast'], 'x--', label='Forecast')
    ax.set_title(iso)
    ax.set_xticks(jaren)
    if iso == landen[0]:
        ax.legend()
plt.tight_layout()
plt.show()

#latex
latex_tabel = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
Model & AIC & BIC & MAE (2021--2022) & RMSE (2021--2022) \\\
\\midrule
Bayesiaanse ARDL(2,1) met shrinkage & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{ARDL(2,1) met Bayesian shrinkage: In-sample (AIC, BIC) en out-of-sample forecast errors (2021--2022)}}
\\label{{tab:ardl21_bayes_shrinkage}}
\\end{{table}}
"""
print(latex_tabel)
