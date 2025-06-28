import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#data
train_path = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
test_path  = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"

df_train = pd.read_excel(train_path)
df_test  = pd.read_excel(test_path)

# lang
def to_long(df):
    df_long = pd.wide_to_long(df,
        stubnames=['vulner'],
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    return df_long

df_train_long = to_long(df_train)
df_test_long  = to_long(df_test)

df_train_long['country_idx'] = df_train_long['ISO3'].astype('category').cat.codes
country_names = df_train_long['ISO3'].astype('category').cat.categories
n_countries = df_train_long['country_idx'].nunique()


df_train_long = df_train_long.sort_values(['ISO3', 'Year'])
df_train_long['vulner_lag1'] = df_train_long.groupby('ISO3')['vulner'].shift(1)
df_train_long['vulner_lag2'] = df_train_long.groupby('ISO3')['vulner'].shift(2)
df_train_long['eps_lag1'] = df_train_long.groupby('ISO3')['vulner'].diff() - df_train_long['vulner_lag1'].diff()
df_train_long = df_train_long.dropna().copy()

#arma2.1
with pm.Model() as model:
    mu_phi1 = pm.Normal('mu_phi1', mu=0.6, sigma=0.2)
    sigma_phi1 = pm.Exponential('sigma_phi1', 5.0)
    phi1 = pm.Normal('phi1', mu=mu_phi1, sigma=sigma_phi1, shape=n_countries)

    mu_phi2 = pm.Normal('mu_phi2', mu=0.2, sigma=0.2)
    sigma_phi2 = pm.Exponential('sigma_phi2', 5.0)
    phi2 = pm.Normal('phi2', mu=mu_phi2, sigma=sigma_phi2, shape=n_countries)

    mu_theta = pm.Normal('mu_theta', mu=0.0, sigma=0.2)
    sigma_theta = pm.Exponential('sigma_theta', 5.0)
    theta = pm.Normal('theta', mu=mu_theta, sigma=sigma_theta, shape=n_countries)

    sigma = pm.Exponential('sigma', lam=1/0.03)

    idx = df_train_long['country_idx'].values
    mu = (
        phi1[idx] * df_train_long['vulner_lag1'].values +
        phi2[idx] * df_train_long['vulner_lag2'].values +
        theta[idx] * df_train_long['eps_lag1'].values
    )

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=df_train_long['vulner'].values)
    
    approx = pm.fit(method="advi", n=15000)
    trace = approx.sample(1000)

#insample doen
phi1_means = trace.posterior['phi1'].mean(('chain', 'draw')).values
phi2_means = trace.posterior['phi2'].mean(('chain', 'draw')).values
theta_means = trace.posterior['theta'].mean(('chain', 'draw')).values

idx_vec = df_train_long['country_idx'].values
phi1_vec = phi1_means[idx_vec]
phi2_vec = phi2_means[idx_vec]
theta_vec = theta_means[idx_vec]

vulner_pred_in = (
    phi1_vec * df_train_long['vulner_lag1'].values +
    phi2_vec * df_train_long['vulner_lag2'].values +
    theta_vec * df_train_long['eps_lag1'].values
)
vulner_true_in = df_train_long['vulner'].values

in_sample_mse = mean_squared_error(vulner_true_in, vulner_pred_in)
in_sample_mae = mean_absolute_error(vulner_true_in, vulner_pred_in)
n = len(vulner_true_in)
k = 3
aic = n * np.log(in_sample_mse) + 2 * k
bic = n * np.log(in_sample_mse) + np.log(n) * k

# forecasting qweer
forecast_df = df_test_long.sort_values(['ISO3', 'Year']).copy()
last_data = (
    df_train_long
    .sort_values(['ISO3', 'Year'])
    .groupby('ISO3')
    .tail(2)
    .reset_index(drop=True)
    .copy()
)

forecast_records = []
country_idx_map = dict(zip(country_names, range(n_countries)))

for year in [2021, 2022]:
    new_last_data = []
    for iso3 in country_names:
        lag_data = last_data[last_data['ISO3'] == iso3].sort_values('Year')

        if len(lag_data) < 2:
            print(f"Waarschuwing: te weinig data om te forecasten voor {iso3} in {year}.")
            continue

        idx = country_idx_map[iso3]
        phi1_i = phi1_means[idx]
        phi2_i = phi2_means[idx]
        theta_i = theta_means[idx]

        y_lag1 = lag_data['vulner'].iloc[-1]
        y_lag2 = lag_data['vulner'].iloc[-2]

        eps_lag1 = lag_data['vulner'].iloc[-1] - (
            phi1_i * lag_data['vulner_lag1'].iloc[-1] +
            phi2_i * lag_data['vulner_lag2'].iloc[-1] +
            theta_i * lag_data['eps_lag1'].iloc[-1]
        )

        y_hat = phi1_i * y_lag1 + phi2_i * y_lag2 + theta_i * eps_lag1
        y_true = forecast_df.loc[
            (forecast_df['ISO3'] == iso3) & (forecast_df['Year'] == year), 'vulner'
        ].values[0]

        forecast_records.append({
            'ISO3': iso3,
            'Year': year,
            'vulner_forecast': y_hat,
            'vulner_true': y_true
        })

        new_last_data.append({
            'ISO3': iso3,
            'vulner': y_hat,
            'vulner_lag1': y_lag1,
            'vulner_lag2': y_lag2,
            'eps_lag1': y_true - y_hat,
            'Year': year
        })

last_data = pd.concat([last_data, pd.DataFrame(new_last_data)], ignore_index=True)
last_data = (
    last_data.sort_values(['ISO3', 'Year'])
    .groupby('ISO3')
    .tail(2)
    .reset_index(drop=True)
)


forecast_result = pd.DataFrame(forecast_records)

# pouf of sample
mae_out = mean_absolute_error(forecast_result['vulner_true'], forecast_result['vulner_forecast'])
rmse_out = np.sqrt(mean_squared_error(forecast_result['vulner_true'], forecast_result['vulner_forecast']))

#result
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i >= len(country_names):
        ax.axis('off')
        continue
    name = country_names[i]
    data = forecast_result[forecast_result['ISO3'] == name]
    ax.plot(data['Year'], data['vulner_true'], label='Echt', color='black')
    ax.plot(data['Year'], data['vulner_forecast'], '--', color='red', label='Forecast')
    ax.set_title(f'{name}')
    ax.legend()

plt.tight_layout()
plt.show()

latex = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\\
\\midrule
Bayesian ARMA(2,1) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Model performance: ARMA(2,1) met Bayesiaanse schatting, in-sample fit (AIC/BIC) en out-of-sample forecast errors voor 2021--2022.}}
\\end{{table}}
"""
print(latex)
