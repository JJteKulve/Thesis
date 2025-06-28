import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# data
train_path = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
test_path  = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"

df_train = pd.read_excel(train_path)
df_test  = pd.read_excel(test_path)

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
df_train_lang['kwetsbaarheid_lag1'] = (
    df_train_lang.groupby('ISO3')['kwetsbaarheid']
    .shift(1)
)
df_train_lang = df_train_lang.dropna().copy()

#ar1
with pm.Model() as model:
    mu_phi    = pm.Normal('mu_phi', mu=0.85, sigma=0.1)
    sigma_phi = pm.Exponential('sigma_phi', 5.0)
    phi       = pm.Normal(
        'phi',
        mu=mu_phi,
        sigma=sigma_phi,
        shape=aantal_landen
    )

    sigma = pm.Exponential('sigma', lam=1/0.03)

    mu = phi[df_train_lang['land_idx'].values] * df_train_lang['kwetsbaarheid_lag1'].values


    y_obs = pm.Normal(
        'y_obs',
        mu=mu,
        sigma=sigma,
        observed=df_train_lang['kwetsbaarheid'].values
    )


    approx = pm.fit(method="advi", n=15000)
    trace  = approx.sample(1000)


phi_gemiddeld = trace.posterior['phi'].mean(('chain', 'draw')).values
phi_vec = phi_gemiddeld[df_train_lang['land_idx'].values]


y_pred_in = phi_vec * df_train_lang['kwetsbaarheid_lag1'].values
y_true_in = df_train_lang['kwetsbaarheid'].values


mse_in = mean_squared_error(y_true_in, y_pred_in)
mae_in = mean_absolute_error(y_true_in, y_pred_in)
n = len(y_true_in)
k = 1 
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k


forecast_df = df_test_lang.sort_values(['ISO3', 'Jaar']).reset_index(drop=True)


laatste_waarde = (
    df_train_lang.sort_values('Jaar')
    .groupby('ISO3')['kwetsbaarheid']
    .last()
    .copy()
)


land_idx_map = dict(zip(land_codes, range(aantal_landen)))

records = []
for jaar in [2021, 2022]:
    for iso3 in land_codes:
        phi_i = phi_gemiddeld[land_idx_map[iso3]]
        y_lag = laatste_waarde[iso3]
        y_hat = phi_i * y_lag


        y_true = forecast_df.loc[
            (forecast_df['ISO3'] == iso3) &
            (forecast_df['Jaar'] == jaar),
            'kwetsbaarheid'
        ].values[0]

        records.append({
            'ISO3': iso3,
            'Jaar': jaar,
            'kwetsbaarheid_lag1': y_lag,
            'kwetsbaarheid_voorspelling': y_hat,
            'kwetsbaarheid_werkelijk': y_true
        })


        laatste_waarde[iso3] = y_hat

forecast_result = pd.DataFrame(records)


mae_out  = mean_absolute_error(
    forecast_result['kwetsbaarheid_werkelijk'],
    forecast_result['kwetsbaarheid_voorspelling']
)
rmse_out = np.sqrt(mean_squared_error(
    forecast_result['kwetsbaarheid_werkelijk'],
    forecast_result['kwetsbaarheid_voorspelling']
))


fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i >= len(land_codes):
        ax.axis('off')
        continue
    iso3 = land_codes[i]
    data = forecast_result[forecast_result['ISO3'] == iso3]
    ax.plot(
        data['Jaar'],
        data['kwetsbaarheid_werkelijk'],
        label='Waargenomen',
        color='black'
    )
    ax.plot(
        data['Jaar'],
        data['kwetsbaarheid_voorspelling'],
        '--',
        color='red',
        label='Voorspelling'
    )
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
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\\
\\midrule
Bayesiaans AR(1) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestatie: in-sample fit (AIC, BIC) en out-of-sample forecast fouten (2021--2022).}}
\\end{{table}}
"""
print(latex)
