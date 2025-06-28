import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#data
pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"
gegevens_train = pd.read_excel(pad_train)
gegevens_test = pd.read_excel(pad_test)

# exppanatory variablen
variabelen_kort = ['food', 'water', 'health', 'eco', 'infra', 'habi']


def naar_longformaat(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen_kort + ['vulner'],
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix='\\d{4}'
    ).reset_index()
    return df_lang

gegevens_train_lang = naar_longformaat(gegevens_train)
gegevens_test_lang = naar_longformaat(gegevens_test)

gegevens_train_lang['land_idx'] = gegevens_train_lang['ISO3'].astype('category').cat.codes
land_namen = gegevens_train_lang['ISO3'].astype('category').cat.categories
aantal_landen = gegevens_train_lang['land_idx'].nunique()

#1 lag
for var in variabelen_kort:
    gegevens_train_lang[f'{var}_lag1'] = gegevens_train_lang.groupby('ISO3')[var].shift(1)
# Verwijder rijen met ontbrekende waarden
gegevens_train_lang = gegevens_train_lang.dropna().copy()

# === 4. Bayesian model met variational inference ===
with pm.Model() as model:
    mu_phi = pm.Normal('mu_phi', mu=0.2, sigma=0.3)
    sigma_phi = pm.Exponential('sigma_phi', 2.0)

    # Regionale parameters per variabele
    phi = {
        var: pm.Normal(
            f'phi_{var}',
            mu=mu_phi,
            sigma=sigma_phi,
            shape=aantal_landen
        )
        for var in variabelen_kort
    }

    sigma = pm.Exponential('sigma', lam=1/0.03)

    #mean forecasts doen
    mu = sum(
        phi[var][gegevens_train_lang['land_idx'].values] * gegevens_train_lang[f'{var}_lag1'].values
        for var in variabelen_kort
    ) / len(variabelen_kort)

    #werkelijke y
    y_obs = pm.Normal(
        'y_obs',
        mu=mu,
        sigma=sigma,
        observed=gegevens_train_lang['vulner'].values
    )

    #vdi
    approx = pm.fit(method="advi", n=20000)
    trace = approx.sample(1000)

#in sample
X_lag = np.stack([gegevens_train_lang[f'{v}_lag1'].values for v in variabelen_kort], axis=1)
phi_means = np.stack([
    trace.posterior[f'phi_{v}'].mean(('chain', 'draw')).values[gegevens_train_lang['land_idx'].values]
    for v in variabelen_kort
], axis=1)

# prediction
voorsp_in = phi_means * X_lag
vulner_pred = voorsp_in.mean(axis=1)
vulner_true = gegevens_train_lang['vulner'].values

# mse en mae in sample doen
in_sample_mse = mean_squared_error(vulner_true, vulner_pred)
in_sample_mae = mean_absolute_error(vulner_true, vulner_pred)

n = len(vulner_true)
k = len(variabelen_kort)
aic = n * np.log(in_sample_mse) + 2 * k
bic = n * np.log(in_sample_mse) + np.log(n) * k

#latex
laatste_X = (
    gegevens_train_lang
    .sort_values('Year')
    .groupby('ISO3')[variabelen_kort]
    .last()
    .copy()
)
land_idx_map = dict(zip(land_namen, range(aantal_landen)))

#posterior phi
phi_post = {
    var: trace.posterior[f'phi_{var}'].mean(('chain', 'draw')).values
    for var in variabelen_kort
}

#loop voor 2021 en 2022
resultaten_prognose = []
prognose_df = gegevens_test_lang.copy()
prognose_df['land_idx'] = prognose_df['ISO3'].map(land_idx_map)

for jaar in [2021, 2022]:
    temp = prognose_df[prognose_df['Year'] == jaar].copy()

    # Voeg lag-1 X toe
    for var in variabelen_kort:
        temp[f'{var}_lag1'] = temp['ISO3'].map(laatste_X[var])

    # Voorspel X_it
    for var in variabelen_kort:
        temp[var] = phi_post[var][temp['land_idx'].values] * temp[f'{var}_lag1'].values

    # Bereken kwetsbaarheid
    X_mat = np.stack([temp[var].values for var in variabelen_kort], axis=1)
    phi_mat = np.stack([phi_post[var][temp['land_idx'].values] for var in variabelen_kort], axis=1)
    temp['vulner_forecast'] = (phi_mat * X_mat).mean(axis=1)

    # Update X voor volgend jaar
    for var in variabelen_kort:
        laatste_X[var] = temp[var].values

    resultaten_prognose.append(temp)

prognose_df = pd.concat(resultaten_prognose).sort_values(['ISO3', 'Year']).reset_index(drop=True)

#results
y_true = prognose_df['vulner'].values
y_pred = prognose_df['vulner_forecast'].values
mae_out = mean_absolute_error(y_true, y_pred)
rmse_out = np.sqrt(mean_squared_error(y_true, y_pred))

latex_tabel = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021–22) & RMSE (2021–22) \\\\
\\midrule
Bayesiaans VARMA (shrinkage) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestaties: in-sample fit (AIC, BIC) en out-of-sample fouten (2021–2022).}}
\\end{{table}}
"""
print(latex_tabel)

#plot
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
jaarlijst = [2021, 2022]

for idx, ax in enumerate(axes.flatten()):
    if idx >= len(land_namen):
        ax.axis('off')
        continue
    naam = land_namen[idx]
    data_land = prognose_df[prognose_df['ISO3'] == naam]
    ax.plot(jaarlijst, data_land['vulner'], label='Werkelijk')
    ax.plot(jaarlijst, data_land['vulner_forecast'], '--', label='Prognose')
    ax.set_title(naam)
    ax.legend()

plt.tight_layout()
plt.show()
