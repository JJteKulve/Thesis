import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#DATA
pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test  = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"

df_train = pd.read_excel(pad_train)
df_test  = pd.read_excel(pad_test)
korte_variabelen = ['food', 'water', 'health', 'eco', 'infra', 'habi']

def maak_lange_vorm(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=korte_variabelen + ['vulner'],
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix=r'\d{4}'
    ).reset_index()
    df_lang['Year'] = df_lang['Year'].astype(int)
    return df_lang

df_train_lang = maak_lange_vorm(df_train)
df_test_lang  = maak_lange_vorm(df_test)

df_train_lang['land_idx'] = df_train_lang['ISO3'].astype('category').cat.codes
land_namen    = df_train_lang['ISO3'].astype('category').cat.categories
aantal_landen = df_train_lang['land_idx'].nunique()

# lags
for var in korte_variabelen:
    df_train_lang[f'{var}_vertraging1'] = df_train_lang.groupby('ISO3')[var].shift(1)
df_train_lang['vulner_vertraging1'] = df_train_lang.groupby('ISO3')['vulner'].shift(1)
df_train_lang['vulner_vertraging2'] = df_train_lang.groupby('ISO3')['vulner'].shift(2)

df_train_lang['gem_x_vertraging1'] = df_train_lang[[f'{v}_vertraging1' for v in korte_variabelen]].mean(axis=1)
df_train_lang['gem_x_vertraging2'] = (
    df_train_lang[[f'{v}_vertraging1' for v in korte_variabelen]].shift(1).mean(axis=1)
)

df_train_lang['residu_vertraging1'] = (
    df_train_lang['vulner_vertraging1'] - df_train_lang['gem_x_vertraging1']
)
df_train_lang['residu_vertraging2'] = (
    df_train_lang['vulner_vertraging2'] - df_train_lang['gem_x_vertraging2']
)

df_train_lang = df_train_lang.dropna().copy()

#Bayesiaans VARMA(1,2)
with pm.Model() as model:
    # AR
    mu_phi1    = pm.Normal('mu_phi1', mu=0.2, sigma=0.3)
    sigma_phi1 = pm.Exponential('sigma_phi1', 2)
    phi1 = {
        v: pm.Normal(f'phi1_{v}', mu=mu_phi1, sigma=sigma_phi1, shape=aantal_landen)
        for v in korte_variabelen
    }

    #MA(1) -MA(2)
    mu_theta1    = pm.Normal('mu_theta1', mu=0.0, sigma=0.2)
    sigma_theta1 = pm.Exponential('sigma_theta1', 5)
    theta1 = pm.Normal('theta1', mu=mu_theta1, sigma=sigma_theta1, shape=aantal_landen)

    mu_theta2    = pm.Normal('mu_theta2', mu=0.0, sigma=0.2)
    sigma_theta2 = pm.Exponential('sigma_theta2', 5)
    theta2 = pm.Normal('theta2', mu=mu_theta2, sigma=sigma_theta2, shape=aantal_landen)

    sigma = pm.Exponential('sigma', 1/0.03)

    land_idx = df_train_lang['land_idx'].values

    ar_raw  = sum(phi1[v][land_idx] * df_train_lang[f'{v}_vertraging1'].values
                  for v in korte_variabelen)
    deel_ar = ar_raw / len(korte_variabelen)

    deel_ma1 = theta1[land_idx] * df_train_lang['residu_vertraging1'].values
    deel_ma2 = theta2[land_idx] * df_train_lang['residu_vertraging2'].values
    deel_ma  = deel_ma1 + deel_ma2

    #mean ll
    mu = deel_ar + deel_ma

    y_obs = pm.Normal(
        'y_obs',
        mu=mu,
        sigma=sigma,
        observed=df_train_lang['vulner'].values
    )

    approx = pm.fit(method="advi", n=20000)
    trace  = approx.sample(1000)

#insample
x1 = np.stack(
    [df_train_lang[f'{v}_vertraging1'].values for v in korte_variabelen],
    axis=1
)
land_idx = df_train_lang['land_idx'].values

phi1_gem   = np.stack(
    [trace.posterior[f'phi1_{v}'].mean(('chain','draw')).values[land_idx]
     for v in korte_variabelen],
    axis=1
)
theta1_gem = trace.posterior['theta1'].mean(('chain','draw')).values[land_idx]
theta2_gem = trace.posterior['theta2'].mean(('chain','draw')).values[land_idx]

res1 = df_train_lang['residu_vertraging1'].values
res2 = df_train_lang['residu_vertraging2'].values

ar_voorspelling            = (phi1_gem * x1).mean(axis=1)
kwets_gt_gevoorspeld       = (
    ar_voorspelling + theta1_gem * res1 + theta2_gem * res2
)
kwets_gt_echt              = df_train_lang['vulner'].values

mse_in     = mean_squared_error(kwets_gt_echt, kwets_gt_gevoorspeld)
mae_in     = mean_absolute_error(kwets_gt_echt, kwets_gt_gevoorspeld)
n_in       = len(kwets_gt_echt)
k_in       = len(korte_variabelen) + 2
aic        = n_in * np.log(mse_in) + 2 * k_in
bic        = n_in * np.log(mse_in) + np.log(n_in) * k_in

# forecast
df_voorsp = df_test_lang.copy().sort_values(['ISO3','Year']).reset_index(drop=True)
df_voorsp['land_idx'] = df_voorsp['ISO3'].astype('category').cat.codes

laatste_waardes = df_train_lang.set_index(['ISO3','Year']).sort_index()
voorspellingen  = {}
kwets_voorsp     = []

phi1_vals   = {
    v: trace.posterior[f'phi1_{v}'].mean(('chain','draw')).values
    for v in korte_variabelen
}
theta1_vals = trace.posterior['theta1'].mean(('chain','draw')).values
theta2_vals = trace.posterior['theta2'].mean(('chain','draw')).values

for jaar in [2021, 2022]:
    df_j = df_voorsp[df_voorsp['Year'] == jaar].copy()
    idx = df_j['land_idx'].values
    landen = df_j['ISO3'].values

    lag1 = laatste_waardes.xs(jaar-1, level='Year').copy()
    lag2 = laatste_waardes.xs(jaar-2, level='Year')

    if jaar == 2022:
        lag1['vulner'] = voorspellingen[2021]

    for v in korte_variabelen:
        lag1[v] = df_j.set_index('ISO3').loc[landen, v].values

    x1 = np.stack([lag1.loc[landen, v] for v in korte_variabelen], axis=1)
    resid1 = lag1.loc[landen, 'vulner'] - x1.mean(axis=1)
    resid2 = lag2.loc[landen, 'vulner'] - x1.mean(axis=1)

    deel_phi   = np.stack(
        [phi1_vals[v][idx] * x1[:,i] for i,v in enumerate(korte_variabelen)],
        axis=1
    ).mean(axis=1)
    deel_theta = theta1_vals[idx]*resid1 + theta2_vals[idx]*resid2

    voorsp = deel_phi + deel_theta
    voorspellingen[jaar] = voorsp
    kwets_voorsp.extend(voorsp)

    if jaar == 2021:
        nieuwe_idx = pd.MultiIndex.from_arrays(
            [landen, [2021]*len(landen)],
            names=['ISO3','Year']
        )
        nieuwe_rijen = pd.DataFrame(index=nieuwe_idx, columns=laatste_waardes.columns)
        laatste_waardes = pd.concat([laatste_waardes, nieuwe_rijen])
        laatste_waardes.loc[(landen,2021),'vulner'] = voorsp
        for v in korte_variabelen:
            laatste_waardes.loc[(landen,2021),v] = df_j.set_index('ISO3').loc[landen,v].values

echte_jaarlijkse = df_voorsp[df_voorsp['Year'].isin([2021,2022])]['vulner'].values
mae_uit = mean_absolute_error(echte_jaarlijkse, kwets_voorsp)
rmse_uit = np.sqrt(mean_squared_error(echte_jaarlijkse, kwets_voorsp))

# resultaten lat4ex en plot
latex = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\\\
\\midrule
VARMA(1,2) Bayesian shrinkage & {aic:.2f} & {bic:.2f} & {mae_uit:.4f} & {rmse_uit:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestaties: in-sample fit (AIC, BIC) en out-of-sample forecast (2021--2022).}}
\\end{{table}}
"""
print(latex)

fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i >= len(land_namen):
        ax.axis('off')
        continue
    naam = land_namen[i]
    data = df_voorsp[df_voorsp['ISO3'] == naam]
    ax.plot([2021, 2022], data['vulner'], label='Echte waarden', color='black')
    ax.plot([2021, 2022], kwets_voorsp[i*2:i*2+2], '--', color='red', label='Voorspelling')
    ax.set_title(f'{naam}')
    ax.legend()
plt.tight_layout()
plt.show()
