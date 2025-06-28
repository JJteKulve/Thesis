import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"  
pad_test  = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"  
variabelen = ['food', 'water', 'health', 'eco', 'infra', 'habi']

def lees_excel(pad):  
    return pd.read_excel(pad)

df_train = lees_excel(pad_train)
df_test  = lees_excel(pad_test)
  
def zet_naar_lang(df):  
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen + ['vulner'],
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix='\\d{4}'
    ).reset_index()
    df_lang['Year'] = df_lang['Year'].astype(int)
    return df_lang

train_lang = zet_naar_lang(df_train)
test_lang  = zet_naar_lang(df_test)

train_lang['land_idx'] = train_lang['ISO3'].astype('category').cat.codes
land_namen = train_lang['ISO3'].astype('category').cat.categories
n_landen   = train_lang['land_idx'].nunique()

#2.1
for v in variabelen:
    train_lang[f'{v}_lag1'] = train_lang.groupby('ISO3')[v].shift(1)
    train_lang[f'{v}_lag2'] = train_lang.groupby('ISO3')[v].shift(2)

# Extra variabelen voor residu
train_lang['vulner_lag1'] = train_lang.groupby('ISO3')['vulner'].shift(1)
train_lang['x_mean_lag1'] = train_lang[[f'{v}_lag1' for v in variabelen]].mean(axis=1)
train_lang['x_mean_lag2'] = train_lang[[f'{v}_lag2' for v in variabelen]].mean(axis=1)
train_lang['resid_lag1']  = train_lang['vulner_lag1'] - train_lang['x_mean_lag1']

train_lang = train_lang.dropna().reset_index(drop=True)

#BAY 2.1
with pm.Model() as varma21:
    mu_phi1    = pm.Normal('mu_phi1', mu=0.2, sigma=0.3)
    sigma_phi1 = pm.Exponential('sigma_phi1', 2.0)
    phi1       = {
        v: pm.Normal(f'phi1_{v}', mu=mu_phi1, sigma=sigma_phi1, shape=n_landen)
        for v in variabelen
    }


    mu_phi2    = pm.Normal('mu_phi2', mu=0.2, sigma=0.3)
    sigma_phi2 = pm.Exponential('sigma_phi2', 2.0)
    phi2       = {
        v: pm.Normal(f'phi2_{v}', mu=mu_phi2, sigma=sigma_phi2, shape=n_landen)
        for v in variabelen
    }


    mu_theta   = pm.Normal('mu_theta', mu=0.0, sigma=0.2)
    sigma_theta= pm.Exponential('sigma_theta', 5.0)
    theta      = pm.Normal('theta', mu=mu_theta, sigma=sigma_theta, shape=n_landen)

    sigma_y    = pm.Exponential('sigma_y', lam=1/0.03)

    idx = train_lang['land_idx'].values
    ar1 = sum(phi1[v][idx] * train_lang[f'{v}_lag1'].values for v in variabelen)
    ar2 = sum(phi2[v][idx] * train_lang[f'{v}_lag2'].values for v in variabelen)
    ar  = (ar1 + ar2) / len(variabelen)
    ma  = theta[idx] * train_lang['resid_lag1'].values
    mu  = ar + ma


    pm.Normal('y_obs', mu=mu, sigma=sigma_y, observed=train_lang['vulner'].values)


    approx = pm.fit(method='advi', n=20000)
    trace  = approx.sample(1000)

 
x1 = np.stack([train_lang[f'{v}_lag1'] for v in variabelen], axis=1)
x2 = np.stack([train_lang[f'{v}_lag2'] for v in variabelen], axis=1)
r  = train_lang['resid_lag1'].values
idx = train_lang['land_idx'].values

phi1_m = np.stack([trace.posterior[f'phi1_{v}'].mean(('chain','draw')).values[idx] for v in variabelen], axis=1)
phi2_m = np.stack([trace.posterior[f'phi2_{v}'].mean(('chain','draw')).values[idx] for v in variabelen], axis=1)
theta_m= trace.posterior['theta'].mean(('chain','draw')).values[idx]

ar_pred= ((phi1_m * x1) + (phi2_m * x2)).mean(axis=1)
v_pred = ar_pred + theta_m * r
v_true = train_lang['vulner'].values

mse_in  = mean_squared_error(v_true, v_pred)
mae_in  = mean_absolute_error(v_true, v_pred)
n       = len(v_true)
k       = 2 * len(variabelen) + 1
aic     = n * np.log(mse_in) + 2 * k
bic     = n * np.log(mse_in) + np.log(n) * k

#OFRECASAT
forecast_df = test_lang.sort_values(['ISO3','Year']).reset_index(drop=True)
forecast_df['land_idx'] = forecast_df['ISO3'].astype('category').cat.codes
last_vals    = train_lang.set_index(['ISO3','Year']).sort_index()
fc_2021      = None
v_fc         = []

#posterior gemi
phi1_p = {v: trace.posterior[f'phi1_{v}'].mean(('chain','draw')).values for v in variabelen}
phi2_p = {v: trace.posterior[f'phi2_{v}'].mean(('chain','draw')).values for v in variabelen}
theta_p= trace.posterior['theta'].mean(('chain','draw')).values

for jaar in [2021, 2022]:
    sub = forecast_df[forecast_df['Year']==jaar].copy()
    idx = sub['land_idx'].values
    iso = sub['ISO3'].values

    if jaar==2021:
        lag1 = last_vals.xs(2020, level='Year')
        lag2 = last_vals.xs(2019, level='Year')
    else:
        lag1 = last_vals.xs(2021, level='Year').loc[iso].copy()
        lag1['vulner'] = fc_2021
        lag2 = last_vals.xs(2020, level='Year')

    for v in variabelen:
        lag1[v] = sub.set_index('ISO3').loc[iso, v].values

    X1 = np.stack([lag1.loc[iso, v] for v in variabelen], axis=1)
    X2 = np.stack([lag2.loc[iso, v] for v in variabelen], axis=1)
    mu1= (np.stack([phi1_p[v][idx] * X1[:, i] for i, v in enumerate(variabelen)], axis=1).mean(axis=1))
    mu2= (np.stack([phi2_p[v][idx] * X2[:, i] for i, v in enumerate(variabelen)], axis=1).mean(axis=1))
    ma = theta_p[idx] * (lag1['vulner'] - X1.mean(axis=1))
    fc = (mu1+mu2)/2 + ma
    v_fc.extend(fc)
    if jaar==2021:
        fc_2021 = fc.copy()

    mi = pd.MultiIndex.from_arrays([iso, [jaar]*len(iso)], names=['ISO3','Year'])
    nr = pd.DataFrame(index=mi, columns=last_vals.columns)
    last_vals = pd.concat([last_vals, nr])
    last_vals.loc[(iso,jaar),'vulner'] = fc
    for v in variabelen:
        last_vals.loc[(iso,jaar), v] = sub.set_index('ISO3').loc[iso, v].values

v_fc = np.array(v_fc)

#OOS
y_true = forecast_df[forecast_df['Year'].isin([2021,2022])]['vulner'].values
mae_out= mean_absolute_error(y_true, v_fc)
rmse_out=np.sqrt(mean_squared_error(y_true, v_fc))

# RESULTS
fig, axes = plt.subplots(4,2,figsize=(14,10), sharex=True)
for i, ax in enumerate(axes.flatten()):
    if i>=len(land_namen): ax.axis('off'); continue
    land = land_namen[i]
    df_l = forecast_df[forecast_df['ISO3']==land]
    ax.plot([2021,2022], df_l['vulner'], label='Werkelijk')
    ax.plot([2021,2022], v_fc[i*2:(i*2+2)], '--', label='Forecast')
    ax.set_title(land)
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
Bayesiaans VARMA(2,1) & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Modelprestaties: in-sample fit (AIC, BIC) en out-of-sample fouten (2021–2022).}}
\\end{{table}}
"""
print(latex)
