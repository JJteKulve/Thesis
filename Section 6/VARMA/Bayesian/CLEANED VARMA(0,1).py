import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#data
pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"
train_df = pd.read_excel(pad_train)
test_df = pd.read_excel(pad_test)

variabelen_kort = ['food', 'water', 'health', 'eco', 'infra', 'habi']

def naar_long(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen_kort + ['vulner'],
        i=['ISO3', 'Name'],
        j='Year',
        sep='_',
        suffix='\\d{4}'
    ).reset_index()
    df_lang['Year'] = df_lang['Year'].astype(int)
    return df_lang

train_lang = naar_long(train_df)
test_lang = naar_long(test_df)

train_lang['land_idx'] = train_lang['ISO3'].astype('category').cat.codes
land_namen = train_lang['ISO3'].astype('category').cat.categories
aantal_landen = train_lang['land_idx'].nunique()

#residuals 1 lag 
train_lang['x_gemiddeld_lag1'] = train_lang.groupby('ISO3')[variabelen_kort].shift(1).mean(axis=1)
train_lang['vulner_lag1']  = train_lang.groupby('ISO3')['vulner'].shift(1)
train_lang['resid_lag1']   = train_lang['vulner_lag1'] - train_lang['x_gemiddeld_lag1']
train_lang = train_lang.dropna().copy()

#bayesian varma 0,1
with pm.Model() as model:
    mu_theta1    = pm.Normal('mu_theta1', mu=0.0, sigma=0.2)
    sigma_theta1 = pm.Exponential('sigma_theta1', 5)
    theta1       = pm.Normal('theta1', mu=mu_theta1, sigma=sigma_theta1, shape=aantal_landen)
    sigma        = pm.Exponential('sigma', lam=1/0.03)

    idx = train_lang['land_idx'].values
    mu  = theta1[idx] * train_lang['resid_lag1'].values
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=train_lang['vulner'].values)

    approx = pm.fit(method='advi', n=20000)
    trace  = approx.sample(1000)

#in sample evaluatie
theta1_mean = trace.posterior['theta1'].mean(dim=('chain','draw')).values[train_lang['land_idx'].values]
res1        = train_lang['resid_lag1'].values
vul_pred_in = theta1_mean * res1
vul_true_in = train_lang['vulner'].values

mse_in = mean_squared_error(vul_true_in, vul_pred_in)
mae_in = mean_absolute_error(vul_true_in, vul_pred_in)
n = len(vul_true_in)
k = 1
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k

#2021 en 2022
prognose_df = test_lang.sort_values(['ISO3','Year']).reset_index(drop=True)
prognose_df['land_idx'] = prognose_df['ISO3'].astype('category').cat.codes

recente = train_lang.set_index(['ISO3','Year']).copy()
doorslag = {}
vul_fc  = []
theta1_post = trace.posterior['theta1'].mean(dim=('chain','draw')).values
for jaar in [2021,2022]:
    this = prognose_df[prognose_df['Year']==jaar].copy()
    idx = this['land_idx'].values
    iso = this['ISO3'].values

    prev = recente.xs(jaar-1,level='Year').copy()
    if jaar==2022:
        prev['vulner'] = doorslag[2021]
    for v in variabelen_kort:
        prev[v] = this.set_index('ISO3').loc[iso,v].values

    resid = prev['vulner'] - prev[variabelen_kort].mean(axis=1)
    pred  = theta1_post[idx] * resid.values
    doorslag[jaar] = pred
    vul_fc.extend(pred)

    idxs = pd.MultiIndex.from_arrays([iso,[jaar]*len(iso)], names=['ISO3','Year'])
    new  = pd.DataFrame(index=idxs, columns=recente.columns)
    recente = pd.concat([recente,new])
    recente.loc[(iso,jaar),'vulner'] = pred
    for v in variabelen_kort:
        recente.loc[(iso,jaar),v] = this.set_index('ISO3').loc[iso,v].values

#evaluatie out of sample
y_true_out = prognose_df[prognose_df['Year'].isin([2021,2022])]['vulner'].values
mae_out = mean_absolute_error(y_true_out, vul_fc)
rmse_out  = np.sqrt(mean_squared_error(y_true_out, vul_fc))

#latex
latex_tabel = f"""
\begin{{table}}[htbp]
\centering
\small
\begin{{tabular}}{{lcccc}}
\toprule
 & AIC & BIC & MAE (2021--22) & RMSE (2021--22) \\
\midrule
VARMA(0,1) Bayesiaans shrinkage & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\
\bottomrule
\end{{tabular}}
\caption{{Modelprestaties: in-sample fit (AIC, BIC) en out-of-sample fouten (2021–2022).}}
\end{{table}}
"""
print(latex_tabel)

# =plotten
fig,axes = plt.subplots(4,2,figsize=(14,10),sharex=True)
for i,ax in enumerate(axes.flatten()):
    if i>=len(land_namen): ax.axis('off'); continue
    land = land_namen[i]
    data = prognose_df[prognose_df['ISO3']==land]
    y_pred_year1 = doorslag[2021][i]
    y_pred_year2 = doorslag[2022][i]
    ax.plot([2021,2022], data['vulner'], label='Werkelijk')
    ax.plot([2021,2022], [y_pred_year1,y_pred_year2], '--', label='Prognose')
    ax.set_title(land)
    ax.legend()
plt.tight_layout()
plt.show()
