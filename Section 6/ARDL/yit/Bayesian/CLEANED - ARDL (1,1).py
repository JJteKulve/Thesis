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


variabelen_kort = ['food', 'water', 'health', 'eco', 'infra', 'habi']
def naar_lang(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen_kort + ['vulner'],
        i=['ISO3','Name'], j='Jaar', sep='_', suffix=r'\d{4}'
    ).reset_index()
    df_lang.rename(columns={'vulner':'kwetsbaarheid'}, inplace=True)
    df_lang['Jaar'] = pd.to_datetime(df_lang['Jaar'].astype(str)+'-12-31')
    return df_lang

train_lang = naar_lang(data_train)
test_lang = naar_lang(data_test)

data_voorspelling.rename(columns={'Year':'Jaar'}, inplace=True)
data_voorspelling['Jaar'] = pd.to_datetime(data_voorspelling['Jaar'].astype(str)+'-12-31')


train_lang['land_idx'] = train_lang['ISO3'].astype('category').cat.codes
data_voorspelling['land_idx'] = data_voorspelling['ISO3'].map(
    dict(zip(train_lang['ISO3'].astype('category').cat.categories, range(len(train_lang['ISO3'].astype('category').cat.categories))))
)
landen = train_lang['ISO3'].astype('category').cat.categories
n_landen = len(landen)

#   lags
for var in ['vulner']+variabelen_kort:
    train_lang[f"{var}_lag1"] = train_lang.groupby('ISO3')[var].shift(1)
train_lang = train_lang.dropna().copy()

# ardl 1.1.1
with pm.Model() as ardl11:
    mu_phi = pm.Normal('mu_phi', mu=0.85, sigma=0.2)
    sigma_phi = pm.Exponential('sigma_phi', 2.0)
    phi_y = pm.Normal('phi_y', mu=mu_phi, sigma=sigma_phi, shape=n_landen)

    mu_beta = pm.Normal('mu_beta', mu=0.7, sigma=0.3)
    sigma_beta = pm.Exponential('sigma_beta', 2.0)
    beta_x = {var: pm.Normal(f'beta_{var}', mu=mu_beta, sigma=sigma_beta, shape=n_landen) for var in variabelen_kort}
    beta_x_lag = {var: pm.Normal(f'beta_{var}_lag', mu=mu_beta, sigma=sigma_beta, shape=n_landen) for var in variabelen_kort}

    sigma = pm.Exponential('sigma', lam=1/0.03)
    idx = train_lang['land_idx'].values
    mu = (
        phi_y[idx] * train_lang['vulner_lag1'].values +
        sum(beta_x[var][idx] * train_lang[var].values + beta_x_lag[var][idx] * train_lang[f'{var}_lag1'].values for var in variabelen_kort)
    )
    pm.Normal('waarnemingen', mu=mu, sigma=sigma, observed=train_lang['kwetsbaarheid'].values)
    approx = pm.fit(method='advi', n=20000)
    trace = approx.sample(1000)

# insamp
phi_y_mean = trace.posterior['phi_y'].mean(('chain','draw')).values
beta_x_mean = {var: trace.posterior[f'beta_{var}'].mean(('chain','draw')).values for var in variabelen_kort}
beta_x_lag_mean = {var: trace.posterior[f'beta_{var}_lag'].mean(('chain','draw')).values for var in variabelen_kort}
idx = train_lang['land_idx'].values
mu_pred = (
    phi_y_mean[idx] * train_lang['vulner_lag1'].values +
    sum(beta_x_mean[var][idx] * train_lang[var].values + beta_x_lag_mean[var][idx] * train_lang[f'{var}_lag1'].values for var in variabelen_kort)
)
y_true = train_lang['kwetsbaarheid'].values
mse_in = mean_squared_error(y_true, mu_pred)
mae_in = mean_absolute_error(y_true, mu_pred)
n = len(y_true)
k = 1 + 2 * len(variabelen_kort)
aic = n * np.log(mse_in) + 2*k
bic = n * np.log(mse_in) + np.log(n)*k
print(f"In-sample: AIC={aic:.2f}, BIC={bic:.2f}, MAE={mae_in:.4f}")

    # forecasting
last_vals = train_lang.sort_values('Jaar').groupby('ISO3')[['kwetsbaarheid']+variabelen_kort].last()
forecast_list = []
for year in [2021,2022]:
    df_temp = data_voorspelling[data_voorspelling['Jaar'].dt.year==year].copy()
    df_temp = df_temp[df_temp['ISO3'].isin(landen)].copy()
    df_temp['land_idx'] = df_temp['ISO3'].map({c:i for i,c in enumerate(landen)})

    df_temp['vulner_lag1'] = df_temp['ISO3'].map(last_vals['kwetsbaarheid'])
    for var in variabelen_kort:
        df_temp[f'{var}_lag1'] = df_temp['ISO3'].map(last_vals[var])

    idx = df_temp['land_idx'].values
    mu_fc = (
        phi_y_mean[idx] * df_temp['vulner_lag1'].values +
        sum(beta_x_mean[var][idx] * df_temp[var].values + beta_x_lag_mean[var][idx] * df_temp[f'{var}_lag1'].values for var in variabelen_kort)
    )
    df_temp['kwetsbaarheid_forecast'] = mu_fc
    for iso,val in zip(df_temp['ISO3'], mu_fc): last_vals.at[iso,'kwetsbaarheid']=val
    for var in variabelen_kort: last_vals[var] = df_temp[var].values

    true_vals = test_lang[test_lang['Jaar'].dt.year==year].set_index('ISO3')['kwetsbaarheid']
    df_temp['kwetsbaarheid_echt'] = df_temp['ISO3'].map(true_vals)
    forecast_list.append(df_temp)

forecast_df = pd.concat(forecast_list).sort_values(['ISO3','Jaar']).reset_index(drop=True)

# OOSAMP
y_true_out = test_lang[test_lang['Jaar'].dt.year.isin([2021,2022])].sort_values(['ISO3','Jaar'])['kwetsbaarheid'].values
y_pred_out = forecast_df['kwetsbaarheid_forecast'].values
mae_out = mean_absolute_error(y_true_out, y_pred_out)
rmse_out = np.sqrt(mean_squared_error(y_true_out, y_pred_out))
print(f"MAE (2021-22): {mae_out:.4f}, RMSE: {rmse_out:.4f}")

# PLOTTEN
fig,axes = plt.subplots(4,2,figsize=(14,10),sharex=True)
jaren=[2021,2022]
for ax,iso in zip(axes.flatten(),landen):
    true_d = test_lang[(test_lang['ISO3']==iso)&(test_lang['Jaar'].dt.year.isin(jaren))]
    pred_d = forecast_df[forecast_df['ISO3']==iso]
    ax.plot(true_d['Jaar'].dt.year,true_d['kwetsbaarheid'],'o-',label='Echt')
    ax.plot(pred_d['Jaar'].dt.year,pred_d['kwetsbaarheid_forecast'],'x--',label='Forecast')
    ax.set_title(iso); ax.set_xticks(jaren)
    if iso==landen[0]: ax.legend()
plt.tight_layout(); plt.show()

# LAT
latex = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
Model & AIC & BIC & MAE (2021--2022) & RMSE (2021--2022) \\\
\\midrule
Bayesiaanse ARDL(1,1) met shrinkage & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{ARDL(0,1) met Bayesian shrinkage: In-sample (AIC, BIC) en out-of-sample forecast errors (2021--2022)}}
\\label{{tab:ardl01_bayes_shrinkage_beta}}
\\end{{table}}
"""
print(latex)
