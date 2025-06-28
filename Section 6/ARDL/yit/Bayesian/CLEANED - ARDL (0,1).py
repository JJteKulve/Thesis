import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#data
pad_train = r"/Users/macbookair/Desktop/8_landen_logit_train.xlsx"
pad_test = r"/Users/macbookair/Desktop/8_landen_logit_test.xlsx"
pad_voorspelling = r"/Users/macbookair/Desktop/xit_forecast_pooled_2021_2025.xlsx"

data_train = pd.read_excel(pad_train)
data_test = pd.read_excel(pad_test)
data_voorspelling = pd.read_excel(pad_voorspelling)

#lang
variabelen_kort = ['food', 'water', 'health', 'eco', 'infra', 'habi']
def naar_lang_formaat(df):
    df_lang = pd.wide_to_long(
        df,
        stubnames=variabelen_kort + ['vulner'],
        i=['ISO3', 'Name'],
        j='Jaar', sep='_', suffix=r'\d{4}'
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

#lags toevoegen
for var in variabelen_kort:
    train_lang[f"{var}_lag1"] = train_lang.groupby('ISO3')[var].shift(1)
train_lang = train_lang.dropna().copy()

# ardl 0,1 model zonder p
with pm.Model() as ardl01_model:
    mu_beta = pm.Normal('mu_beta', mu=0.7, sigma=0.3)
    sigma_beta = pm.Exponential('sigma_beta', 2.0)

    phi_lag1_per_variabele = {
        var: pm.Normal(f'phi_{var}_lag1', mu=mu_beta, sigma=sigma_beta, shape=n_landen)
        for var in variabelen_kort
    }
    sigma = pm.Exponential('sigma', lam=1/0.03)
    idx = train_lang['land_index'].values
    mu = sum(phi_lag1_per_variabele[var][idx] * train_lang[f"{var}_lag1"].values for var in variabelen_kort)
    pm.Normal('waarnemingen', mu=mu, sigma=sigma, observed=train_lang['kwetsbaarheid'].values)
    approx = pm.fit(method='advi', n=10000)
    trace = approx.sample(1000)

# insamp
phi_lag1_mean = {var: trace.posterior[f'phi_{var}_lag1'].mean(('chain','draw')).values for var in variabelen_kort}
idx = train_lang['land_index'].values
voorspelling_in = sum(phi_lag1_mean[var][idx] * train_lang[f"{var}_lag1"].values for var in variabelen_kort)
echt_in = train_lang['kwetsbaarheid'].values

mse_in = mean_squared_error(echt_in, voorspelling_in)
mae_in = mean_absolute_error(echt_in, voorspelling_in)
n = len(echt_in)
k = len(variabelen_kort)
aic = n * np.log(mse_in) + 2 * k
bic = n * np.log(mse_in) + np.log(n) * k
print(f"In-sample: AIC={aic:.2f}, BIC={bic:.2f}, MAE={mae_in:.4f}")

# forecasting
laatste_X = train_lang.sort_values(['ISO3','Jaar']).groupby('ISO3').last()[variabelen_kort].to_dict(orient='index')
laatste_kw = train_lang.sort_values(['ISO3','Jaar']).groupby('ISO3')['kwetsbaarheid'].last().to_dict()

voorsp_records = []
for jaar in [2021, 2022]:
    df_temp = data_voorspelling[data_voorspelling['Jaar'].dt.year == jaar].copy()

    df_temp = df_temp[df_temp['ISO3'].isin(lijst_landen)].copy()
    df_temp['land_index'] = df_temp['ISO3'].map({c: i for i, c in enumerate(lijst_landen)})

    for var in variabelen_kort:
        df_temp[f"{var}_lag1"] = df_temp['ISO3'].map(lambda iso: laatste_X[iso][var])

    idx = df_temp['land_index'].values
    voorsp = sum(phi_lag1_mean[var][idx] * df_temp[f"{var}_lag1"].values for var in variabelen_kort)
    df_temp['kwetsbaarheid_voorspelling'] = voorsp


    for _, row in df_temp.iterrows():
        iso = row['ISO3']
        laatste_kw[iso] = row['kwetsbaarheid_voorspelling']
        for var in variabelen_kort:
            laatste_X[iso][var] = row[var]

    echte_waarden = test_lang[test_lang['Jaar'].dt.year == jaar].set_index('ISO3')['kwetsbaarheid']
    df_temp['kwetsbaarheid_echt'] = df_temp['ISO3'].map(echte_waarden)
    voorsp_records.append(df_temp)

resultaat_voorspelling = pd.concat(voorsp_records).sort_values(['ISO3','Jaar']).reset_index(drop=True)

#OOSAMPING
echt_out = test_lang[test_lang['Jaar'].dt.year.isin([2021, 2022])].sort_values(['ISO3','Jaar'])['kwetsbaarheid'].values
voorsp_out = resultaat_voorspelling['kwetsbaarheid_voorspelling'].values
mae_out = mean_absolute_error(echt_out, voorsp_out)
rmse_out = np.sqrt(mean_squared_error(echt_out, voorsp_out))
print(f"Out-of-sample MAE={mae_out:.4f}, RMSE={rmse_out:.4f}")

#RESULTATEN
fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
jaren = [2021, 2022]
for ax, iso in zip(axes.flatten(), lijst_landen):
    echte = test_lang[(test_lang['ISO3']==iso) & (test_lang['Jaar'].dt.year.isin(jaren))]
    voorsp = resultaat_voorspelling[resultaat_voorspelling['ISO3']==iso]
    ax.plot(echte['Jaar'].dt.year, echte['kwetsbaarheid'], 'o-', label='Echt')
    ax.plot(voorsp['Jaar'].dt.year, voorsp['kwetsbaarheid_voorspelling'], 'x--', label='Voorspelling')
    ax.set_title(iso)
    ax.set_xticks(jaren)
    ax.legend()
plt.tight_layout()
plt.show()

#TABLE
latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
Model & AIC & BIC & MAE (2021--2022) & RMSE (2021--2022) \\\
\\midrule
Bayesiaanse ARDL(0,1) met shrinkage & {aic:.2f} & {bic:.2f} & {mae_out:.4f} & {rmse_out:.4f} \\\
\\bottomrule
\\end{{tabular}}
\\caption{{ARDL(0,1) met Bayesian shrinkage: In-sample (AIC, BIC) en out-of-sample forecast errors (2021--2022)}}
\\label{{tab:ardl01_bayes_shrinkage_beta}}
\\end{{table}}
"""
print(latex_table)
