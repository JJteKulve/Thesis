import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

#bestand
PAD_INVOER = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit.xlsx"

#variablen weer
AFKORTINGEN = {
    "vulnerability":  "vulner",
    "ecosystems":     "eco",
    "food":           "food",
    "health":         "health",
    "infrastructure": "infra",
    "habitat":        "habi",
    "water":          "water",
}
VARIABELENLIJST = list(AFKORTINGEN.keys())

#omzetten naar long format
def laad_en_smelt(pad: str) -> pd.DataFrame:
    df = pd.read_excel(pad)
    df_long = pd.wide_to_long(
        df,
        stubnames=list(AFKORTINGEN.values()),
        i=["ISO3", "Name"],
        j="Year",
        sep="_",
        suffix=r"\d{4}"
    ).reset_index()
    df_long["Year"] = df_long["Year"].astype(int)
    terug_naar_naam = {afk: var for var, afk in AFKORTINGEN.items()}
    return df_long.rename(columns=terug_naar_naam)

#harris test
def harris_test(df: pd.DataFrame,
                variabele: str,
                met_trend: bool = False,
                n_simulaties: int = 5000) -> tuple:

    data = df[["ISO3", "Year", variabele]].copy()
    data[variabele] = data[variabele].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[variabele])

    data["y_lag"] = data.groupby("ISO3")[variabele].shift(1)
    data["dy"]    = data.groupby("ISO3")[variabele].diff()

    if met_trend:
        data["trend"] = data["Year"] - data["Year"].min()

    X_lijst, y_lijst = [], []
    onafhankelijke = ["y_lag"] + (["trend"] if met_trend else [])
    for _, groep in data.groupby("ISO3"):
        sub = groep.dropna(subset=onafhankelijke + ["dy"])
        if sub.empty:
            continue
        X_i = sm.add_constant(sub[onafhankelijke])
        y_i = sub["dy"]
        X_lijst.append(X_i)
        y_lijst.append(y_i)

    X = pd.concat(X_lijst, axis=0)
    y = pd.concat(y_lijst, axis=0)
    valid_mask = X.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1) & y.notnull()
    X, y = X.loc[valid_mask], y.loc[valid_mask]
    if X.empty:
        raise ValueError(f"Geen data over voor {variabele}, trend={met_trend}")

    resultaat = sm.OLS(y, X).fit()
    rho_hat = resultaat.params["y_lag"]
    N = df["ISO3"].nunique()
    Z_stat = np.sqrt(N) * (rho_hat - 1)
    p_waarde = 2 * (1 - stats.norm.cdf(abs(Z_stat)))

    T = df["Year"].nunique()
    sim_gem = []
    for _ in range(n_simulaties):
        eps = np.random.normal(size=(N, T))
        ysim = np.cumsum(eps, axis=1)
        rhos_i = []
        for i in range(N):
            yi = ysim[i, :]
            dyi = np.diff(yi)
            if met_trend:
                t = np.arange(len(dyi))
                Xi = sm.add_constant(np.column_stack([yi[:-1], t]))
            else:
                Xi = sm.add_constant(yi[:-1])
            rhos_i.append(sm.OLS(dyi, Xi).fit().params[1])
        sim_gem.append(np.mean(rhos_i))
    sim_mean = np.mean(sim_gem)
    sim_stdv = np.std(sim_gem, ddof=1)

    return rho_hat, Z_stat, p_waarde, sim_mean, sim_stdv


if __name__ == '__main__':
    df_panel = laad_en_smelt(PAD_INVOER)

    print("\n=== Harris (1999) paneltest voor 8 landen ===")
    for var in VARIABELENLIJST:
        try:
            rho, Z, p, sim_mu, sim_sigma = harris_test(df_panel, var, met_trend=False)
            rho_t, Z_t, p_t, mu_t, sigma_t = harris_test(df_panel, var, met_trend=True)
            print(f"{var:12} level  → Z={Z:7.4f}, p={p:7.4f}, ρ̂={rho:7.4f}, simμ={sim_mu:7.4f}, σ={sim_sigma:7.4f}")
            print(f"{var:12} trend → Z={Z_t:7.4f}, p={p_t:7.4f}, ρ̂={rho_t:7.4f}, simμ={mu_t:7.4f}, σ={sigma_t:7.4f}\n")
        except ValueError as err:
            print(f"{var:12} ERROR: {err}")