import os
import pandas as pd
import numpy as np

#INLADEN
BASISMAP = r"\\ftfilesrv\shares\home\jkulve\Desktop\JJ\DATA_thesis\vulnerability"
UITVOERMAP = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\junk\DATA\bestanden"
os.makedirs(UITVOERMAP, exist_ok=True)

PAD_ALLE_LANDEN_CSV = os.path.join(UITVOERMAP, "TEST_alle_landen_logit.csv")
PAD_8_LANDEN_XLSX = os.path.join(UITVOERMAP, "TEST_8_landen_logit.xlsx")

#VARIBALES
VARIABELEN_LIJST = [
    "ecosystems",
    "food",
    "health",
    "infrastructure",
    "habitat",
    "water",
    "vulnerability",
]
AFKORTINGEN = {
    "ecosystems":     "eco",
    "food":           "food",
    "health":         "health",
    "infrastructure": "infra",
    "habitat":        "habi",
    "water":          "water",
    "vulnerability":  "vulner",
}

#GRENZEN STELLEN VOOR LOGIT
EPSILON_ONDER, EPSILON_BOVEN = 1e-6, 0.999999999

#JAAR ALS TEKST
JAAR_KOLOMMEN = [str(j) for j in range(1995, 2023)]

def laad_en_transformeer(variabele: str) -> pd.DataFrame:
    bestandspad = os.path.join(BASISMAP, f"{variabele}.csv")
    df = pd.read_csv(bestandspad)

    #VERWIJDER NIET VOLLEDIGE DATA
    df = df.dropna(subset=["ISO3", "Name"])

    #JAARKOLOMMEN NAAR NUMERIEK
    df[JAAR_KOLOMMEN] = df[JAAR_KOLOMMEN].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=JAAR_KOLOMMEN)

    #JAARKOLOMMEN TOT EPSILON ONDER EN BOVEN
    df[JAAR_KOLOMMEN] = df[JAAR_KOLOMMEN].clip(lower=EPSILON_ONDER,
                                                upper=EPSILON_BOVEN)

    #LOGIT ZELF
    df[JAAR_KOLOMMEN] = np.log(df[JAAR_KOLOMMEN] /
                               (1 - df[JAAR_KOLOMMEN]))

    return df.reset_index(drop=True)

def filter_gemeenschappelijke_landen(gegevens: dict, variabelen: list) -> None:
    gemeenschappelijk = set(gegevens[variabelen[0]]["ISO3"])
    for var in variabelen[1:]:
        gemeenschappelijk &= set(gegevens[var]["ISO3"])

    for var in variabelen:
        df = gegevens[var]
        gegevens[var] = df[df["ISO3"].isin(gemeenschappelijk)].reset_index(drop=True)

#COMBINEER GEGEVEN
def combineer_gegevens(gegevens: dict,
                       variabelen: list,
                       afkortingen: dict) -> pd.DataFrame:
    eerste_var = variabelen[0]
    df_start = gegevens[eerste_var].copy()
    df_start = df_start.rename(columns={
        jaar: f"{afkortingen[eerste_var]}_{jaar}" for jaar in JAAR_KOLOMMEN
    })
    gecombineerd = df_start

    for var in variabelen[1:]:
        afk = afkortingen[var]
        df_var = gegevens[var].copy()
        df_var = df_var.rename(columns={
            jaar: f"{afk}_{jaar}" for jaar in JAAR_KOLOMMEN
        })
        benodigde_kolommen = ["ISO3", "Name"] + [f"{afk}_{jaar}" for jaar in JAAR_KOLOMMEN]
        gecombineerd = gecombineerd.merge(df_var[benodigde_kolommen],
                                          on=["ISO3", "Name"],
                                          how="inner")
    return gecombineerd

def main():
    #LAAD EN BEWERK ALLE VARIABELS
    gegevens = {var: laad_en_transformeer(var) for var in VARIABELEN_LIJST}

    #ZOEK NAAR ONZE 8 LANDEN
    filter_gemeenschappelijke_landen(gegevens, VARIABELEN_LIJST)

    #ALLES NAAR 1
    df_gecombineerd = combineer_gegevens(gegevens, VARIABELEN_LIJST, AFKORTINGEN)

    #OPSLAAN
    df_gecombineerd.to_csv(PAD_ALLE_LANDEN_CSV, index=False)
    print(f"Alle landen opgeslagen naar:\n  {PAD_ALLE_LANDEN_CSV}")

    #MAAK SUBSET AAN
    specifieke_landen = ["NLD", "FRA", "ITA", "SGP", "USA", "GBR", "ROU", "CHN"]
    subset8 = df_gecombineerd[df_gecombineerd["ISO3"].isin(specifieke_landen)].reset_index(drop=True)
    subset8.to_excel(PAD_8_LANDEN_XLSX, index=False)
    print(f"Subset van 8 landen opgeslagen naar:\n  {PAD_8_LANDEN_XLSX}")


if __name__ == "__main__":
    main()
