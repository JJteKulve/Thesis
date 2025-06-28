import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# hbestand
PAD_INVOER = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit.xlsx"

#omzetten
CATEGORIEN = [
    "ecosystems",
    "food",
    "health",
    "infrastructure",
    "habitat",
    "water",
]
AFKORTINGEN = {
    "ecosystems":     "eco",
    "food":           "food",
    "health":         "health",
    "infrastructure": "infra",
    "habitat":        "habi",
    "water":          "water",
}

#8landen
LANDENLIJST = ["NLD", "GBR", "USA", "ROU", "ITA", "FRA", "CHN", "SGP"]

#hele jaartallen
JAAR_KOLOMMEN = [str(j) for j in range(1995, 2023)]


def lees_data(pad: str) -> pd.DataFrame:
    return pd.read_excel(pad)

#Weer naar long
def transformeer_naar_lang(df: pd.DataFrame, categorie: str) -> pd.DataFrame:
    afk = AFKORTINGEN[categorie]
    kolommen = [f"{afk}_{jaar}" for jaar in JAAR_KOLOMMEN]
    df_kort = df[["ISO3", "Name"] + kolommen]
    df_lang = (
        df_kort
        .melt(
            id_vars=["ISO3", "Name"],
            value_vars=kolommen,
            var_name="jaar",
            value_name=f"waarde_{categorie}"
        )
    )
    #jaartal eraf
    df_lang['jaar'] = df_lang['jaar'].str.replace(f"{afk}_", "", regex=False)
    df_lang['jaar'] = pd.to_datetime(df_lang['jaar'].astype(int), format="%Y")
    return df_lang

#PLOT MAKEN
def maak_plots(data_per_categorie: dict):
    sns.set_style("whitegrid")
    sns.set_context("paper")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    axes = axes.flatten()
    markeringen = ['o', 's', '^', 'v', 'D', 'x']

    for ax, land in zip(axes, LANDENLIJST):
        for idx, categorie in enumerate(CATEGORIEN):
            df_cat = data_per_categorie[categorie]
            df_land = df_cat[df_cat['ISO3'] == land].sort_values('jaar')
            tijdreeks = df_land[f"waarde_{categorie}"].dropna()
            if tijdreeks.empty:
                continue
            ax.plot(
                df_land['jaar'], tijdreeks,
                color='black', linestyle='-', linewidth=0.8,
                marker=markeringen[idx], markersize=4, alpha=0.5
            )
        ax.set_title(land, fontsize=11)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

    for lege_ax in axes[len(LANDENLIJST):]:
        fig.delaxes(lege_ax)

    #1 legenda
    legend_handles = [
        Line2D([0], [0], color='black', linestyle='-', marker=markeringen[i],
               markersize=6, linewidth=0.8, alpha=0.8,
               label=CATEGORIEN[i].capitalize())
        for i in range(len(CATEGORIEN))
    ]
    fig.subplots_adjust(bottom=0.05)
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(CATEGORIEN), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def print_summary_latex(df: pd.DataFrame):
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small % Maak de tekst wat kleiner")
    print(r"\renewcommand{\arraystretch}{0.85} % Verticaal compacter")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\toprule")
    kop = "Land / Stat & Food & Water & Health & Ecosystems & Infrastructure & Habitat & Vulnerability \\\""
    print(kop)
    print(r"\midrule")

    for land in LANDENLIJST:
        print(rf"\multicolumn{{8}}{{l}}{{\textbf{{{land}}}}} \\")
        for stat_naam, func in [("gemiddelde", pd.Series.mean),
                                 ("sd", pd.Series.std),
                                 ("min", pd.Series.min),
                                 ("max", pd.Series.max)]:
            waarden = []
            for categorie in CATEGORIEN + ["vulnerability"]:
                afk = AFKORTINGEN.get(categorie, "vulner")
                kolommen = [f"{afk}_{jaar}" for jaar in JAAR_KOLOMMEN]
                serie = df.loc[df['ISO3'] == land, kolommen].iloc[0]
                waarden.append(f"{func(serie):.2f}")
            regel = " & ".join([stat_naam] + waarden) + r" \\\""
            print(regel)
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\caption{Summary statistieken per land en categorie.}")
    print(r"\label{summary_statistics}")
    print(r"\end{table}")


if __name__ == '__main__':
    #data
    df_breed = lees_data(PAD_INVOER)

    #omzetten
    data_per_cat = {cat: transformeer_naar_lang(df_breed, cat)
                    for cat in CATEGORIEN}

    #plotten
    maak_plots(data_per_cat)

    #sum in latex
    print_summary_latex(df_breed)
