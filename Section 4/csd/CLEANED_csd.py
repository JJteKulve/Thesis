import pandas as pd
import numpy as np
from scipy.stats import norm

#over maar 8 landen
PAD_INVOER = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\final\DATA voorbereiden 0\8_landen_logit.xlsx"
#vulnerability
VULN_PREFIX = "vulner_"


def laad_vulnerability_data(pad: str) -> pd.DataFrame:
    df = pd.read_excel(pad)
    vuln_cols = [col for col in df.columns if col.startswith(VULN_PREFIX)]
    df_vuln = df[['ISO3'] + vuln_cols].dropna()
    mat = df_vuln.set_index('ISO3').T
    #demeanen
    mat_demeaned = mat - mat.mean()
    return mat_demeaned

#test zelf
def bereken_pesaran_cd(data: pd.DataFrame) -> tuple:
    N = data.shape[1]  
    #gezamelijke correelaties
    rho_waarden = []
    for i in range(N):
        for j in range(i + 1, N):
            xi = data.iloc[:, i]
            xj = data.iloc[:, j]
            rho = np.corrcoef(xi, xj)[0, 1]
            rho_waarden.append(rho)
    #pesaran statistiek
    factor = np.sqrt(2 / (N * (N - 1)))
    CD_stat = factor * np.sum(rho_waarden)
    #p value
    p_value = 2 * (1 - norm.cdf(abs(CD_stat)))
    return CD_stat, p_value


if __name__ == '__main__':
    #laden
    data_panel = laad_vulnerability_data(PAD_INVOER)

    #test
    CD, p = bereken_pesaran_cd(data_panel)

    #results
    print(f"Pesaran CD-statistiek: {CD:.4f}")
    print(f"P-waarde: {p:.4f}")
    if p < 0.05:
        print("⇒ Significante cross-sectionele afhankelijkheid tussen landen.")
    else:
        print("⇒ Geen significant bewijs voor cross-sectionele afhankelijkheid.")
