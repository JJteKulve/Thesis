import os
import pandas as pd

#bestanden
PAD_INPUT = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\junk\DATA\bestanden\8_landen_logit.xlsx"
UITVOERMAP = r"\\ftfilesrv\shares\home\jkulve\Desktop\Thesis\junk\DATA\bestanden"
os.makedirs(UITVOERMAP, exist_ok=True)

#
def laad_dataset(pad: str) -> pd.DataFrame:
    return pd.read_excel(pad)


def splits_jaarkolommen(df: pd.DataFrame, start_train: int, einde_train: int,
                       start_test: int, einde_test: int) -> tuple:
    jaar_kolommen = [kol for kol in df.columns
                     if any(kol.endswith(str(j)) for j in range(start_train, einde_test + 1))]

    #TRAIN EN TEST
    train_jaren = [str(j) for j in range(start_train, einde_train + 1)]
    test_jaren = [str(j) for j in range(start_test, einde_test + 1)]
    train_kolommen = [kol for kol in jaar_kolommen if any(kol.endswith(jjr) for jjr in train_jaren)]
    test_kolommen = [kol for kol in jaar_kolommen if any(kol.endswith(jjr) for jjr in test_jaren)]

    #voeg is3 en name toe
    id_kolommen = ['ISO3', 'Name']
    train_df = df[id_kolommen + train_kolommen].copy()
    test_df = df[id_kolommen + test_kolommen].copy()
    return train_df, test_df


def sla_op(df: pd.DataFrame, bestandsnaam: str) -> None:
    pad = os.path.join(UITVOERMAP, bestandsnaam)
    df.to_excel(pad, index=False)
    print(f"Bestand opgeslagen: {pad}")


def main():
    #laad alle data
    df_volledig = laad_dataset(PAD_INPUT)

    #splits in traing en data
    train_df, test_df = splits_jaarkolommen(df_volledig,
                                           start_train=1995,
                                           einde_train=2020,
                                           start_test=2021,
                                           einde_test=2022)

    #opslaan
    sla_op(train_df, '8_landen_logit_train_1995_2020.xlsx')
    sla_op(test_df, '8_landen_logit_test_2021_2022.xlsx')


if __name__ == '__main__':
    main()
