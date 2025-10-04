#!/usr/bin/env python3
"""
Unisci train.csv, val.csv e test.csv in un unico CSV finale.

⚙️ Da configurare all'inizio del file:
  TRAIN_PATH = "path/al/train.csv"
  VAL_PATH   = "path/al/val.csv"
  TEST_PATH  = "path/al/test.csv"
  OUT_PATH   = "path/output/unified.csv"

Le colonne restano esattamente le stesse dei file originali
(usati da Themis), e non si perde alcuna informazione.
"""

import pandas as pd

# --- SPECIFICA QUI I PATH ---
TRAIN_PATH = "../StatoDellArte/Projects/Is-It-Fake-Or-Not/Recovery/train.csv"
VAL_PATH   = "../StatoDellArte/Projects/Is-It-Fake-Or-Not/Recovery/val.csv"
TEST_PATH  = "../StatoDellArte/Projects/Is-It-Fake-Or-Not/Recovery/test.csv"
OUT_PATH   = "Data/ReCOVery/recovery-unified.csv"
# ----------------------------

def main():
    print("[INFO] Caricamento train...")
    df_train = pd.read_csv(TRAIN_PATH)
    print(f"  -> {len(df_train)} righe")

    print("[INFO] Caricamento val...")
    df_val = pd.read_csv(VAL_PATH)
    print(f"  -> {len(df_val)} righe")

    print("[INFO] Caricamento test...")
    df_test = pd.read_csv(TEST_PATH)
    print(f"  -> {len(df_test)} righe")

    # Controllo che abbiano le stesse colonne
    cols_train, cols_val, cols_test = set(df_train.columns), set(df_val.columns), set(df_test.columns)
    if not (cols_train == cols_val == cols_test):
        raise ValueError(f"Colonne diverse tra i dataset:\n"
                         f"train={cols_train}\nval={cols_val}\ntest={cols_test}")

    # Concatena i tre dataset
    df_all = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    print(f"[INFO] Totale righe unite: {len(df_all)}")
    print(f"[INFO] Colonne: {df_all.columns.tolist()}")

    # Salva output
    df_all.to_csv(OUT_PATH, index=False)
    print(f"[OK] Dataset unificato salvato in {OUT_PATH}")

if __name__ == "__main__":
    main()
