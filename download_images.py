import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ==== PARAMETRI DA MODIFICARE ====
CSV_PATH = "Data/ReCOVery/recovery-news-data.csv"   # CSV con colonne id,title,text,label
OUTPUT_DIR = "Data/ReCOVery/images"    # cartella per salvare le immagini
TIMEOUT = 100000                            # timeout per il download
# =================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carica dataset
df = pd.read_csv(CSV_PATH)

print(f"Dataset: {len(df)} righe trovate")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
    # appena leggo .jpg oppure .png oppure .jpeg in una qualsiasi riga in una qualsiasi colonna, arrivo fino a inizio elemento dove c'è https e salvo l'immagine con il nome dell'id della news
    img_url = None
    for col in ['title', 'body_text']:
        if isinstance(row[col], str):
            for ext in ['.jpg', '.jpeg', '.png']:
                ext_pos = row[col].lower().find(ext)
                if ext_pos != -1:
                    start_pos = row[col].rfind('https://', 0, ext_pos)
                    if start_pos != -1:
                        img_url = row[col][start_pos:ext_pos + len(ext)]
                        break
        if img_url:
            break # esco dal ciclo se ho trovato l'URL
    if not img_url:
        continue # nessuna immagine trovata
    try:
        response = requests.get(img_url, timeout=TIMEOUT)
        response.raise_for_status()  # controlla che la richiesta sia andata a buon fine
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_save_path = os.path.join(OUTPUT_DIR, f"{row[0]}.jpg")
        img.save(img_save_path)
    except Exception:
        continue # ignoro errori di download/salvataggio