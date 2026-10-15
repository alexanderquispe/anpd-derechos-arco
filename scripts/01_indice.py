"""Índice de la colección "Decisiones sobre derechos ARCO" de la ANPD (gob.pe).

Recorre todas las páginas de la colección principal y de la subcolección
"Concluido 2026" y guarda un CSV por colección con: fecha de publicación,
título, URL de detalle y resumen corto.

Uso:  .venv/bin/python -I scripts/01_indice.py
"""
from pathlib import Path
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim"
BASE = "https://www.gob.pe"
COLECCIONES = {
    "arco": "/institucion/anpd/colecciones/1749-decisiones-sobre-derechos-arco",
    "concluido_2026": "/institucion/anpd/colecciones/96062-concluido-2026",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (investigacion academica PUCP)"}
MESES = {m: i for i, m in enumerate(
    ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
     "septiembre", "octubre", "noviembre", "diciembre"], start=1)}
MESES["setiembre"] = 9


def fecha_iso(txt):
    m = re.search(r"(\d{1,2}) de (\w+) de (\d{4})", txt)
    if not m:
        return None
    d, mes, a = m.groups()
    return f"{a}-{MESES[mes.lower()]:02d}-{int(d):02d}"


def parsear_pagina(html):
    soup = BeautifulSoup(html, "lxml")
    filas = []
    for card in soup.select("div.shadow-campaign-card"):
        a = card.select_one("a[href*='/normas-legales/']")
        if a is None:
            continue
        fecha = card.select_one("div.border-b")
        resumen = a.next_sibling
        filas.append({
            "fecha_pub_txt": fecha.get_text(strip=True) if fecha else None,
            "titulo": a.get_text(strip=True),
            "url": BASE + a["href"],
            "id_norma": int(re.search(r"/normas-legales/(\d+)", a["href"]).group(1)),
            "resumen_corto": resumen.strip() if isinstance(resumen, str) else None,
        })
    paginas = [int(x) for x in re.findall(r"sheet=(\d+)", html)]
    return filas, max(paginas, default=1)


def recorrer(ruta):
    s = requests.Session()
    s.headers.update(HEADERS)
    filas, total, p = [], 1, 1
    while p <= total:
        r = s.get(BASE + ruta, params={"sheet": p}, timeout=60)
        r.raise_for_status()
        f, ultima = parsear_pagina(r.text)
        total = max(total, ultima)
        for x in f:
            x["pagina"] = p
        filas += f
        print(f"  página {p}/{total}: {len(f)} ítems")
        p += 1
        time.sleep(0.4)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for nombre, ruta in COLECCIONES.items():
        print(f"Colección {nombre}")
        df = recorrer(ruta)
        df["fecha_pub"] = pd.to_datetime(df["fecha_pub_txt"].map(fecha_iso))
        df = df.drop_duplicates("id_norma")
        df.to_csv(OUT / f"indice_{nombre}.csv", index=False)
        print(f"  -> {len(df)} decisiones; rango {df.fecha_pub.min():%Y-%m-%d} a {df.fecha_pub.max():%Y-%m-%d}")
        print(df["fecha_pub"].dt.year.value_counts().sort_index().to_string())
