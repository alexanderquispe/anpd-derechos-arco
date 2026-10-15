"""Descarga la ficha de detalle y los PDFs de cada decisión seleccionada.

Grupos:
  2026            decisiones ARCO publicadas en 2026 (colección 1749)
  medios_historico decisiones ARCO de años previos contra medios de prensa,
                   buscadores o redes sociales (filtro por nombre del reclamado)

Uso:  .venv/bin/python -I scripts/02_descargar.py 2026
"""
from pathlib import Path
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PDFS = ROOT / "data" / "raw" / "pdfs"
HEADERS = {"User-Agent": "Mozilla/5.0 (investigacion academica PUCP)"}

# Reclamados que son medios, buscadores o plataformas donde se difunde contenido periodístico
PATRON_MEDIOS = (
    r"el comercio|rep[uú]blica publicaciones|grupo la rep[uú]blica|diario|radiodifusi[oó]n|\batv\b|latina|"
    r"am[eé]rica tv|panamericana|\brpp\b|willax|per[uú] ?21|chirimoya|prensmart|prensa popular|"
    r"editora|epensa|peri[oó]dic|televisi|radio |canal |infobae|trome|correo|expreso|caretas|"
    r"exitosa|google|youtube|facebook|meta platforms|tiktok|twitter|microsoft|yahoo|wikipedia"
)


def seleccionar(grupo):
    idx = pd.read_csv(INTERIM / "indice_arco.csv", parse_dates=["fecha_pub"])
    if grupo == "2026":
        return idx[idx.fecha_pub.dt.year == 2026]
    if grupo == "medios_historico":
        reclamado = idx.titulo.str.split(" - ", n=1).str[1].fillna(idx.titulo)
        es_medio = reclamado.str.contains(PATRON_MEDIOS, case=False, regex=True)
        return idx[(idx.fecha_pub.dt.year < 2026) & es_medio]
    raise ValueError(grupo)


def campo(desc, etiqueta):
    """Texto que sigue a una etiqueta en negrita dentro de la ficha."""
    for strong in desc.find_all("strong"):
        if strong.get_text(strip=True).rstrip(":").lower().startswith(etiqueta.lower()):
            partes = []
            for sib in strong.next_siblings:
                if getattr(sib, "name", None) == "strong":
                    break
                partes.append(sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib))
            return re.sub(r"\s+", " ", " ".join(partes)).strip(" :") or None
    return None


def ficha(html):
    soup = BeautifulSoup(html, "lxml")
    desc = soup.select_one("div.description")
    texto = re.sub(r"\s+", " ", desc.get_text(" ", strip=True)) if desc else ""
    m_res = re.search(r"Resoluci[oó]n N[°º.]?\s*([\w\-/ ]+?) de (\d{1,2} de \w+ de \d{4})", texto)
    m_dec = re.search(r"Decisi[oó]n:\s*(.+?)(?:Esta norma pertenece|$)", texto)
    pdfs = sorted({a["href"] for a in soup.select("a[href*='cdn.www.gob.pe/uploads/document/file']")
                   if ".pdf" in a["href"].lower()})
    return {
        "reclamado": campo(desc, "Reclamado") if desc else None,
        "derechos_arco": campo(desc, "Derechos ARCO") if desc else None,
        "tratamiento_cuestionado": campo(desc, "Tratamiento Cuestionado") if desc else None,
        "resolucion_num": m_res.group(1).strip() if m_res else None,
        "resolucion_fecha_txt": m_res.group(2) if m_res else None,
        "decision_ficha": m_dec.group(1).strip() if m_dec else None,
        "ficha_texto": texto,
        "pdf_urls": " | ".join(pdfs),
    }


def main(grupo):
    PDFS.mkdir(parents=True, exist_ok=True)
    sel = seleccionar(grupo)
    print(f"{grupo}: {len(sel)} decisiones")
    s = requests.Session()
    s.headers.update(HEADERS)
    filas = []
    for i, r in enumerate(sel.itertuples(), 1):
        det = ficha(s.get(r.url, timeout=60).text)
        archivos = []
        for k, u in enumerate(filter(None, det["pdf_urls"].split(" | "))):
            nombre = f"{r.id_norma}_{k}_{Path(u.split('?')[0]).name}"
            destino = PDFS / nombre
            if not destino.exists():
                resp = s.get(u, timeout=120)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    print(f"   ! {u} no es PDF")
                    continue
                destino.write_bytes(resp.content)
                time.sleep(0.3)
            archivos.append(nombre)
        filas.append({**r._asdict(), **det, "pdfs": " | ".join(archivos), "n_pdfs": len(archivos)})
        print(f"  [{i}/{len(sel)}] {r.titulo[:90]} -> {len(archivos)} PDF")
        time.sleep(0.3)
    df = pd.DataFrame(filas).drop(columns=["Index"])
    df.to_csv(INTERIM / f"detalle_{grupo}.csv", index=False)
    print(f"Guardado detalle_{grupo}.csv; PDFs totales: {df.n_pdfs.sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026")
