"""Arma el corpus a nivel documento desde el OCR, mide la calidad del OCR y
marca los casos relevantes para la consulta:

  ¿Ha resuelto la ANPD reclamaciones ARCO (cancelación / oposición /
  rectificación) para retirar o corregir datos personales difundidos en
  notas periodísticas o reportajes que vinculan a una persona con un delito?

Salidas (data/interim/ y outputs/):
  corpus_<grupo>.parquet    texto OCR completo por decisión + metadatos
  calidad_ocr_<grupo>.csv   similitud OCR vs. texto nativo por página
  casos_<grupo>.csv         banderas temáticas y fallo extraído por decisión

Uso:  .venv/bin/python -I scripts/04_corpus.py 2026 [medios_historico]
"""
from collections import Counter
from pathlib import Path
import json
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9ñ ]", " ", s)).strip()


# Cada bandera es un conjunto de patrones sobre el texto normalizado (sin tildes)
BANDERAS = {
    "prensa": r"nota(s)? periodistic|reportaje|articulo periodistic|medio(s)? de comunicacion|"
              r"publicacion(es)? periodistic|noticia|diario |portal web|plataforma(s)? digital(es)? de",
    "buscador": r"motor(es)? de busqueda|buscador|desindex|resultados de busqueda|url",
    "delito_penal": r"delito|penal|fiscal|denuncia|investigacion preliminar|proceso judicial|"
                    r"acusacion|imputa|carpeta fiscal|sentencia",
    "absuelto_archivado": r"absuel|absolu|archiv|sobresei|inocen|no ha sido condenad|"
                          r"rehabilita|sin responsabilidad",
    "inexactitud": r"inexact|falso|falsa|erroneo|erronea|incorrect|no es cierto|difamat|"
                   r"calumni|veracidad|desactualiz",
    "derecho_olvido": r"derecho al olvido|olvido",
    "libertad_expresion": r"libertad de expresion|libertad de informacion|libertad de prensa|"
                          r"interes publico|relevancia publica|articulo 2 inciso 4|"
                          r"fuentes accesibles|finalidad periodistic|periodism",
    "rectificacion": r"derecho de rectificacion|rectificar|rectifique",
    "cancelacion_oposicion": r"derecho de cancelacion|derecho de oposicion|supresion",
}

SENTIDOS = [
    ("fundada en parte", r"fundad[ao] en parte|fundad[ao] parcialmente"),
    ("infundada", r"infundad[ao]"),
    ("improcedente", r"improcedente"),
    ("fundada", r"fundad[ao]"),
    ("concluido / sustraccion", r"sustraccion de la materia|conclusion del procedimiento"),
]


PIE_PAGINA = re.compile(
    r"[“\"]?Esta es una copia aut[eé]ntica imprimible.*?(?:seg[uú]n corresponda\.?[”\"]?|P[aá]gina \d+ de \d+)",
    re.S)


def limpiar(texto):
    """Quita el pie de autenticación del MINJUS que se repite en cada página."""
    texto = PIE_PAGINA.sub("", texto)
    return re.sub(r"\n?P[aá]gina \d+ de \d+", "", texto)


def leer_ocr(nombre_pdf):
    ruta = INTERIM / "ocr" / (Path(nombre_pdf).stem + ".jsonl")
    if not ruta.exists():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]


def cobertura(ocr, nativo):
    """Recall por bolsa de palabras: share de palabras del texto nativo que el OCR recupera.
    No depende del orden de lectura (el texto nativo trae pies de página en otro orden)."""
    o, t = Counter(norm(ocr).split()), Counter(norm(nativo).split())
    if sum(t.values()) < 20:
        return np.nan  # página escaneada sin capa de texto
    return sum((o & t).values()) / sum(t.values())


def parte_resolutiva(texto):
    """Texto desde 'SE RESUELVE' (o equivalente) hasta el final."""
    t = norm(texto)
    m = None
    for pat in [r"se resuelve", r"resuelve primero", r"resuelve"]:
        ms = list(re.finditer(pat, t))
        if ms:
            m = ms[-1]
            break
    return t[m.start():] if m else ""


def sentido_fallo(texto):
    res = parte_resolutiva(texto)[:1500]
    for etiqueta, pat in SENTIDOS:
        if re.search(pat, res):
            return etiqueta
    return None


def main(grupos):
    for g in grupos:
        det = pd.read_csv(INTERIM / f"detalle_{g}.csv")
        docs, paginas = [], []
        for r in det.itertuples():
            pags = [p for n in str(r.pdfs).split(" | ") if n and n != "nan" for p in leer_ocr(n)]
            for p in pags:
                paginas.append({"id_norma": r.id_norma, "pdf": p["pdf"], "pagina": p["pagina"],
                                "conf_media": p["conf_media"], "n_lineas": p["n_lineas"],
                                "chars_ocr": len(p["texto_ocr"]), "chars_nativo": len(p["texto_nativo"]),
                                "cobertura_vs_nativo": cobertura(p["texto_ocr"], p["texto_nativo"])})
            texto = "\n\n".join(f"[pág. {p['pagina']}]\n{limpiar(p['texto_ocr'])}" for p in pags)
            docs.append({"id_norma": r.id_norma, "texto_ocr": texto, "n_paginas": len(pags)})
        corpus = det.merge(pd.DataFrame(docs), on="id_norma")
        corpus["grupo"] = g
        corpus.to_parquet(INTERIM / f"corpus_{g}.parquet", index=False)
        dir_txt = INTERIM / f"textos_{g}"
        dir_txt.mkdir(exist_ok=True)
        for r in corpus.itertuples():
            (dir_txt / f"{r.id_norma}.txt").write_text(f"{r.titulo}\n{r.url}\n\n{r.texto_ocr}", encoding="utf-8")

        cal = pd.DataFrame(paginas)
        cal.to_csv(OUTPUTS / f"calidad_ocr_{g}.csv", index=False)
        print(f"\n== {g}: {len(corpus)} decisiones, {len(cal)} páginas OCR ==")
        if len(cal):
            print(f"Confianza media PaddleOCR: {cal.conf_media.mean():.3f} "
                  f"(p10 {cal.conf_media.quantile(.1):.3f})")
            print(f"Cobertura de palabras OCR vs texto nativo: mediana {cal.cobertura_vs_nativo.median():.3f}, "
                  f"p10 {cal.cobertura_vs_nativo.quantile(.1):.3f}; "
                  f"páginas escaneadas sin texto nativo (solo OCR): {cal.cobertura_vs_nativo.isna().sum()}")

        tn = corpus.texto_ocr.map(norm)
        for b, pat in BANDERAS.items():
            pat = re.sub(r"\((?!\?)", "(?:", pat)  # grupos no capturantes
            corpus[f"f_{b}"] = tn.str.contains(pat, regex=True)
            corpus[f"n_{b}"] = tn.str.count(pat)
        corpus["sentido_fallo_ocr"] = corpus.texto_ocr.map(sentido_fallo)
        corpus["exp_ocr"] = corpus.texto_ocr.str.extract(
            r"(?i)Expediente\s*N?\s*[°º.]*\s*:?\s*([0-9]{2,4}\s*-\s*20[0-9]{2}\s*-?\s*PTT)", expand=False)
        # Puntaje de relevancia: prensa/buscador + vínculo penal + (inexactitud u olvido o absolución)
        corpus["relevancia"] = (
            2 * (corpus.f_prensa | corpus.f_buscador).astype(int)
            + 2 * corpus.f_delito_penal.astype(int)
            + corpus.f_absuelto_archivado.astype(int)
            + corpus.f_inexactitud.astype(int)
            + corpus.f_derecho_olvido.astype(int)
            + corpus.f_libertad_expresion.astype(int)
        )
        cols = (["id_norma", "fecha_pub", "titulo", "reclamado", "derechos_arco", "resolucion_num",
                 "decision_ficha", "sentido_fallo_ocr", "exp_ocr", "n_paginas", "relevancia"]
                + [c for c in corpus if c.startswith("f_")] + ["url", "pdfs"])
        casos = corpus[cols].sort_values(["relevancia", "fecha_pub"], ascending=False)
        casos.to_csv(OUTPUTS / f"casos_{g}.csv", index=False)
        pd.set_option("display.width", 250, "display.max_colwidth", 60)
        print(casos[["fecha_pub", "reclamado", "derechos_arco", "sentido_fallo_ocr", "relevancia"]]
              .head(25).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["2026"])
