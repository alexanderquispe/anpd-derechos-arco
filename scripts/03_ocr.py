"""OCR de los PDFs con PaddleOCR (español) acelerado en la GPU de Apple (MPS).

PaddlePaddle no tiene backend para Metal, así que se usa el motor
"transformers" de PaddleOCR 3.7: los mismos modelos PP-OCRv6_medium (det + rec)
en PyTorch, movidos a `mps`. Si MPS no está disponible, cae a CPU.

Cada página se rasteriza a 200 dpi con PyMuPDF. Se guarda un JSONL por PDF en
data/interim/ocr/ con, por página: texto OCR, confianza media de
reconocimiento y texto nativo del PDF (si existe) para control de calidad.

Uso:  .venv/bin/python -I scripts/03_ocr.py 2026 medios_historico [--shard i/n]
      (varios procesos con distinto --shard comparten la GPU y se reparten los PDFs)
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import pymupdf
import torch

ROOT = Path(__file__).resolve().parents[1]
PDFS = ROOT / "data" / "raw" / "pdfs"
OCR_DIR = ROOT / "data" / "interim" / "ocr"
DPI = 200


def crear_motor():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="es", engine="transformers",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"
    # PaddleX valida `device` contra su lista (sin "mps"); se mueven los modelos a mano.
    # Los predictores envían las entradas al dispositivo del modelo.
    pipe = ocr.paddlex_pipeline
    pipe = getattr(pipe, "_pipeline", pipe)
    for pred in (pipe.text_det_model, pipe.text_rec_model):
        pred.infer.to(dispositivo)
        print(f"  {pred.model_name} -> {next(pred.infer.parameters()).device}")
    return ocr


def procesar_pdf(ocr, nombre):
    doc = pymupdf.open(PDFS / nombre)
    paginas = []
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(dpi=DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        img = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR (convención de PaddleOCR)
        res = ocr.predict(img)[0]
        scores = list(res["rec_scores"])
        paginas.append({
            "pdf": nombre, "pagina": i,
            "texto_ocr": "\n".join(res["rec_texts"]),
            "conf_media": float(np.mean(scores)) if scores else None,
            "n_lineas": len(scores),
            "texto_nativo": page.get_text(),
        })
    return paginas


if __name__ == "__main__":
    args = sys.argv[1:]
    shard, n_shards = 0, 1
    if "--shard" in args:
        i = args.index("--shard")
        shard, n_shards = map(int, args[i + 1].split("/"))
        del args[i:i + 2]
    grupos = args or ["2026"]
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    nombres = []  # en el orden de los grupos: 2026 primero
    for g in grupos:
        det = pd.read_csv(ROOT / "data" / "interim" / f"detalle_{g}.csv")
        nombres += sorted({x for s in det.pdfs.dropna() for x in s.split(" | ")} - set(nombres))
    nombres = nombres[shard::n_shards]
    pendientes = [n for n in nombres if not (OCR_DIR / (Path(n).stem + ".jsonl")).exists()]
    print(f"shard {shard}/{n_shards}: {len(nombres)} PDFs ({len(pendientes)} pendientes)")
    ocr = crear_motor()
    t0, n_pags = time.time(), 0
    for k, nombre in enumerate(pendientes, 1):
        t = time.time()
        try:
            paginas = procesar_pdf(ocr, nombre)
        except Exception as e:  # PDF dañado o ilegible: se registra y se sigue
            print(f"  [{k}/{len(pendientes)}] {nombre}: ERROR {e!r}", flush=True)
            continue
        with open(OCR_DIR / (Path(nombre).stem + ".jsonl"), "w", encoding="utf-8") as f:
            for p in paginas:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        n_pags += len(paginas)
        print(f"  [{k}/{len(pendientes)}] {nombre}: {len(paginas)} págs en {time.time() - t:.1f}s",
              flush=True)
    seg = time.time() - t0
    print(f"Listo: {n_pags} páginas en {seg / 60:.1f} min ({seg / max(n_pags, 1):.2f} s/pág)")
