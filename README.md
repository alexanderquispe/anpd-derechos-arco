# ANPD – Decisiones sobre derechos ARCO (2026)

Descarga, OCR y análisis de las decisiones publicadas en
https://www.gob.pe/institucion/anpd/colecciones/1749-decisiones-sobre-derechos-arco

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas numpy requests beautifulsoup4 lxml pymupdf pyarrow openpyxl \
    paddlepaddle paddleocr torch torchvision transformers
.venv/bin/python -I scripts/01_indice.py          # índice de la colección (1 271 decisiones)
.venv/bin/python -I scripts/02_descargar.py 2026  # fichas + PDFs de las 25 decisiones de 2026
for i in 0 1 2; do .venv/bin/python -I scripts/03_ocr.py 2026 --shard $i/3 & done; wait   # PaddleOCR en MPS
.venv/bin/python -I scripts/04_corpus.py 2026     # corpus, calidad OCR, banderas temáticas
.venv/bin/python -I scripts/05_analisis.py        # análisis por caso + controles
```

El OCR usa el motor `transformers` de PaddleOCR 3.7 (PP-OCRv6_medium en PyTorch) con los
modelos movidos a `mps`, porque PaddlePaddle no tiene backend para la GPU de Apple.

Resultados en `outputs/`: `informe_casos_2026.md` (memo), `analisis_casos_2026.xlsx|csv`
(una fila por decisión), `casos_2026.csv` (banderas automáticas), `calidad_ocr_2026.csv`.
`data/` (PDFs, OCR) no se versiona.
