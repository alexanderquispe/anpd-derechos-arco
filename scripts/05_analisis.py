"""Análisis de las 25 decisiones ARCO publicadas en 2026.

La codificación de cada caso (hechos, resultado, motivo, relevancia) se hizo
leyendo el texto OCR completo de cada resolución (data/interim/textos_2026/).
Este script la cruza con las banderas automáticas del OCR (04_corpus.py) y con
la ficha de gob.pe para detectar inconsistencias, y genera las tablas finales.

Pregunta de la consulta: ¿la ANPD ha ordenado o logrado el retiro de datos
personales de notas periodísticas que vinculan a una persona con un delito
(inexacto, archivado o absuelto) vía derechos ARCO?

Uso:  .venv/bin/python -I scripts/05_analisis.py
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# (id_norma, expediente, resolución, fecha, reclamado según la resolución, tipo, derecho,
#  hechos, vínculo penal, alega inexactitud, resultado, ¿se retiró el dato?, motivo, relevancia)
CASOS = [
    (8283677, "173-2025-PTT", "196-2026", "2026-02-24", "Media Chirimoya (Perú 21) y Andina de Radiodifusión (ATV)",
     "Medio de prensa", "Cancelación y oposición",
     "Notas periodísticas digitales con nombre y foto, asociadas a una acusación penal de la que fue absuelto "
     "(resolución firme 2020). Invoca derecho al olvido; adjunta archivo fiscal, anulación de registro fiscal y "
     "certificado de antecedentes.",
     "Sí", "Sí", "Improcedente: no subsanó requisitos", "No",
     "No acreditó la solicitud previa recibida por los medios (su correo rebotó), ni precisó URLs, ni adjuntó "
     "búsqueda nominal en Google, ni motivó la oposición. No hubo pronunciamiento de fondo.", "Muy alta"),
    (8283714, "179-2025-PTT", "070-2026", "2026-01-30", "Empresa Editora El Comercio (Perú.com)",
     "Medio de prensa", "Cancelación y oposición",
     "La placa de su auto aparecía en una nota sobre un crimen cometido por un tercero (capitán del Ejército). "
     "Un juzgado le devolvió el vehículo al acreditar que ella era ajena al delito.",
     "Sí", "Sí", "Improcedente: sustracción de la materia (el medio retiró la nota)", "Sí (nota eliminada)",
     "El Comercio defendió libertad de prensa, interés público y hemerotecas, pero ante la resolución judicial "
     "eliminó la nota durante el procedimiento. La ANPD exhortó a El Comercio a responder ARCO en plazo.", "Muy alta"),
    (8283779, "180-2025-PTT", "072-2026", "2026-01-30", "Prensmart (Diario Correo)",
     "Medio de prensa", "Cancelación y oposición",
     "Misma reclamante y hechos que el EXP-179: placa de su auto en nota de Correo sobre el mismo crimen.",
     "Sí", "Sí", "Improcedente: sustracción de la materia (el medio suprimió el dato)", "Parcial (placa suprimida, nota sigue)",
     "Correo mantuvo la nota pero eliminó la placa (carta del 23/12/2025) tras conocer la resolución judicial.",
     "Muy alta"),
    (8284070, "223-2025-PTT", "130-2026", "2026-02-04", "Grupo La República Publicaciones",
     "Medio de prensa", "Oposición (+ desindexación)",
     "Nota 'fiscal habría favorecido a un empresario...'. El reclamante alega datos inexactos: control interno "
     "del Ministerio Público archivó por falta de indicios; pide cese del tratamiento y desindexación.",
     "Sí", "Sí", "Improcedente: no subsanó requisitos", "No",
     "Envió la solicitud previa por correo. La ANPD indicó que, al no tener La República canal ARCO, debía "
     "presentarse en su domicilio fiscal (Jr. Camaná 320, Lima); el cargo que adjuntó es posterior a la reclamación.",
     "Muy alta"),
    (8283505, "146-2025-PTT", "188-2026", "2026-02-19", "Empresa Editora El Comercio (Trome)",
     "Medio de prensa", "Cancelación",
     "Publicación vinculada a un proceso penal seguido en su contra, del que dice estar rehabilitado.",
     "Sí", "No", "Improcedente: requisito previo no cumplido", "No",
     "La solicitud ARCO por el canal habilitado (ventanilla ARCO de grupoelcomercio.com.pe) se presentó recién "
     "el 12/02/2026, después de iniciar la reclamación; el correo de 2023 no era el canal válido.", "Alta"),
    (8283316, "127-2025-PTT", "262-2026", "2026-03-03", "Google LLC (Google Search)",
     "Buscador", "Cancelación y oposición",
     "Noticias indexadas en Google sobre una denuncia penal que califica de calumniosa y que fue archivada "
     "(Fiscalía Mixta de Olmos). Pide que se borren sus datos de Google.",
     "Sí", "Sí", "Improcedente: no subsanó requisitos", "No",
     "No presentó el cargo de la solicitud previa dirigida a Google LLC ni la búsqueda nominal que acredite "
     "la indexación de las noticias.", "Alta"),
    (8280980, "105-2025-PTT", "066-2026", "2026-01-29", "Meta Platforms (Facebook)",
     "Red social", "Cancelación y oposición",
     "Video de un reportaje (página 'Perú Comunica' en Facebook) sobre la desaparición de la madre de su hijo; "
     "alega que no es persona pública y que el contexto noticioso concluyó.",
     "No", "No", "Improcedente: sustracción de la materia (la plataforma retiró el video)", "Sí (video retirado)",
     "Tras la admisión, Meta retiró el video por normas comunitarias; la ANPD verificó que ya no aparece en "
     "búsqueda nominal y cerró el caso.", "Alta"),
    (8280928, "096-2025-PTT", "314-2026", "2026-03-17", "Policía Nacional del Perú (DIRTIC)",
     "Entidad pública", "Cancelación",
     "Denuncia policial registrada en SIDPOL que el reclamante califica de falsa y que le impide postular a empleos.",
     "Sí", "Sí", "Improcedente: falsedad no acreditada", "No",
     "Único caso con análisis de fondo sobre una imputación penal: la ANPD no puede suponer que la denuncia es "
     "falsa sin un pronunciamiento de autoridad competente (archivo, sobreseimiento o absolución).", "Alta"),
    (8283424, "134-2025-PTT", "260-2026", "2026-02-27", "Poder Judicial",
     "Entidad pública", "Cancelación (+ oposición)",
     "Sentencia penal condenatoria publicada en el portal del PJ e indexada en Google; el reclamante está "
     "rehabilitado. Cita la Casación 3217-2022-Ica, retirada del portal y de Google en sede administrativa.",
     "Sí", "No", "Improcedente: no subsanó requisitos", "No",
     "No presentó la solicitud previa al Poder Judicial ni la búsqueda nominal.", "Media"),
    (8283539, "149-2025-PTT", "406-2026", "2026-03-30", "Poder Judicial",
     "Entidad pública", "Cancelación y oposición",
     "Publicación en el portal del PJ indexada en Google que afectaría sus relaciones laborales y familiares.",
     "No consta", "No", "Improcedente: no subsanó requisitos", "No",
     "Faltó la solicitud previa, la URL, la búsqueda nominal y el motivo fundado de la oposición.", "Media"),
    (8283433, "137-2025-PTT", "194-2026", "2026-02-23", "Ministerio de Justicia y Derechos Humanos",
     "Entidad pública", "Cancelación",
     "Convocatoria laboral de hace 13 años y viáticos de 2010 con su firma, en la web del MINJUS e indexados en Google.",
     "No", "No", "Improcedente: sustracción de la materia (la entidad retiró las publicaciones)", "Sí (publicaciones retiradas)",
     "El MINJUS eliminó las dos URLs y la ANPD verificó que ya no se indexan en Google.", "Media"),
    (8284016, "217-2025-PTT", "128-2026", "2026-02-03", "Ministerio de Justicia y Derechos Humanos – Tribunal de Transparencia (TTAIP)",
     "Entidad pública", "Cancelación",
     "Resolución del TTAIP con sus datos publicada en la web institucional.",
     "No consta", "No", "Improcedente: requisito previo no cumplido", "No",
     "El cargo de la solicitud previa al TTAIP es posterior a la reclamación; no acreditó búsqueda nominal.", "Media"),
    (8283826, "182-2025-PTT", "074-2026", "2026-01-30", "Ministerio Público – Fiscalía de la Nación",
     "Entidad pública", "Acceso",
     "Imputado en una investigación fiscal pide saber cómo y por qué el MP recopiló sus datos en RENIEC.",
     "Sí", "No", "Improcedente: sustracción de la materia (la entidad respondió)", "n.a.",
     "El MP respondió el pedido de acceso durante el procedimiento (Carta 000546-2025-MP-FN).", "Baja"),
    (8283641, "161-2025-PTT", "137-2026", "2026-02-11", "Superintendencia Nacional de Aduanas y de Administración Tributaria (SUNAT)",
     "Entidad pública", "Cancelación",
     "Pide cambio y eliminación de sus datos; el canal web exige captura biométrica que no funcionó.",
     "No", "No", "Improcedente: requisito previo no cumplido", "No",
     "La solicitud a SUNAT se presentó el mismo día que la reclamación ante la ANPD (sin dar plazo de respuesta).", "Baja"),
    (8283591, "152-2025-PTT", "263-2026", "2026-03-05", "Integratel Perú (Movistar)",
     "Empresa privada", "Acceso y cancelación",
     "Recibe cobros y mensajes de deudas de un tercero asociados a su número de celular.",
     "No", "Sí", "Improcedente: no subsanó requisitos", "No",
     "No subsanó la observación en el plazo otorgado.", "Baja"),
    (8283985, "216-2025-PTT", "264-2026", "2026-03-06", "Clínica Florez Salud",
     "Empresa privada", "Acceso", "Pide videos y audios de cámaras de seguridad de su visita a la clínica.",
     "No", "No", "Fundada", "n.a.", "Se ordena entregar copia de videos y audios en 20 días hábiles.", "Baja"),
    (8283940, "215-2025-PTT", "189-2026", "2026-02-20", "Clínica Emmel (Tecnologías Médicas Arequipa)",
     "Empresa privada", "Acceso", "Pide videos de cámaras del primer piso de la clínica.",
     "No", "No", "Fundada", "n.a.", "Se ordena entregar copia de los videos en 20 días hábiles.", "Baja"),
    (8283876, "201-2025-PTT", "187-2026", "2026-02-18", "Clínica Emmel (Tecnologías Médicas Arequipa)",
     "Empresa privada", "Acceso", "Pide videos y audios de cámaras del quinto piso de la clínica.",
     "No", "No", "Fundada en parte", "n.a.", "Se ordena entregar videos y audios en 20 días hábiles.", "Baja"),
    (8283840, "187-2025-PTT", "007-2026", "2026-01-19", "América Móvil Perú (Claro)",
     "Empresa privada", "Acceso", "Pide videos de cámaras de un centro de atención al cliente.",
     "No", "No", "Fundada en parte", "n.a.", "Se ordena dar acceso a imágenes y voces suyas y de su hija.", "Baja"),
    (8283856, "200-2025-PTT", "143-2026", "2026-02-17", "Interbank",
     "Empresa privada", "Acceso", "Pide videos de cámaras de una agencia bancaria.",
     "No", "No", "Improcedente: los datos ya no existían", "n.a.",
     "Reclamó después del plazo de conservación de las grabaciones (eliminadas).", "Baja"),
    (8279521, "028-2025-PTT", "068-2026", "2026-01-30", "Línea 1 del Metro de Lima",
     "Empresa privada", "Acceso", "Pide el video de un accidente en una estación.",
     "No", "No", "Improcedente: los datos ya no existían", "n.a.",
     "Pidió el video después de los 30 días hábiles de conservación.", "Baja"),
    (8283400, "131-2025-PTT", "259-2026", "2026-02-27", "British American Hospital (Clínica Anglo Americana)",
     "Empresa privada", "Acceso", "Pide la información médica que su empleador compartió con la clínica.",
     "No", "No", "Improcedente: los datos ya no existían", "n.a.",
     "La clínica no almacenó ni trató los archivos (fueron eliminados).", "Baja"),
    (8281038, "116-2025-PTT", "258-2026", "2026-02-27", "Oncocenter Perú (Auna)",
     "Empresa privada", "Acceso", "Pide la información médica que su empleador compartió con el centro.",
     "No", "No", "Improcedente: los datos ya no existían", "n.a.", "No se produce el tratamiento objeto de la tutela.", "Baja"),
    (8281018, "114-2025-PTT", "061-2026", "2026-01-23", "Mercado Libre Perú",
     "Empresa privada", "Acceso", "Pide información sobre la suspensión de su cuenta de vendedor.",
     "No", "No", "Improcedente: incompetencia", "n.a.", "Lo pedido no son datos personales (fuera de la LPDP).", "Baja"),
    (8284053, "221-2025-PTT", "003-2026", "2026-01-14", "Ministerio de Transportes y Comunicaciones",
     "Entidad pública", "Acceso", "Pide IP, MAC y usuario desde los que se registraron papeletas a su nombre.",
     "No", "No", "Improcedente: incompetencia", "n.a.", "No corresponde a la ANPD (materia de acceso a información).", "Baja"),
]
COLUMNAS = ["id_norma", "expediente", "resolucion_rd", "fecha_resolucion", "reclamado_segun_resolucion",
            "tipo_reclamado", "derecho", "hechos", "vinculo_penal", "alega_inexactitud", "resultado",
            "retiro_logrado", "motivo", "relevancia_consulta"]
ORDEN_REL = {"Muy alta": 0, "Alta": 1, "Media": 2, "Baja": 3}


def main():
    cod = pd.DataFrame(CASOS, columns=COLUMNAS)
    cod["fecha_resolucion"] = pd.to_datetime(cod.fecha_resolucion)
    auto = pd.read_csv(OUT / "casos_2026.csv")
    assert len(cod) == 25 and cod.id_norma.is_unique and set(cod.id_norma) == set(auto.id_norma), \
        "La codificación no cubre exactamente las 25 decisiones de 2026"
    df = cod.merge(auto[["id_norma", "reclamado", "derechos_arco", "decision_ficha", "sentido_fallo_ocr",
                         "f_prensa", "f_buscador", "f_delito_penal", "f_inexactitud", "f_derecho_olvido",
                         "url"]], on="id_norma")

    # Control 1: el fallo leído vs. el fallo extraído por regex del OCR
    # (los "Fundada en parte" salen como "revisar": la parte resolutiva dice "FUNDADO el extremo")
    sentido_cod = df.resultado.str.lower().str.extract(r"^(fundada en parte|fundada|improcedente)")[0]
    df["control_fallo_ocr"] = (sentido_cod == df.sentido_fallo_ocr.str.lower()).map({True: "ok", False: "revisar"})
    # Control 2: la ficha de gob.pe nombra a otro reclamado
    def palabras(s):
        return {w for w in re.findall(r"\w+", str(s).lower()) if len(w) >= 4}

    df["ficha_gobpe_reclamado_distinto"] = [not (palabras(f) & palabras(r))
                                            for f, r in zip(df.reclamado, df.reclamado_segun_resolucion)]

    df = df.sort_values(["relevancia_consulta", "fecha_resolucion"],
                        key=lambda s: s.map(ORDEN_REL) if s.name == "relevancia_consulta" else s)
    df.to_csv(OUT / "analisis_casos_2026.csv", index=False)
    with pd.ExcelWriter(OUT / "analisis_casos_2026.xlsx") as xw:
        df.drop(columns=[c for c in df if c.startswith("f_")]).to_excel(xw, sheet_name="casos", index=False)

    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print("Resultados de las 25 decisiones de 2026")
    print(df.resultado.value_counts().to_string(), "\n")
    print("Por relevancia para la consulta")
    print(df.relevancia_consulta.value_counts().reindex(ORDEN_REL).to_string(), "\n")
    prensa = df[df.tipo_reclamado.isin(["Medio de prensa", "Buscador", "Red social"])]
    print(f"Casos contra medios/buscador/red social: {len(prensa)}; con vínculo penal: "
          f"{(prensa.vinculo_penal == 'Sí').sum()}; retiro logrado (total o parcial): "
          f"{prensa.retiro_logrado.str.startswith(('Sí', 'Parcial')).sum()}; "
          f"fondo resuelto a favor por la ANPD: {prensa.resultado.str.startswith('Fundada').sum()}")
    print(prensa[["expediente", "reclamado_segun_resolucion", "vinculo_penal", "resultado", "retiro_logrado"]]
          .to_string(index=False), "\n")
    print("Control fallo leído vs. regex OCR:", df.control_fallo_ocr.value_counts().to_dict())
    print(df.loc[df.control_fallo_ocr == "revisar", ["expediente", "resultado", "sentido_fallo_ocr"]].to_string(index=False))
    malos = df[df.ficha_gobpe_reclamado_distinto]
    print("\nFichas de gob.pe con reclamado distinto al de la resolución:")
    print(malos[["expediente", "reclamado", "reclamado_segun_resolucion"]].to_string(index=False))


if __name__ == "__main__":
    main()
