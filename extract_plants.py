from pathlib import Path
from docx import Document
from pypdf import PdfReader
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unidecode import unidecode

import pytesseract
import json
import re

DOCUMENTS_DIR = Path("documents")

PLANTS_DIR = Path("data/plants")
JSON_DIR = Path("data/json")
CHUNKS_DIR = Path("data/chunks")

for directory in [
    PLANTS_DIR,
    JSON_DIR,
    CHUNKS_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


KNOWN_PLANTS = [
    "FLORIPONDIO",
    "Hierba buena",
    "ALBAHACAR",
    "RUDA",
    "ZACATE LIMON",
    "TOMILLO",
    "OREGANO",
    "VIOLETA",
    "HIERBA MAESTRA",
    "Dieffenbachia",
    "Lindernia crustácea",
    "Hoja Santa"
]


def read_docx(path):

    doc = Document(path)

    return "\n".join(
        p.text.strip()
        for p in doc.paragraphs
        if p.text.strip()
    )


def read_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def pdf_ocr(path):

    images = convert_from_path(path)

    text = ""

    for image in images:

        text += pytesseract.image_to_string(
            image,
            lang="spa"
        )

        text += "\n"

    return text


def read_pdf_smart(path):

    text = read_pdf(path)

    if len(text.strip()) < 500:

        print(f"[OCR] {path.name}")

        text = pdf_ocr(path)

    return text


def load_document(path):

    suffix = path.suffix.lower()

    if suffix == ".docx":
        return read_docx(path)

    elif suffix == ".pdf":
        return read_pdf_smart(path)

    elif suffix == ".txt":

        return path.read_text(
            encoding="utf-8"
        )

    return ""


def clean_text(text):

    text = re.sub(
        r'https?://\S+',
        '',
        text
    )

    text = re.sub(
        r'PDF:',
        '',
        text
    )

    text = re.sub(
        r'Disponible en:',
        '',
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'\n{2,}',
        '\n',
        text
    )

    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    return text.strip()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

all_text = ""

for file in DOCUMENTS_DIR.iterdir():

    print(
        f"Procesando {file.name}"
    )

    all_text += (
        "\n" +
        load_document(file)
    )

all_text = clean_text(
    all_text
)

sections = {}

for i, plant in enumerate(
    KNOWN_PLANTS
):

    start = all_text.find(
        plant
    )

    if start == -1:
        continue

    end = len(all_text)

    for next_plant in KNOWN_PLANTS[i+1:]:

        pos = all_text.find(
            next_plant
        )

        if pos > start:
            end = min(end, pos)

    sections[plant] = (
        all_text[start:end]
    )

print(
    f"Plantas detectadas: {len(sections)}"
)

for plant, content in sections.items():

    safe_name = (
        unidecode(plant)
        .lower()
        .replace(" ", "_")
    )

    plant_file = (
        PLANTS_DIR /
        f"{safe_name}.txt"
    )

    plant_file.write_text(
        content,
        encoding="utf-8"
    )

    metadata = {
        "nombre": plant,
        "archivo": safe_name,
        "caracteres": len(content)
    }

    json_file = (
        JSON_DIR /
        f"{safe_name}.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

    chunks = splitter.split_text(
        content
    )

    for idx, chunk in enumerate(
        chunks
    ):

        chunk_file = (
            CHUNKS_DIR /
            f"{safe_name}_{idx}.txt"
        )

        chunk_file.write_text(
            chunk,
            encoding="utf-8"
        )

print("Extracción terminada")