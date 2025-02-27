import fitz
import pyarrow as pa
import docx
import pandas as pd
import re
import datetime
import lancedb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from lancedb.pydantic import Vector, LanceModel
from sentence_transformers import SentenceTransformer
from pathlib import Path

class Content(LanceModel):
    pageId: str
    vector: Vector(384)
    title: str    
    Namedocument: str
    modificationD: str


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(BASE_DIR / "db" / "csicapdb")
DIRECTORY_PATH = str(BASE_DIR / "data" / "files")
PROCESSED_FILES_LOG = str(BASE_DIR / "data" / "train" / "processed_files.txt")

def load_processed_files():
    try:
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def save_processed_file(file_path):
    with open(PROCESSED_FILES_LOG, 'a') as f:
        f.write(f"{file_path}\n")

db = lancedb.connect(DB_PATH)
table_name = "files"
vector_dim = 384

if table_name not in db.table_names():
    schema = pa.schema([
        pa.field("pageId", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        pa.field("title", pa.string()),
        pa.field("Namedocument", pa.string()),
        pa.field("modificationD", pa.string())
    ])
    db.create_table(table_name, schema=schema)
    print(f"Table '{table_name}' created.")
table = db.open_table(table_name)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text(file_path):
    ext = Path(file_path).suffix.lower()
    text = ""
    try:
        if ext == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            text = " ".join(df.astype(str).values.flatten().tolist())
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Unsupported format: {ext}")
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text

def split_into_chunks(text, chunk_size=100, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return splitter.split_text(text)

def embed_text(text):
    return embedding_model.encode(text).tolist()

def process_large_pdf(pdf_path, chunk_size=800, chunk_overlap=100):
    try:
        doc = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).name
        modification_date = str(datetime.datetime.fromtimestamp(Path(pdf_path).stat().st_mtime))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        data_list = []
        batch_size = 100
        for page_num in range(len(doc)):
            print(f"Processing page {page_num+1}/{len(doc)} of {pdf_name}...")
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if not page_text.strip():
                    continue
                chunks = text_splitter.split_text(page_text)
                for chunk in chunks:
                    cleaned_text = re.sub(r"[&\[\]\-\)\(\-]", "", chunk).lower().strip()
                    if cleaned_text:
                        embedding_vector = embed_text(cleaned_text)
                        data = {
                            "pageId": str(page_num+1),
                            "vector": embedding_vector,
                            "title": cleaned_text,
                            "Namedocument": pdf_name,
                            "modificationD": modification_date
                        }
                        data_list.append(data)
                        if len(data_list) >= batch_size:
                            table.add([Content(**item).model_dump() for item in data_list])
                            data_list = []
            except Exception as e:
                print(f"Error processing page {page_num+1} of {pdf_name}: {e}")
                continue
        if data_list:
            table.add([Content(**item).model_dump() for item in data_list])
        doc.close()
        print(f"{pdf_name} processed successfully.")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        if 'doc' in locals():
            doc.close()

def process_document(file_path, chunk_size=100):
    try:
        print(f"Processing document: {file_path}")
        text = extract_text(file_path)
        if not text.strip():
            print(f"No text extracted from {file_path}. Skipping.")
            return
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function=len)
        chunks = text_splitter.split_text(text)
        doc_name = Path(file_path).name
        modification_date = str(datetime.datetime.fromtimestamp(Path(file_path).stat().st_mtime))
        page_id = "1"
        data_list = []
        batch_size = 100
        for idx, chunk in enumerate(chunks):
            cleaned_text = re.sub(r"[&\[\]\-\)\(\-]", "", chunk).lower().strip()
            if cleaned_text:
                embedding_vector = embed_text(cleaned_text)
                data = {
                    "pageId": page_id,
                    "vector": embedding_vector,
                    "title": cleaned_text,
                    "Namedocument": doc_name,
                    "modificationD": modification_date
                }
                data_list.append(data)
                if len(data_list) >= batch_size:
                    table.add([Content(**item).model_dump() for item in data_list])
                    data_list = []
        if data_list:
            table.add([Content(**item).model_dump() for item in data_list])
        print(f"{doc_name} processed successfully.")
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")

def process_documents_in_directory(directory_path):
    supported_ext = [".pdf", ".docx", ".xlsx", ".xls", ".txt"]
    processed_files = load_processed_files()
    files = [f for f in Path(directory_path).iterdir() if f.suffix.lower() in supported_ext and str(f) not in processed_files]
    if not files:
        print("No new files to process.")
        return
    print(f"Found {len(files)} new files to process.")
    for file in files:
        try:
            if file.suffix.lower() == ".pdf":
                process_large_pdf(str(file))
            else:
                process_document(str(file))
            save_processed_file(str(file))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    print("All new documents processed and stored in LanceDB.")

if __name__ == "__main__":
    process_documents_in_directory(DIRECTORY_PATH)