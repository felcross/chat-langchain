import os
import re
import shutil
import logging
from collections import Counter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import RAG_FILES_DIR, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)

MODEL_PATH = "/app/hf_models/multilingual-e5-base"

# ─────────────────────────────────────────
# Singleton — embeddings
# ─────────────────────────────────────────
_embeddings_instance: HuggingFaceEmbeddings | None = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info("Carregando modelo de embeddings...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        logger.info("Modelo de embeddings carregado.")
    return _embeddings_instance


# ─────────────────────────────────────────
# Limpeza de texto
# ─────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Limpa texto extraído de PDF/CSV antes de vetorizar.
    Ordem importa: cada passo depende do anterior.
    """
    # 1. Remove caracteres de controle e unicode lixo (exceto \n e \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # 2. Resolve hifenização de fim de linha ("condi-\nção" → "condição")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # 3. Une linhas que são continuação de parágrafo
    #    Linha que NÃO termina com . ! ? : não é fim de frase — junta com a próxima
    text = re.sub(r'(?<![.!?:])\n(?!\n)(?![A-Z\d•\-])', ' ', text)

    # 4. Normaliza múltiplos espaços
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Normaliza múltiplas quebras de linha (> 2 vira 2 — separa parágrafos)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def remove_repeated_blocks(pages: list[str], threshold: float = 0.4) -> list[str]:
    """
    Remove cabeçalhos/rodapés que aparecem em >= threshold% das páginas.
    Estratégia: pega as primeiras e últimas linhas de cada página,
    conta frequência, remove as que aparecem demais.
    """
    if len(pages) < 3:
        return pages  # Poucos páginas — não vale filtrar

    line_counter: Counter = Counter()
    for page in pages:
        lines = page.strip().split('\n')
        candidates = set()
        # Primeiras 2 e últimas 2 linhas de cada página
        for line in lines[:2] + lines[-2:]:
            line = line.strip()
            if len(line) > 5:  # Ignora linhas muito curtas
                candidates.add(line)
        line_counter.update(candidates)

    min_occurrences = max(2, int(len(pages) * threshold))
    noise_lines = {line for line, count in line_counter.items() if count >= min_occurrences}

    if noise_lines:
        logger.info(f"Removendo {len(noise_lines)} linhas repetitivas (cabeçalhos/rodapés).")

    cleaned = []
    for page in pages:
        lines = page.split('\n')
        filtered = [l for l in lines if l.strip() not in noise_lines]
        cleaned.append('\n'.join(filtered))

    return cleaned


# ─────────────────────────────────────────
# Extração de PDF com PyMuPDF + fallback OCR
# ─────────────────────────────────────────
def extract_pdf(filepath: str) -> list[Document]:
    """
    Extrai texto de PDF com PyMuPDF.
    Se uma página não tiver texto útil, tenta OCR com pytesseract (fallback).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF não instalado. Execute: pip install pymupdf")

    docs = []
    pdf = fitz.open(filepath)
    pages_text = []

    for page_num, page in enumerate(pdf):
        text = page.get_text("text")

        # Fallback OCR — página sem texto útil (escaneada)
        if len(text.strip()) < 30:
            text = _ocr_page(page, page_num, filepath)

        pages_text.append(text)

    pdf.close()

    # Remove cabeçalhos/rodapés repetitivos antes de limpar
    pages_text = remove_repeated_blocks(pages_text)

    for page_num, text in enumerate(pages_text):
        text = clean_text(text)
        if len(text.strip()) < 20:
            continue  # Página vazia mesmo após tudo — ignora
        docs.append(Document(
            page_content=text,
            metadata={"source": os.path.basename(filepath), "page": page_num + 1}
        ))

    logger.info(f"PDF '{os.path.basename(filepath)}' → {len(docs)} páginas extraídas.")
    return docs


def _ocr_page(page, page_num: int, filepath: str) -> str:
    """Roda OCR em uma página via pytesseract. Retorna string vazia se não disponível."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang='por+eng')
        logger.info(f"OCR aplicado — página {page_num + 1} de '{os.path.basename(filepath)}'")
        return text
    except ImportError:
        logger.warning(
            f"Página {page_num + 1} sem texto e pytesseract não instalado — página ignorada."
        )
        return ""
    except Exception as e:
        logger.error(f"Erro no OCR página {page_num + 1}: {e}")
        return ""


# ─────────────────────────────────────────
# Extração de CSV
# ─────────────────────────────────────────
def extract_csv(filepath: str) -> list[Document]:
    """
    Cada linha do CSV vira um Document com contexto:
    'coluna1: valor1 | coluna2: valor2 | ...'
    Muito melhor do que texto cru para embedding.
    """
    import csv

    docs = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Ignora linhas completamente vazias
            values = [v.strip() for v in row.values() if v.strip()]
            if not values:
                continue

            text = ' | '.join(
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                if v.strip()
            )
            text = clean_text(text)
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(filepath), "row": i + 1}
            ))

    logger.info(f"CSV '{os.path.basename(filepath)}' → {len(docs)} linhas extraídas.")
    return docs


# ─────────────────────────────────────────
# Carregamento geral
# ─────────────────────────────────────────
def load_docs() -> list[Document]:
    """Carrega PDF, TXT e CSV de RAG_FILES_DIR e move para processed/ após processar."""
    docs = []
    processed_dir = os.path.join(RAG_FILES_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    supported = ('.pdf', '.txt', '.csv')
    files = [
        os.path.join(RAG_FILES_DIR, f)
        for f in os.listdir(RAG_FILES_DIR)
        if f.lower().endswith(supported)
    ]

    for file in files:
        try:
            ext = os.path.splitext(file)[1].lower()

            if ext == '.pdf':
                file_docs = extract_pdf(file)
            elif ext == '.csv':
                file_docs = extract_csv(file)
            else:  # .txt
                with open(file, encoding='utf-8') as f:
                    text = clean_text(f.read())
                file_docs = [Document(
                    page_content=text,
                    metadata={"source": os.path.basename(file)}
                )]

            docs.extend(file_docs)
            shutil.move(file, os.path.join(processed_dir, os.path.basename(file)))
            logger.info(f"Arquivo processado e movido: {os.path.basename(file)}")

        except Exception as e:
            logger.error(f"Erro ao carregar {file}: {e}")

    return docs


# ─────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────
def split_docs(docs: list[Document]) -> list[Document]:
    """
    Tenta preservar parágrafos. Só divide quando o parágrafo
    ultrapassa chunk_size — aí o RecursiveCharacterTextSplitter entra.
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for doc in docs:
        paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]

        for para in paragraphs:
            if len(para) <= 600:
                # Parágrafo cabe inteiro — preserva sem cortar
                chunks.append(Document(
                    page_content=para,
                    metadata=doc.metadata
                ))
            else:
                # Parágrafo longo — divide com overlap
                sub = splitter.create_documents([para], metadatas=[doc.metadata])
                chunks.extend(sub)

    return chunks


# ─────────────────────────────────────────
# Singleton — vectorstore
# ─────────────────────────────────────────
_vectorstore_instance: Chroma | None = None

def get_vectorstore() -> Chroma | None:
    global _vectorstore_instance

    if _vectorstore_instance is not None:
        return _vectorstore_instance

    embeddings = get_embeddings()

    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        logger.info("Banco vetorial encontrado. Carregando...")
        _vectorstore_instance = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings,
        )
        return _vectorstore_instance

    logger.info("Banco vetorial não encontrado. Vetorizando documentos...")
    docs = load_docs()
    if not docs:
        logger.warning("Nenhum documento encontrado em RAG_FILES_DIR.")
        return None

    chunks = split_docs(docs)

    _vectorstore_instance = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )
    logger.info(f"Banco vetorial criado com {len(chunks)} chunks.")
    return _vectorstore_instance