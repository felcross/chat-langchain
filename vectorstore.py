"""
vectorstore.py — Pipeline PDF para o doc-analista.

Fluxo por sessão:
    upload PDF → extract_pdf() → split_docs() → Chroma em dir temporário
    app.py guarda vs e vs_dir no session_state
    sessão encerrada → app.py apaga vs_dir via shutil.rmtree()
    MCP acessa via vs_dir enquanto sessão está ativa

Nenhum arquivo fica acumulando — limpeza explícita no limpar_sessao() do app.py.
"""

import os
import re
import logging
import tempfile
from collections import Counter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import HF_MODEL_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings — singleton (modelo carrega uma vez, vive enquanto o container vive)
# ─────────────────────────────────────────────────────────────────────────────

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Retorna a instância singleton do modelo de embeddings.
    O modelo HF é pesado (1.1 GB) — carrega uma vez e fica em memória.
    Path vem de config.py → HF_MODEL_PATH (default: /app/hf_models/multilingual-e5-base).
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Carregando modelo de embeddings: {HF_MODEL_PATH}")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=HF_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Modelo de embeddings carregado.")
    return _embeddings_instance


# ─────────────────────────────────────────────────────────────────────────────
# Limpeza de texto
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpa texto extraído de PDF antes de vetorizar.
    Ordem importa: cada passo depende do anterior.
    """
    # 1. Remove caracteres de controle e unicode lixo (exceto \n e \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # 2. Resolve hifenização de fim de linha ("condi-\nção" → "condição")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Une linhas que são continuação de parágrafo
    text = re.sub(r"(?<![.!?:])\n(?!\n)(?![A-Z\d•\-])", " ", text)

    # 4. Normaliza múltiplos espaços
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Normaliza múltiplas quebras de linha (> 2 vira 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_repeated_blocks(pages: list[str], threshold: float = 0.4) -> list[str]:
    """
    Remove cabeçalhos/rodapés que aparecem em >= threshold% das páginas.
    """
    if len(pages) < 3:
        return pages

    line_counter: Counter = Counter()
    for page in pages:
        lines = page.strip().split("\n")
        candidates = set()
        for line in lines[:2] + lines[-2:]:
            line = line.strip()
            if len(line) > 5:
                candidates.add(line)
        line_counter.update(candidates)

    min_occurrences = max(2, int(len(pages) * threshold))
    noise_lines = {line for line, count in line_counter.items() if count >= min_occurrences}

    if noise_lines:
        logger.info(f"Removendo {len(noise_lines)} linhas repetitivas (cabeçalhos/rodapés).")

    cleaned = []
    for page in pages:
        lines = page.split("\n")
        filtered = [l for l in lines if l.strip() not in noise_lines]
        cleaned.append("\n".join(filtered))

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Extração de PDF
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf(filepath: str) -> list[Document]:
    """
    Extrai texto de PDF com PyMuPDF.
    Páginas sem texto útil (< 30 chars) tentam OCR via pytesseract.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF não instalado. Execute: pip install pymupdf")

    pdf = fitz.open(filepath)
    pages_text = []

    for page_num, page in enumerate(pdf):
        text = page.get_text("text")
        if len(text.strip()) < 30:
            text = _ocr_page(page, page_num, filepath)
        pages_text.append(text)

    pdf.close()

    pages_text = remove_repeated_blocks(pages_text)

    docs = []
    for page_num, text in enumerate(pages_text):
        text = clean_text(text)
        if len(text.strip()) < 20:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": os.path.basename(filepath), "page": page_num + 1},
        ))

    logger.info(f"PDF '{os.path.basename(filepath)}' → {len(docs)} páginas extraídas.")
    return docs


def _ocr_page(page, page_num: int, filepath: str) -> str:
    """Fallback OCR via pytesseract para páginas escaneadas."""
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="por+eng")
        logger.info(f"OCR aplicado — página {page_num + 1} de '{os.path.basename(filepath)}'")
        return text
    except ImportError:
        logger.warning(f"Página {page_num + 1} sem texto e pytesseract não instalado — ignorada.")
        return ""
    except Exception as e:
        logger.error(f"Erro no OCR página {page_num + 1}: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def split_docs(docs: list[Document]) -> list[Document]:
    """
    Preserva parágrafos quando cabem inteiros (≤ 600 chars).
    Divide com overlap apenas quando o parágrafo ultrapassa o limite.
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for doc in docs:
        paragraphs = [p.strip() for p in doc.page_content.split("\n\n") if p.strip()]
        for para in paragraphs:
            if len(para) <= 600:
                chunks.append(Document(page_content=para, metadata=doc.metadata))
            else:
                sub = splitter.create_documents([para], metadatas=[doc.metadata])
                chunks.extend(sub)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada público — chamado pelo app.py
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(pdf_path: str) -> tuple[int, Chroma, str]:
    """
    Constrói o vectorstore para um PDF específico e persiste em dir temporário.

    O dir temporário é criado aqui e retornado para o app.py guardar no
    session_state. O MCP acessa via esse path enquanto a sessão está ativa.
    O app.py apaga o dir via shutil.rmtree() no limpar_sessao().

    Args:
        pdf_path: caminho absoluto do PDF (arquivo temporário do upload)

    Returns:
        (n_chunks, chroma_instance, vs_dir)
        n_chunks  — exibido no app ("X trechos indexados")
        chroma    — vectorstore pronto para similarity_search
        vs_dir    — path do dir temporário (para o MCP e para limpeza)
    """
    embeddings = get_embeddings()

    docs = extract_pdf(pdf_path)
    if not docs:
        raise ValueError(f"Nenhum conteúdo extraído do PDF: {pdf_path}")

    chunks = split_docs(docs)
    if not chunks:
        raise ValueError("Nenhum chunk gerado após o processamento do PDF.")

    vs_dir = tempfile.mkdtemp(prefix="docanalista_vs_")
    logger.info(f"Construindo vectorstore em '{vs_dir}' com {len(chunks)} chunks...")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vs_dir,
    )

    logger.info("Vectorstore pronto.")
    return len(chunks), vs, vs_dir