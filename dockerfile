FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Tesseract para OCR fallback + libgomp1 para DuckDB paralelo
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-por \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Torch CPU-only PRIMEIRO — evita o sentence-transformers puxar a versão CUDA (532 MB)
# CPU-only: ~180 MB vs ~530 MB da versão padrão com CUDA
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Restante das dependências (sentence-transformers já encontra torch instalado)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/vectorstore /app/rag_files /app/hf_models

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]