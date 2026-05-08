# ── Dockerfile — FinanceBot BR ────────────────────────────────────────────────
# Imagem base slim para eficiência de espaço e segurança
FROM python:3.11-slim

# Evita prompts interativos e gera logs do Python em tempo real
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instala dependências do sistema
# libgomp1: Essencial para o DuckDB rodar processamento paralelo
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Copia apenas o requirements primeiro para aproveitar o cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copia TODO o restante do projeto (Isso resolve o erro do analista.py)
# O Docker vai ignorar o que estiver no seu .dockerignore (como venv ou chaves)
COPY . .

# Porta padrão do Streamlit
EXPOSE 8501

# Healthcheck — Garante que o container reinicie se o Streamlit travar
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização otimizado para Docker
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]