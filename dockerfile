# ── Dockerfile — FinanceBot BR ────────────────────────────────────────────────
#
# Build multi-stage não é necessário aqui pois não há compilação pesada.
# Imagem base slim para reduzir tamanho final (~200MB vs ~1GB do full).
#
# Volumes esperados no docker-compose:
#   ./finance.db  → /app/finance.db   (banco DuckDB persistido)
#   ./.cache      → /app/.cache       (cache diskcache)
#   ./query_logs.jsonl → /app/query_logs.jsonl (logs de queries)
#
# Variáveis de ambiente obrigatórias (via .env ou docker-compose):
#   GROQ_API_KEY

FROM python:3.11-slim

# Evita prompts interativos durante apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema
# libgomp1: necessário para DuckDB (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (camada cacheável pelo Docker)
# Se só o código mudar, essa camada não é recompilada — build muito mais rápido.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY app.py .
COPY etl_bcb.py .
COPY db_manager.py .
COPY query_logger.py .
COPY cache_manager.py .

# Porta padrão do Streamlit
EXPOSE 8501

# Healthcheck — Docker reinicia o container se o Streamlit parar de responder
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização
# --server.address=0.0.0.0 → aceita conexões externas (necessário no Docker)
# --server.headless=true   → desativa abertura automática do browser
# --server.fileWatcherType=none → evita inotify errors em alguns sistemas
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]