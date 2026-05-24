"""
config.py — Configurações centralizadas do doc-analista.

Unifica variáveis do FinanceBot BR (Groq, cache) e do Chatbot WhatsApp
(Redis, modelo HF, vectorstore), todas vindas do .env via python-decouple.
"""

from decouple import config

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str    = config("GROQ_API_KEY")
MODEL_NAME: str      = config("MODEL_NAME",        default="llama-3.3-70b-versatile")
MODEL_TEMPERATURE    = config("MODEL_TEMPERATURE", default=0,    cast=float)
MODEL_MAX_TOKENS     = config("MODEL_MAX_TOKENS",  default=1024, cast=int)

# ── Redis ─────────────────────────────────────────────────────────────────────
CACHE_REDIS_URI: str = config("CACHE_REDIS_URI", default="redis://redis:6379/0")

# Histórico de conversa (memory.py)
HISTORY_TTL: int        = config("HISTORY_TTL",        default=3600, cast=int)   # 1h

# Cache de insights gerados pelo LLM (cache_manager.py)
INSIGHTS_CACHE_TTL: int = config("INSIGHTS_CACHE_TTL", default=3600, cast=int)   # 1h

# ── Vectorstore / Embeddings ──────────────────────────────────────────────────
# Caminho do modelo local (montado via volume Docker)
HF_MODEL_PATH: str     = config("HF_MODEL_PATH",     default="/app/hf_models/multilingual-e5-base")
VECTOR_STORE_PATH: str = config("VECTOR_STORE_PATH", default="/app/vectorstore")
RAG_FILES_DIR: str     = config("RAG_FILES_DIR",     default="/app/rag_files")

# ── Prompts do agente ─────────────────────────────────────────────────────────
AI_CONTEXTUALIZE_PROMPT: str = config(
    "AI_CONTEXTUALIZE_PROMPT",
    default=(
        "Dado o histórico da conversa e a pergunta mais recente do usuário, "
        "reformule a pergunta de forma independente para que possa ser entendida "
        "sem o histórico. NÃO responda — apenas reformule se necessário, "
        "ou retorne como está."
    ),
)

AI_SYSTEM_PROMPT: str = config(
    "AI_SYSTEM_PROMPT",
    default=(
        "Você é um analista de documentos especialista. "
        "Use os trechos de contexto recuperados para responder à pergunta. "
        "Seja objetivo e claro. Se não souber a resposta com base no contexto, "
        "diga que não encontrou a informação no documento. "
        "Responda SEMPRE em português do Brasil.\n\n"
        "{context}"
    ),
)