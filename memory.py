"""
memory.py — Histórico de conversa por sessão via Redis.

Usado pelo RunnableWithMessageHistory (fase 2 — chat livre sobre PDF).
Cada session_id tem seu próprio histórico isolado com TTL configurável.
"""

from langchain_community.chat_message_histories import RedisChatMessageHistory
from config import CACHE_REDIS_URI, HISTORY_TTL


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """
    Retorna o histórico Redis para uma sessão específica.
    Cria automaticamente se não existir.
    TTL vem de config.py → HISTORY_TTL (default: 3600s).
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=CACHE_REDIS_URI,
        ttl=HISTORY_TTL,
    )