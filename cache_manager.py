"""
cache_manager.py — Cache Redis para insights gerados pelo LLM.

Adaptado do FinanceBot BR (diskcache → Redis), unificando com o Redis
já usado para histórico de conversa (memory.py).

Estratégia de chave:
    insights:{hash(nome + shape + colunas)}
    
Se o arquivo mudar (nome, tamanho ou colunas diferentes), o hash muda
e o cache é automaticamente invalidado — sem lógica extra.
"""

import hashlib
import json
import logging

import redis

from config import CACHE_REDIS_URI, INSIGHTS_CACHE_TTL

log = logging.getLogger(__name__)

# Conexão síncrona (Streamlit não é async)
_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(CACHE_REDIS_URI, decode_responses=True)
    return _client


# ── Chave ─────────────────────────────────────────────────────────────────────

def _build_key(file_name: str, n_rows: int, columns: list[str]) -> str:
    """
    Gera chave determinística baseada nas características do arquivo.
    Troca de arquivo → hash diferente → cache miss → regera insights.
    """
    fingerprint = f"{file_name}|{n_rows}|{','.join(sorted(columns))}"
    digest = hashlib.md5(fingerprint.encode()).hexdigest()
    return f"insights:{digest}"


# ── API pública ───────────────────────────────────────────────────────────────

def get_cached_insights(file_name: str, n_rows: int, columns: list[str]) -> list[dict] | None:
    """
    Retorna sugestões em cache ou None se não encontrado / expirado.
    """
    try:
        key = _build_key(file_name, n_rows, columns)
        raw = _get_client().get(key)
        if raw:
            log.info(f"[CACHE] Hit para '{file_name}'")
            return json.loads(raw)
        log.info(f"[CACHE] Miss para '{file_name}'")
        return None
    except Exception as e:
        log.warning(f"[CACHE] Erro ao ler cache: {e}")
        return None


def set_cached_insights(
    file_name: str,
    n_rows: int,
    columns: list[str],
    insights: list[dict],
) -> None:
    """
    Salva sugestões no Redis com TTL configurável.
    """
    try:
        key = _build_key(file_name, n_rows, columns)
        _get_client().setex(key, INSIGHTS_CACHE_TTL, json.dumps(insights, ensure_ascii=False))
        log.info(f"[CACHE] Insights salvos para '{file_name}' (TTL={INSIGHTS_CACHE_TTL}s)")
    except Exception as e:
        log.warning(f"[CACHE] Erro ao salvar cache: {e}")


def invalidar_cache_insights(file_name: str, n_rows: int, columns: list[str]) -> None:
    """
    Remove o cache de um arquivo específico (ex: usuário clica em 'Regenerar').
    """
    try:
        key = _build_key(file_name, n_rows, columns)
        _get_client().delete(key)
        log.info(f"[CACHE] Cache invalidado para '{file_name}'")
    except Exception as e:
        log.warning(f"[CACHE] Erro ao invalidar cache: {e}")