"""
cache_manager.py — Cache de respostas do agente SQL com diskcache.

Por que cachear?
    O agente SQL envolve uma chamada à API Groq (LLM) + uma query DuckDB.
    Perguntas idênticas (ex: "qual a Selic hoje?") feitas várias vezes
    consomem tokens desnecessariamente. O cache guarda a resposta por N horas.

Como funciona o diskcache:
    • Persiste em disco (diretório .cache/) — sobrevive a reinícios do Streamlit
    • Em Docker: montar o diretório .cache/ como volume para não perder o cache
    • TTL configurável por tipo de dado (dados diários vs mensais têm ritmos diferentes)
    • Thread-safe por padrão

Estratégia de chave:
    cache_key = hash(pergunta_normalizada + modo)
    Normalização: lowercase + strip — "Qual a Selic?" == "qual a selic?"
"""

import hashlib
import logging

import diskcache

log = logging.getLogger(__name__)

# ── Configuração ───────────────────────────────────────────────────────────────
CACHE_DIR = ".cache"          # diretório de persistência (montar como volume no Docker)
TTL_HORAS = 6                 # tempo de vida padrão das entradas (em horas)
TTL_SEGUNDOS = TTL_HORAS * 3600

# Cache singleton — instanciado uma vez, compartilhado por toda a sessão
_cache = diskcache.Cache(CACHE_DIR)


def _normalizar(texto: str) -> str:
    """Remove variações superficiais que não mudam o significado da pergunta."""
    return texto.strip().lower()


def _make_key(pergunta: str, modo: str) -> str:
    """
    Gera uma chave de cache determinística.

    Usa SHA-256 para evitar colisões e manter chaves curtas.
    O modo ("especialista" / "analista") faz parte da chave para evitar
    que a mesma pergunta retorne resposta do modo errado.
    """
    conteudo = f"{modo}::{_normalizar(pergunta)}"
    return hashlib.sha256(conteudo.encode()).hexdigest()


def get_cached(pergunta: str, modo: str) -> str | None:
    """
    Busca resposta cacheada.

    Retorna a resposta (str) se existir e não tiver expirado, None caso contrário.
    """
    key = _make_key(pergunta, modo)
    resultado = _cache.get(key)
    if resultado is not None:
        log.debug(f"Cache HIT — modo={modo}, pergunta={pergunta[:50]}")
    return resultado


def set_cached(pergunta: str, modo: str, resposta: str, ttl: int = TTL_SEGUNDOS):
    """
    Salva uma resposta no cache.

    Parâmetros
    ----------
    pergunta : texto original do usuário
    modo     : "especialista" ou "analista"
    resposta : resposta final gerada pelo agente
    ttl      : tempo de vida em segundos (padrão: TTL_HORAS)
    """
    key = _make_key(pergunta, modo)
    _cache.set(key, resposta, expire=ttl)
    log.debug(f"Cache SET — modo={modo}, ttl={ttl}s, pergunta={pergunta[:50]}")


def invalidar_cache():
    """
    Limpa todo o cache.
    Chamado quando o ETL atualiza os dados — respostas antigas podem estar desatualizadas.
    """
    _cache.clear()
    log.info("Cache invalidado após atualização do ETL.")


def stats_cache() -> dict:
    """
    Retorna estatísticas do cache para exibir na sidebar (debug/monitoramento).
    """
    return {
        "entradas": len(_cache),
        "tamanho_mb": round(_cache.volume() / 1024 / 1024, 2),
        "diretorio": CACHE_DIR,
    }