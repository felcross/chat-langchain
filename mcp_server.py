"""
mcp_server.py — Servidor MCP via stdio (FastMCP).

Expõe as ferramentas do doc-analista para qualquer cliente MCP compatível
(Claude Desktop, VS Code, ou o próprio agente Groq via langchain-mcp-adapters).

Transporte: stdio — spawneado como subprocess pelo app.py.
NÃO adicionar ao docker-compose; roda como processo filho do Streamlit.

Ferramentas expostas:
    query_csv        → SQL sobre o CSV da sessão (DuckDB :memory:)
    search_pdf       → busca semântica no PDF vetorizado
    get_doc_summary  → metadados do documento atual
"""

import json
import logging

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

mcp = FastMCP("doc-analista")


# ── Estado compartilhado (preenchido pelo app.py antes de spawnar) ────────────
# O app.py grava um JSON temporário com o estado da sessão que o servidor lê.
# Alternativa mais simples que passar tudo via args de linha de comando.

import os
import tempfile

_STATE_FILE = os.environ.get("DOC_ANALISTA_STATE", "")


def _load_state() -> dict:
    if not _STATE_FILE or not os.path.exists(_STATE_FILE):
        return {}
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Tool 1: query_csv ─────────────────────────────────────────────────────────

@mcp.tool()
def query_csv(sql: str) -> dict:
    """
    Executa uma query SQL sobre o CSV carregado na sessão atual.
    A tabela se chama 'dados'. Use DuckDB SQL.
    Retorna as linhas como lista de dicionários ou erro descritivo.

    Exemplo:
        SELECT categoria, SUM(valor) AS total FROM dados GROUP BY categoria ORDER BY total DESC LIMIT 10
    """
    import duckdb
    import pandas as pd

    state = _load_state()
    csv_path = state.get("csv_path")

    if not csv_path or not os.path.exists(csv_path):
        return {"erro": "Nenhum CSV carregado na sessão atual."}

    try:
        df  = pd.read_parquet(csv_path)   # app.py salva como parquet temporário
        con = duckdb.connect(":memory:")
        con.register("dados", df)
        resultado = con.execute(sql).df()
        con.close()
        return {"linhas": resultado.to_dict(orient="records"), "shape": list(resultado.shape)}
    except Exception as e:
        log.warning(f"[MCP] query_csv erro: {e}")
        return {"erro": str(e)}


# ── Tool 2: search_pdf ────────────────────────────────────────────────────────

@mcp.tool()
def search_pdf(query: str, k: int = 4) -> list[str]:
    """
    Busca semântica no PDF vetorizado da sessão atual.
    Retorna os k trechos mais relevantes para a query.
    Use para perguntas conceituais, qualitativas ou que requerem leitura do documento.

    Exemplos de query:
        "quais são os riscos mencionados"
        "metodologia utilizada"
        "conclusões do relatório"
    """
    state = _load_state()
    vs_path = state.get("vectorstore_path")

    if not vs_path:
        return ["Nenhum PDF vetorizado na sessão atual."]

    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from config import HF_MODEL_PATH

        embeddings = HuggingFaceEmbeddings(
            model_name=HF_MODEL_PATH,
            model_kwargs={"device": "cpu"},
        )
        vs   = Chroma(persist_directory=vs_path, embedding_function=embeddings)
        docs = vs.similarity_search(query, k=k)
        return [d.page_content for d in docs]
    except Exception as e:
        log.error(f"[MCP] search_pdf erro: {e}")
        return [f"Erro na busca: {e}"]


# ── Tool 3: get_doc_summary ───────────────────────────────────────────────────

@mcp.tool()
def get_doc_summary() -> dict:
    """
    Retorna metadados do documento atualmente carregado na sessão.
    Útil para entender o que está disponível antes de escolher a ferramenta certa.

    Retorna tipo do documento (csv|pdf), nome, e stats básicos.
    """
    state = _load_state()

    if not state:
        return {"tipo": None, "mensagem": "Nenhum documento carregado."}

    summary = {
        "tipo":      state.get("tipo"),          # "csv" | "pdf"
        "nome":      state.get("nome"),
        "tamanho":   state.get("tamanho_bytes"),
    }

    if state.get("tipo") == "csv":
        summary["n_linhas"]  = state.get("n_linhas")
        summary["n_colunas"] = state.get("n_colunas")
        summary["colunas"]   = state.get("colunas", [])

    elif state.get("tipo") == "pdf":
        summary["n_paginas"] = state.get("n_paginas")
        summary["n_chunks"]  = state.get("n_chunks")

    return summary


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")