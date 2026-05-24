"""
app.py — Interface principal do doc-analista (Streamlit).

Fluxo CSV:
    upload → analisar_csv() → painel de contexto
    → gerar_sugestoes() [Groq + cache Redis]
    → executar_sugestao() [DuckDB :memory:]
    → tabela + gráfico

Fluxo PDF:
    upload → build_vectorstore() [PyMuPDF → clean → embed → Chroma em dir temporário]
    → gerar_insights_pdf() [Groq]
    → painel de insights
    → [fase 2] chat livre

Estado MCP:
    Após processar PDF, grava JSON com vs_dir para que o mcp_server.py acesse.
    limpar_sessao() apaga o vs_dir do disco.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq

import cache_manager
from analista import (
    analisar_csv,
    build_schema_llm,
    executar_sugestao,
    gerar_sugestoes,
    renderizar_resultado,
)
from chains import gerar_insights_pdf, get_base_rag_chain
from config import GROQ_API_KEY, MODEL_NAME
from query_logger import QueryLogger

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

query_logger = QueryLogger()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_llm() -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=1024)


def _file_id(name: str, size: int) -> str:
    """ID determinístico para um arquivo — usado como chave de cache."""
    return hashlib.md5(f"{name}|{size}".encode()).hexdigest()


def _gravar_estado_mcp(state: dict) -> str:
    """
    Grava o estado da sessão em JSON temporário para o mcp_server.py.
    Seta DOC_ANALISTA_STATE com o path do arquivo.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(state, tmp, ensure_ascii=False)
    tmp.close()
    os.environ["DOC_ANALISTA_STATE"] = tmp.name
    return tmp.name


def limpar_sessao() -> None:
    # Apaga o dir do Chroma do disco antes de limpar o session_state
    vs_dir = st.session_state.get("vs_dir")
    if vs_dir and os.path.exists(vs_dir):
        shutil.rmtree(vs_dir, ignore_errors=True)
        log.info(f"Vectorstore temporário removido: {vs_dir}")

    keys = [
        "tipo_arquivo", "arquivo_nome", "arquivo_id",
        "df_csv", "contexto_csv",
        "sugestoes", "sugestao_ativa",
        "vectorstore", "vs_dir", "insights_pdf",
        "show_sql", "session_id",
        "_mcp_state_file",
    ]
    for k in keys:
        st.session_state.pop(k, None)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — inicialização
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Doc Analista", page_icon="🔍", layout="wide")

defaults = {
    "tipo_arquivo":    None,   # "csv" | "pdf"
    "arquivo_nome":    None,
    "arquivo_id":      None,
    "df_csv":          None,
    "contexto_csv":    None,
    "sugestoes":       [],
    "sugestao_ativa":  None,
    "vectorstore":     None,
    "vs_dir":          None,   # path do dir temporário do Chroma
    "insights_pdf":    [],
    "show_sql":        False,
    "session_id":      str(uuid.uuid4()),
    "_mcp_state_file": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔍 Doc Analista")
    st.caption("Análise inteligente de CSV e PDF com IA")
    st.divider()

    arquivo = st.file_uploader(
        "📁 Envie um CSV ou PDF",
        type=["csv", "pdf"],
        help="CSV → análise via DuckDB + SQL  |  PDF → embeddings + RAG",
    )

    if arquivo:
        file_id = _file_id(arquivo.name, arquivo.size)

        # Só reprocessa se for um arquivo diferente
        if st.session_state.arquivo_id != file_id:
            limpar_sessao()
            st.session_state.arquivo_nome = arquivo.name
            st.session_state.arquivo_id   = file_id
            ext = Path(arquivo.name).suffix.lower()

            # ── CSV ────────────────────────────────────────────────────────
            if ext == ".csv":
                st.session_state.tipo_arquivo = "csv"
                try:
                    with st.spinner("🔍 Analisando CSV..."):
                        df_bruto = pd.read_csv(arquivo)
                        contexto = analisar_csv(df_bruto)
                        st.session_state.df_csv       = contexto["df_convertido"]
                        st.session_state.contexto_csv = contexto

                    tmp_parquet = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
                    contexto["df_convertido"].to_parquet(tmp_parquet.name, index=False)

                    _gravar_estado_mcp({
                        "tipo":             "csv",
                        "nome":             arquivo.name,
                        "tamanho_bytes":    arquivo.size,
                        "n_linhas":         contexto["n_linhas"],
                        "n_colunas":        contexto["n_colunas"],
                        "colunas":          list(contexto["colunas"].keys()),
                        "csv_path":         tmp_parquet.name,
                        "vectorstore_path": None,
                    })
                    st.success(
                        f"✅ {contexto['n_linhas']:,} linhas · {contexto['n_colunas']} colunas"
                    )
                except Exception as e:
                    st.error(f"❌ Erro ao ler CSV: {e}")

            # ── PDF ────────────────────────────────────────────────────────
            elif ext == ".pdf":
                st.session_state.tipo_arquivo = "pdf"
                try:
                    with st.spinner("📄 Vetorizando PDF... (pode demorar na primeira vez)"):
                        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                        tmp_pdf.write(arquivo.read())
                        tmp_pdf.close()

                        from vectorstore import build_vectorstore

                        n_chunks, vs, vs_dir = build_vectorstore(tmp_pdf.name)
                        st.session_state.vectorstore = vs
                        st.session_state.vs_dir      = vs_dir

                    _gravar_estado_mcp({
                        "tipo":             "pdf",
                        "nome":             arquivo.name,
                        "tamanho_bytes":    arquivo.size,
                        "n_chunks":         n_chunks,
                        "vectorstore_path": vs_dir,   # MCP acessa via esse path
                        "csv_path":         None,
                    })
                    st.success(f"✅ PDF vetorizado — {n_chunks} chunks")
                except Exception as e:
                    st.error(f"❌ Erro ao processar PDF: {e}")

    st.divider()
    st.subheader("⚙️ Opções")

    st.session_state.show_sql = st.toggle(
        "🔍 Mostrar SQL gerado",
        value=st.session_state.show_sql,
        key="toggle_sql",
        help="Exibe o SQL executado em cada análise do CSV.",
    )

    if st.button("🗑️ Limpar sessão", use_container_width=True):
        limpar_sessao()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.header("🔍 Doc Analista")

tipo = st.session_state.tipo_arquivo

if tipo is None:
    st.info("📌 Envie um arquivo CSV ou PDF na sidebar para começar.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MODO CSV
# ══════════════════════════════════════════════════════════════════════════════

if tipo == "csv":
    contexto  = st.session_state.contexto_csv
    df_csv    = st.session_state.df_csv
    sugestoes = st.session_state.sugestoes

    with st.expander("🔍 Estrutura do CSV", expanded=True):
        c1, c2 = st.columns(2)
        c1.metric("Linhas",  f"{contexto['n_linhas']:,}")
        c2.metric("Colunas", f"{contexto['n_colunas']}")
        st.divider()

        rows = []
        for col, info in contexto["colunas"].items():
            tipo_col = info["tipo"].upper()
            if tipo_col == "DATA":
                detalhe = f"{info['min']} → {info['max']}"
            elif tipo_col == "NUMERICA":
                detalhe = f"min {info['min']} · max {info['max']} · média {info['media']}"
            elif tipo_col == "CATEGORICA":
                vals    = ", ".join(info["valores"][:5])
                detalhe = f"{vals}{'...' if info['truncado'] else ''}"
            else:
                detalhe = f"{info['n_unicos']} valores únicos"
            rows.append({
                "Coluna":  col,
                "Tipo":    tipo_col,
                "Detalhe": detalhe,
                "Nulos":   info["nulos"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("👀 Primeiras linhas"):
        st.dataframe(df_csv.head(10), use_container_width=True)

    st.subheader("💡 Análises sugeridas")
    col_btn, _ = st.columns([1, 3])

    with col_btn:
        label  = "✨ Gerar sugestões" if not sugestoes else "🔄 Regenerar"
        forcar = st.button(label, type="primary", use_container_width=True)

    if forcar:
        if sugestoes:
            cache_manager.invalidar_cache_insights(
                st.session_state.arquivo_nome,
                contexto["n_linhas"],
                list(contexto["colunas"].keys()),
            )
        with st.spinner("🤔 Gerando análises..."):
            novas = gerar_sugestoes(
                contexto,
                get_llm(),
                file_name=st.session_state.arquivo_nome,
                forcar=forcar and bool(sugestoes),
            )
            st.session_state.sugestoes      = novas
            st.session_state.sugestao_ativa = None
            sugestoes = novas

    if not sugestoes:
        if forcar:
            st.warning("⚠️ Não foi possível gerar sugestões. Verifique se o CSV tem dados suficientes.")
    else:
        cached = cache_manager.get_cached_insights(
            st.session_state.arquivo_nome,
            contexto["n_linhas"],
            list(contexto["colunas"].keys()),
        )
        if cached and not forcar:
            st.caption("💾 Sugestões recuperadas do cache")

        st.caption("Clique em uma análise para executá-la:")
        cols = st.columns(2)
        for i, sug in enumerate(sugestoes):
            with cols[i % 2]:
                ativo = st.session_state.sugestao_ativa == i
                if st.button(
                    f"{'▶ ' if ativo else ''}{sug['titulo']}",
                    key=f"sug_{i}",
                    use_container_width=True,
                ):
                    st.session_state.sugestao_ativa = i
                    st.rerun()

    idx = st.session_state.sugestao_ativa
    if idx is not None and sugestoes:
        sug = sugestoes[idx]
        st.divider()
        st.subheader(f"📈 {sug['titulo']}")
        st.caption(sug["descricao"])

        with st.spinner("Consultando dados..."):
            df_resultado, erro = executar_sugestao(sug, df_csv)

        if erro:
            st.error(f"❌ Erro ao executar análise: `{erro}`")
            query_logger.log(modo="csv", sql=sug["sql"], erro=erro)
        else:
            renderizar_resultado(sug, df_resultado)
            query_logger.log(
                modo="csv",
                sql=sug["sql"],
                resultado_shape=df_resultado.shape if df_resultado is not None else None,
            )
            if st.session_state.show_sql:
                with st.expander("🔍 SQL gerado"):
                    st.code(sug["sql"], language="sql")


# ══════════════════════════════════════════════════════════════════════════════
# MODO PDF
# ══════════════════════════════════════════════════════════════════════════════

elif tipo == "pdf":
    vs           = st.session_state.vectorstore
    insights     = st.session_state.insights_pdf
    arquivo_nome = st.session_state.arquivo_nome

    if vs is None:
        st.warning("⚠️ Vectorstore não encontrado. Tente enviar o arquivo novamente.")
        st.stop()

    st.subheader(f"📄 {arquivo_nome}")
    st.subheader("💡 Insights do documento")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        label = "✨ Gerar insights" if not insights else "🔄 Regenerar insights"
        if st.button(label, type="primary", use_container_width=True):
            with st.spinner("🤔 Analisando documento..."):
                novos = gerar_insights_pdf(vs)
                st.session_state.insights_pdf = novos
                insights = novos
            query_logger.log(
                modo="pdf",
                extra={"arquivo": arquivo_nome, "n_insights": len(novos)},
            )

    if not insights:
        st.info("📌 Clique em 'Gerar insights' para analisar o documento.")
    else:
        cols = st.columns(2)
        for i, ins in enumerate(insights):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"**{ins['titulo']}**")
                    st.caption(ins["descricao"])

    st.divider()
    st.caption(
        "💬 Chat livre sobre o documento — em breve. "
        "Faça perguntas diretamente ao documento após os insights."
    )