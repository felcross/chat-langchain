"""
app.py — Interface principal do FinanceBot BR.

Dois modos de uso:
    📚 Especialista — chat sobre dados macro BR (SELIC, IPCA, CDI, Câmbio)
                      Fonte: API do Banco Central → finance.db (DuckDB persistido)
    📊 Analista     — análise de CSV com sugestões automáticas geradas por IA
                      Fonte: arquivo do usuário → DuckDB em memória (:memory:)

Os dois bancos são completamente separados:
    finance.db  → dados BCB, persistido em disco, read-only nas queries
    :memory:    → dados do CSV do usuário, descartado ao fim da sessão

Fluxo Modo Especialista:
    pergunta → cache? → gerar_sql() → DuckDB (finance.db) → formatar_resposta()

Fluxo Modo Analista:
    upload CSV → analisar_csv() → painel de contexto
    → gerar_sugestoes() → botões clicáveis
    → executar_sugestao() → tabela + gráfico (se viável)
"""

import logging
import os

import pandas as pd
import streamlit as st
from decouple import config
from langchain_groq import ChatGroq

import cache_manager
import db_manager
from analista import (
    analisar_csv,
    build_schema_llm,
    executar_sugestao,
    gerar_sugestoes,
    renderizar_resultado,
)
from etl_bcb import get_last_update, run_etl
from query_logger import QueryLogger

# ── Configuração ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"

query_logger = QueryLogger()


# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM
# ══════════════════════════════════════════════════════════════════════════════

def get_llm() -> ChatGroq:
    """Instancia o LLM Groq. temperature=0 para SQL determinístico."""
    return ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=1024)


# ══════════════════════════════════════════════════════════════════════════════
# 2. AGENTE SQL — MODO ESPECIALISTA
# ══════════════════════════════════════════════════════════════════════════════

def gerar_sql(pergunta: str, schema: str, historico: list[dict]) -> str:
    """
    Estágio 1: LLM transforma pergunta em SQL puro.
    Recebe schema das tabelas BCB + últimas 4 trocas do histórico.
    Retorna SQL limpo sem markdown.
    """
    llm = get_llm()

    historico_txt = ""
    for msg in historico[-4:]:
        papel = "Usuário" if msg["role"] == "user" else "Assistente"
        historico_txt += f"{papel}: {msg['content']}\n"

    prompt = f"""Você é um especialista em SQL e dados macroeconômicos do Brasil.
Gere APENAS o SQL para responder à pergunta abaixo. Sem explicação, sem markdown.
Retorne somente o SQL puro, começando com SELECT.

{schema}

Histórico recente:
{historico_txt if historico_txt else "(sem histórico)"}

Pergunta: {pergunta}

SQL:"""

    resposta = llm.invoke(prompt)
    sql = resposta.content.strip()

    # Remove markdown se o modelo insistir
    if sql.startswith("```"):
        sql = "\n".join(
            l for l in sql.splitlines()
            if not l.strip().startswith("```")
        ).strip()

    return sql


def formatar_resposta(pergunta: str, df_resultado: pd.DataFrame) -> str:
    """
    Estágio 2: LLM transforma resultado da query em resposta humana em PT-BR.
    Recebe pergunta original + dados (até 30 linhas).
    """
    llm = get_llm()

    dados_txt = (
        df_resultado.head(30).to_string(index=False)
        if not df_resultado.empty
        else "A query não retornou resultados."
    )

    prompt = f"""Você é um analista financeiro especializado no mercado brasileiro.
Responda à pergunta com base nos dados abaixo.
Use linguagem clara, objetiva e profissional. Formate em markdown.
Explique o significado dos números para o usuário comum.
Responda SEMPRE em português do Brasil.
Não mencione SQL, DuckDB ou detalhes técnicos.

Pergunta: {pergunta}

Dados:
{dados_txt}

Resposta:"""

    resposta = llm.invoke(prompt)
    return resposta.content.strip()


def responder_especialista(
    pergunta: str,
    historico: list[dict],
) -> tuple[str, str | None]:
    """
    Orquestra o agente SQL do Modo Especialista.

    Fluxo:
        cache hit?  → retorna imediatamente
        gerar_sql() → execute_sql() → formatar_resposta()
        → salva cache → loga query

    Retorna (resposta_final, sql_gerado).
    """
    cached = cache_manager.get_cached(pergunta, "especialista")
    if cached:
        return cached, "(resposta do cache)"

    schema   = db_manager.SCHEMA_DESCRICAO
    sql      = gerar_sql(pergunta, schema, historico)
    df, erro = db_manager.execute_sql(sql)

    # Retry com feedback do erro
    if erro:
        log.warning(f"SQL com erro, tentando corrigir: {erro}")
        sql = gerar_sql(
            f"{pergunta}\n\n[ERRO anterior: {erro}. Corrija o SQL.]",
            schema,
            historico,
        )
        df, erro = db_manager.execute_sql(sql)

        if erro:
            query_logger.log(pergunta=pergunta, sql=sql, erro=erro, modo="especialista")
            return f"❌ Não consegui responder. Erro técnico: `{erro}`", sql

    resposta = formatar_resposta(pergunta, df)

    cache_manager.set_cached(pergunta, "especialista", resposta)
    query_logger.log(
        pergunta=pergunta,
        sql=sql,
        resultado_shape=df.shape if df is not None else None,
        modo="especialista",
    )

    return resposta, sql


# ══════════════════════════════════════════════════════════════════════════════
# 3. SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="FinanceBot BR", page_icon="📊", layout="wide")

if "messages_esp" not in st.session_state:
    st.session_state.messages_esp = []

if "df_csv" not in st.session_state:
    st.session_state.df_csv = None

if "contexto_csv" not in st.session_state:
    st.session_state.contexto_csv = None

if "sugestoes" not in st.session_state:
    st.session_state.sugestoes = []

if "sugestao_ativa" not in st.session_state:
    st.session_state.sugestao_ativa = None

if "arquivo_nome" not in st.session_state:
    st.session_state.arquivo_nome = None

if "show_sql" not in st.session_state:
    st.session_state.show_sql = False

if "modo_anterior" not in st.session_state:
    st.session_state.modo_anterior = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE LIMPEZA
# ══════════════════════════════════════════════════════════════════════════════

def limpar_memoria_csv():
    """Remove completamente os dados do modo Analista da sessão."""
    st.session_state.df_csv = None
    st.session_state.contexto_csv = None
    st.session_state.sugestoes = []
    st.session_state.sugestao_ativa = None
    st.session_state.arquivo_nome = None


def limpar_conversa_especialista():
    """Limpa histórico do chat Especialista."""
    st.session_state.messages_esp = []


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.title("📊 FinanceBot BR")
    st.caption("Análise macroeconômica e de dados financeiros")
    st.divider()

    modo = st.radio(
        "Modo de análise",
        options=["📚 Especialista", "📊 Analista"],
        help=(
            "**Especialista**: chat sobre SELIC, IPCA, CDI e Câmbio (BCB).\n\n"
            "**Analista**: envie seu CSV e receba análises automáticas com IA."
        ),
    )

    modo_key = "especialista" if "Especialista" in modo else "analista"

    # Limpa memória ao trocar de modo
    if (
        st.session_state.modo_anterior
        and st.session_state.modo_anterior != modo_key
    ):
        if modo_key == "especialista":
            limpar_memoria_csv()
        elif modo_key == "analista":
            limpar_conversa_especialista()

    st.session_state.modo_anterior = modo_key

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR ESPECIALISTA
    # ══════════════════════════════════════════════════════════════════════════

    if modo_key == "especialista":

        st.subheader("🏦 Dados do Banco Central")

        ultima = get_last_update()

        if ultima:
            st.caption(f"🕐 Última atualização: {ultima.replace('T', ' ')}")
        else:
            st.caption("⚠️ Banco não populado. Clique em Atualizar.")

        # ✅ CORRIGIDO: width="stretch" → use_container_width=True
        if st.button(
            "🔄 Atualizar dados do BCB",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Coletando dados do Banco Central..."):
                try:
                    run_etl()
                    cache_manager.invalidar_cache()
                    st.success("✅ Dados atualizados!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Falha: {e}")

        st.divider()

        health = db_manager.db_health()

        if health.get("ok"):
            snap = db_manager.get_snapshot_atual()
            if snap:
                col1, col2 = st.columns(2)
                col1.metric("SELIC a.a.", f"{snap.get('selic_ano', '—')}%",
                            help=f"Ref: {snap.get('selic_data', '')}")
                col2.metric("IPCA (mês)", f"{snap.get('ipca_mes', '—')}%",
                            help=f"Ref: {snap.get('ipca_data', '')}")
                col1.metric("CDI a.a.",   f"{snap.get('cdi_ano', '—')}%",
                            help=f"Ref: {snap.get('cdi_data', '')}")
                col2.metric("BRL/USD",    f"R$ {snap.get('cambio', '—')}",
                            help=f"Ref: {snap.get('cambio_data', '')}")
        else:
            st.info("📌 Atualize os dados para ver os indicadores.")

        st.divider()
        st.subheader("⚙️ Opções")

        # ✅ CORRIGIDO: key único para o toggle do Especialista
        st.session_state.show_sql = st.toggle(
            "🔍 Mostrar SQL gerado",
            value=st.session_state.show_sql,
            key="toggle_sql_esp",
        )

        if st.button("🗑️ Limpar conversa", use_container_width=True, key="btn_limpar_esp"):
            limpar_conversa_especialista()
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR ANALISTA
    # ══════════════════════════════════════════════════════════════════════════

    else:

        st.subheader("📁 Seu CSV")

        arquivo = st.file_uploader(
            "Envie um arquivo CSV",
            type=["csv"],
            help=(
                "Carregado em memória — não salvo no servidor. "
                "Nunca toca o banco do BCB."
            ),
        )

        if arquivo:
            try:
                # Evita reprocessar o mesmo arquivo a cada rerun
                if st.session_state.arquivo_nome != arquivo.name:
                    df_bruto = pd.read_csv(arquivo)

                    with st.spinner("🔍 Analisando estrutura do CSV..."):
                        contexto = analisar_csv(df_bruto)

                    st.session_state.df_csv         = contexto["df_convertido"]
                    st.session_state.contexto_csv   = contexto
                    st.session_state.sugestoes      = []
                    st.session_state.sugestao_ativa = None
                    st.session_state.arquivo_nome   = arquivo.name

                st.success(
                    f"✅ {st.session_state.contexto_csv['n_linhas']:,} linhas · "
                    f"{st.session_state.contexto_csv['n_colunas']} colunas"
                )

            except Exception as e:
                st.error(f"❌ Erro ao ler CSV: {e}")

        st.divider()
        st.subheader("⚙️ Opções")

        # ✅ CORRIGIDO: key único para o toggle do Analista
        st.session_state.show_sql = st.toggle(
            "🔍 Mostrar SQL gerado",
            value=st.session_state.show_sql,
            key="toggle_sql_ana",
            help="Exibe o SQL executado em cada análise.",
        )

        if st.button("🗑️ Limpar CSV da memória", use_container_width=True, key="btn_limpar_ana"):
            limpar_memoria_csv()
            st.rerun()

    # ── FIM DA SIDEBAR ─────────────────────────────────────────────────────────
    # ✅ REMOVIDO: bloco duplicado de "⚙️ Opções" que estava aqui causando o erro


# ══════════════════════════════════════════════════════════════════════════════
# 5. ÁREA PRINCIPAL — MODO ESPECIALISTA
# ══════════════════════════════════════════════════════════════════════════════

if modo_key == "especialista":

    st.header("📚 Modo Especialista — Dados Macro BR")
    st.caption("Perguntas sobre SELIC, IPCA, CDI e Câmbio. Fonte: Banco Central do Brasil.")

    historico = st.session_state.messages_esp

    for msg in historico:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and st.session_state.show_sql and msg.get("sql"):
                with st.expander("🔍 SQL gerado"):
                    st.code(msg["sql"], language="sql")

    esp_sem_dados = not db_manager.db_health().get("ok")
    if esp_sem_dados:
        st.warning("⚠️ Clique em **Atualizar dados do BCB** na sidebar antes de perguntar.")

    pergunta = st.chat_input(
        "Ex: Qual a Selic hoje? Como o IPCA evoluiu em 2024?",
        disabled=esp_sem_dados,
    )

    if pergunta:
        with st.chat_message("user"):
            st.markdown(pergunta)
        historico.append({"role": "user", "content": pergunta})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Consultando dados..."):
                resposta, sql = responder_especialista(pergunta, historico)
            st.markdown(resposta)
            if st.session_state.show_sql and sql:
                with st.expander("🔍 SQL gerado"):
                    st.code(sql, language="sql")

        historico.append({"role": "assistant", "content": resposta, "sql": sql})
        st.session_state.messages_esp = historico


# ══════════════════════════════════════════════════════════════════════════════
# 6. ÁREA PRINCIPAL — MODO ANALISTA
# ══════════════════════════════════════════════════════════════════════════════

else:

    st.header("📊 Modo Analista — Seus Dados")

    contexto  = st.session_state.get("contexto_csv")
    df_csv    = st.session_state.get("df_csv")
    sugestoes = st.session_state.get("sugestoes", [])

    if contexto is None:
        st.info("📌 Envie um arquivo CSV na sidebar para começar.")
        st.stop()

    # ── PAINEL DE CONTEXTO ─────────────────────────────────────────────────────
    with st.expander("🔍 O que encontramos no seu CSV", expanded=True):

        c1, c2 = st.columns(2)
        c1.metric("Linhas",  f"{contexto['n_linhas']:,}")
        c2.metric("Colunas", f"{contexto['n_colunas']}")
        st.divider()

        rows = []
        for col, info in contexto["colunas"].items():
            tipo = info["tipo"].upper()
            if tipo == "DATA":
                detalhe = f"{info['min']} → {info['max']}"
            elif tipo == "NUMERICA":
                detalhe = f"min {info['min']} · max {info['max']} · média {info['media']}"
            elif tipo == "CATEGORICA":
                vals = ", ".join(info["valores"][:5])
                detalhe = f"{vals}{'...' if info['truncado'] else ''}"
            else:
                detalhe = f"{info['n_unicos']} valores únicos"

            rows.append({
                "Coluna":  col,
                "Tipo":    tipo,
                "Detalhe": detalhe,
                "Nulos":   info["nulos"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("👀 Primeiras linhas do CSV"):
            st.dataframe(df_csv.head(10), use_container_width=True)

    # ── SUGESTÕES ──────────────────────────────────────────────────────────────
    st.subheader("💡 Análises sugeridas")

    gerar_col, _ = st.columns([1, 3])
    with gerar_col:
        label_btn = "✨ Gerar sugestões" if not sugestoes else "🔄 Regenerar sugestões"
        if st.button(label_btn, type="primary", use_container_width=True):
            with st.spinner("🤔 Gerando análises personalizadas para o seu CSV..."):
                novas = gerar_sugestoes(contexto, get_llm())
                st.session_state.sugestoes      = novas
                st.session_state.sugestao_ativa = None
                sugestoes = novas

            if not sugestoes:
                st.warning("⚠️ Não foi possível gerar sugestões. Verifique se o CSV tem dados suficientes.")

    # Grid 2x2 de botões
    if sugestoes:
        st.caption("Clique em uma análise para executá-la:")
        cols = st.columns(2)
        for i, sug in enumerate(sugestoes):
            with cols[i % 2]:
                ativo = st.session_state.sugestao_ativa == i
                label = f"{'▶ ' if ativo else ''}{sug['titulo']}"
                if st.button(label, key=f"sug_{i}", use_container_width=True):
                    st.session_state.sugestao_ativa = i
                    st.rerun()

    # ── RESULTADO DA SUGESTÃO SELECIONADA ──────────────────────────────────────
    idx_ativo = st.session_state.get("sugestao_ativa")

    if idx_ativo is not None and sugestoes:
        sug = sugestoes[idx_ativo]

        st.divider()
        st.subheader(f"📈 {sug['titulo']}")
        st.caption(sug["descricao"])

        with st.spinner("Consultando dados..."):
            df_resultado, erro = executar_sugestao(sug, df_csv)

        if erro:
            st.error(f"❌ Erro ao executar análise: `{erro}`")
        else:
            renderizar_resultado(sug, df_resultado)

        if st.session_state.get("show_sql"):
            with st.expander("🔍 SQL gerado"):
                st.code(sug["sql"], language="sql")