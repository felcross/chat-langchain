"""
analista.py — Módulo completo do Modo Analista.

Responsabilidades:
    1. analisar_csv()         → lê o DataFrame e gera um contexto rico (tipos, stats, únicos)
    2. build_schema_llm()     → transforma o contexto em texto para o LLM gerar SQL correto
    3. gerar_sugestoes()      → chama o Groq com schema + amostra → retorna 4 sugestões em JSON
    4. executar_sugestao()    → roda o SQL no DuckDB em memória → retorna DataFrame
    5. renderizar_resultado() → tabela sempre + gráfico se viável

Banco de dados:
    DuckDB em memória (:memory:) — os dados do CSV NUNCA tocam o finance.db.
    A conexão é aberta e fechada a cada execução de SQL (stateless por design).
"""

import json
import logging
import re

import duckdb
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq

log = logging.getLogger(__name__)

MAX_CATEGORIAS = 15


# ══════════════════════════════════════════════════════════════════════════════
# 1. ANÁLISE DO CSV — gera contexto rico
# ══════════════════════════════════════════════════════════════════════════════

def _detectar_e_converter_datas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tenta converter colunas de texto que parecem datas.
    Formatos tentados: dd/mm/yyyy → yyyy-mm-dd → dd-mm-yyyy → mm/dd/yyyy → inferência.
    Só converte se >= 80% dos valores da amostra forem datas válidas.
    """
    formatos = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        amostra = df[col].dropna().head(20).astype(str)
        convertido = False

        for fmt in formatos:
            try:
                convertida = pd.to_datetime(amostra, format=fmt, errors="coerce")
                if convertida.notna().mean() >= 0.8:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
                    convertido = True
                    log.info(f"Coluna '{col}' convertida para data com formato '{fmt}'")
                    break
            except Exception:
                continue

        if not convertido:
            try:
                tentativa = pd.to_datetime(amostra, errors="coerce")
                if tentativa.notna().mean() >= 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    log.info(f"Coluna '{col}' convertida via inferência automática")
            except Exception:
                pass

    return df


def _stats_numerica(serie: pd.Series) -> dict:
    s = serie.dropna()
    return {
        "min":   round(float(s.min()), 4)  if not s.empty else None,
        "max":   round(float(s.max()), 4)  if not s.empty else None,
        "media": round(float(s.mean()), 4) if not s.empty else None,
        "nulos": int(serie.isna().sum()),
    }


def _stats_categorica(serie: pd.Series) -> dict:
    unicos = serie.dropna().unique().tolist()
    return {
        "n_unicos": len(unicos),
        "valores":  [str(v) for v in unicos[:MAX_CATEGORIAS]],
        "truncado": len(unicos) > MAX_CATEGORIAS,
        "nulos":    int(serie.isna().sum()),
    }


def _stats_data(serie: pd.Series) -> dict:
    s = serie.dropna()
    if s.empty:
        return {"min": None, "max": None, "nulos": int(serie.isna().sum())}
    fmt = lambda v: str(v.date()) if hasattr(v, "date") else str(v)
    return {"min": fmt(s.min()), "max": fmt(s.max()), "nulos": int(serie.isna().sum())}


def analisar_csv(df: pd.DataFrame) -> dict:
    """
    Analisa o DataFrame e retorna contexto estruturado com:
        df_convertido : DataFrame com datas convertidas
        n_linhas / n_colunas
        colunas       : dict por coluna com tipo e stats
        tem_data / tem_numerica
        colunas_data / colunas_num / colunas_cat
    """
    df = _detectar_e_converter_datas(df)

    ctx: dict = {
        "df_convertido": df,
        "n_linhas":      len(df),
        "n_colunas":     len(df.columns),
        "colunas":       {},
        "colunas_data":  [],
        "colunas_num":   [],
        "colunas_cat":   [],
    }

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            ctx["colunas"][col] = {"tipo": "data", **_stats_data(df[col])}
            ctx["colunas_data"].append(col)

        elif pd.api.types.is_numeric_dtype(dtype):
            ctx["colunas"][col] = {"tipo": "numerica", **_stats_numerica(df[col])}
            ctx["colunas_num"].append(col)

        else:
            stats = _stats_categorica(df[col])
            tipo  = "categorica" if stats["n_unicos"] <= MAX_CATEGORIAS else "texto"
            ctx["colunas"][col] = {"tipo": tipo, **stats}
            ctx["colunas_cat"].append(col)

    ctx["tem_data"]     = len(ctx["colunas_data"]) > 0
    ctx["tem_numerica"] = len(ctx["colunas_num"])  > 0
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCHEMA PARA O LLM
# ══════════════════════════════════════════════════════════════════════════════

def build_schema_llm(contexto: dict) -> str:
    """
    Transforma o contexto em texto descritivo para o LLM.
    Inclui tipos reais, ranges, valores únicos e nulos.
    Quanto mais preciso, menos o LLM inventa colunas ou valores errados.
    """
    linhas = [
        "Você tem acesso a uma tabela DuckDB chamada 'dados':\n",
        f"  Linhas  : {contexto['n_linhas']:,}",
        f"  Colunas : {contexto['n_colunas']}\n",
        "  Colunas:",
    ]

    for col, info in contexto["colunas"].items():
        tipo = info["tipo"]
        if tipo == "data":
            linhas.append(f"    - {col}  [DATA]  de {info['min']} até {info['max']}  |  {info['nulos']} nulos")
        elif tipo == "numerica":
            linhas.append(f"    - {col}  [NÚMERO]  min={info['min']}  max={info['max']}  média={info['media']}  |  {info['nulos']} nulos")
        elif tipo == "categorica":
            vals  = ", ".join(f'"{v}"' for v in info["valores"])
            extra = "  (+ outros)" if info["truncado"] else ""
            linhas.append(f"    - {col}  [CATEGORIA]  {info['n_unicos']} valores: {vals}{extra}  |  {info['nulos']} nulos")
        else:
            linhas.append(f"    - {col}  [TEXTO]  {info['n_unicos']} valores únicos  |  {info['nulos']} nulos")

    linhas += [
        "",
        "  Regras SQL:",
        "    1. Tabela se chama exatamente 'dados'.",
        "    2. Use nomes de colunas exatamente como listados.",
        "    3. Valores de CATEGORIA são case-sensitive — use exatamente como listado.",
        "    4. Para DATA use CAST(col AS DATE) se necessário.",
        "    5. Retorne no máximo 50 linhas. Use ORDER BY + LIMIT.",
        "    6. Nunca invente colunas.",
    ]

    return "\n".join(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GERAÇÃO DE SUGESTÕES VIA LLM
# ══════════════════════════════════════════════════════════════════════════════

def gerar_sugestoes(contexto: dict, llm: ChatGroq) -> list[dict]:
    """
    Chama o Groq e retorna 4 sugestões de análise em JSON estruturado.

    Cada sugestão:
        titulo    : str       — nome curto
        descricao : str       — o que revela
        sql       : str       — SQL pronto para DuckDB
        grafico   : str       — "line" | "bar" | "none"
        x         : str|None  — coluna eixo X
        y         : str|None  — coluna eixo Y

    Retorna lista vazia se o LLM falhar — sem crash.
    """
    schema  = build_schema_llm(contexto)
    df      = contexto["df_convertido"]
    amostra = df.head(5).to_string(index=False)

    prompt = f"""Você é um analista de dados especialista em SQL e DuckDB.
Analise o schema e a amostra abaixo e gere EXATAMENTE 4 sugestões de análise relevantes.

{schema}

Amostra (5 primeiras linhas):
{amostra}

Retorne APENAS um array JSON válido. Sem markdown, sem texto antes ou depois.
Cada objeto deve ter exatamente estes campos:
{{
  "titulo":    "nome curto",
  "descricao": "o que essa análise revela",
  "sql":       "SELECT ... FROM dados ... LIMIT 50",
  "grafico":   "line" | "bar" | "none",
  "x":         "coluna_x ou null",
  "y":         "coluna_y ou null"
}}

Regras críticas:
- "line" SOMENTE se X for coluna de DATA e Y for numérica.
- "bar"  SOMENTE se X for CATEGORIA e Y for numérica.
- "none" em qualquer dúvida — melhor "none" do que gráfico errado.
- x e y devem ser EXATAMENTE os aliases/nomes que aparecem no SELECT resultante.
- Varie os tipos de análise: não repita o mesmo agrupamento 4 vezes.
- SQL deve funcionar em DuckDB — sem funções exclusivas de MySQL/PostgreSQL.

JSON:"""

    try:
        resposta = llm.invoke(prompt)
        texto    = resposta.content.strip()
        texto    = re.sub(r"```(?:json)?", "", texto).replace("```", "").strip()

        sugestoes = json.loads(texto)

        campos = {"titulo", "descricao", "sql", "grafico", "x", "y"}
        validas = []
        for s in sugestoes:
            if isinstance(s, dict) and campos.issubset(s.keys()):
                if s["grafico"] not in ("line", "bar", "none"):
                    s["grafico"] = "none"
                    s["x"] = None
                    s["y"] = None
                validas.append(s)

        log.info(f"Sugestões geradas: {len(validas)}")
        return validas

    except json.JSONDecodeError as e:
        log.error(f"JSON inválido do LLM: {e}")
        return []
    except Exception as e:
        log.error(f"Erro ao gerar sugestões: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXECUÇÃO E RENDERIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def executar_sugestao(sugestao: dict, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Executa o SQL da sugestão no DuckDB em memória.
    O CSV nunca toca o finance.db — conexão descartada após execução.
    """
    sql = sugestao.get("sql", "").strip()
    if not sql:
        return None, "SQL vazio na sugestão."
    try:
        con = duckdb.connect(":memory:")
        con.register("dados", df)
        resultado = con.execute(sql).df()
        con.close()
        return resultado, None
    except Exception as e:
        log.warning(f"Erro ao executar '{sugestao.get('titulo')}': {e}")
        return None, str(e)


def _validar_grafico(sugestao: dict, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Verifica se o gráfico é viável dado o resultado real.
    Retorna (viavel, motivo_se_nao_viavel).
    """
    tipo  = sugestao.get("grafico", "none")
    col_x = sugestao.get("x")
    col_y = sugestao.get("y")

    if tipo == "none":
        return False, ""  # intencional, sem mensagem

    if df is None or df.empty:
        return False, "Sem dados suficientes para gerar o gráfico."

    cols = df.columns.tolist()

    if col_x and col_x not in cols:
        return False, f"Coluna '{col_x}' não encontrada no resultado."

    if col_y and col_y not in cols:
        return False, f"Coluna '{col_y}' não encontrada no resultado."

    if col_y and not pd.api.types.is_numeric_dtype(df[col_y]):
        return False, f"A coluna '{col_y}' não é numérica — não é possível usar no eixo Y."

    if tipo == "line" and len(df) < 3:
        return False, "Pontos insuficientes para gráfico de linha (mínimo 3)."

    if tipo == "bar" and len(df) < 2:
        return False, "Categorias insuficientes para gráfico de barras (mínimo 2)."

    return True, ""


def renderizar_resultado(sugestao: dict, df_resultado: pd.DataFrame):
    """
    Renderiza tabela + gráfico (se viável) no Streamlit.
    A tabela é sempre exibida.
    O gráfico só aparece se passar pela validação — caso contrário,
    exibe mensagem clara e discreta, sem quebrar nada.
    """
    st.dataframe(df_resultado, use_container_width=True)

    viavel, motivo = _validar_grafico(sugestao, df_resultado)

    if not viavel:
        if motivo:
            st.caption(f"📊 Gráfico não disponível: {motivo}")
        return

    col_x = sugestao["x"]
    col_y = sugestao["y"]
    tipo  = sugestao["grafico"]

    try:
        df_plot = df_resultado.set_index(col_x)[[col_y]]
        if tipo == "line":
            st.line_chart(df_plot)
        elif tipo == "bar":
            st.bar_chart(df_plot)
    except Exception as e:
        st.caption(f"📊 Não foi possível renderizar o gráfico: {e}")
        log.warning(f"Erro ao renderizar gráfico: {e}")