"""
db_manager.py — Interface entre o app e o DuckDB.

Responsabilidades:
    • Abrir conexão (read-only para queries, read-write apenas no ETL)
    • Expor o schema das tabelas para o Groq montar SQL correto
    • Executar SQL gerado pelo LLM de forma segura
    • Retornar resultados como DataFrame ou dict para o app consumir

Por que centralizar aqui?
    Se precisarmos trocar o banco (ex: PostgreSQL na VPS), só mudamos este arquivo.
    O app.py e o agente SQL nem ficam sabendo.
"""

import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "finance.db")

# ── Schema descritivo para o LLM ──────────────────────────────────────────────
# Este texto é injetado no system prompt do agente SQL.
# Quanto mais preciso o schema, melhor o SQL gerado pelo Groq.
# Inclui: nome da tabela, colunas, tipos, unidades e exemplos reais.

SCHEMA_DESCRICAO = """
Você tem acesso a um banco DuckDB com os seguintes indicadores macroeconômicos do Brasil:

TABELA: selic
  Descrição: Taxa SELIC diária publicada pelo Banco Central do Brasil.
  Colunas:
    - data           DATE        — data de referência
    - selic_pct_dia  DOUBLE      — taxa SELIC (% ao dia), ex: 0.0521
    - selic_pct_ano  DOUBLE      — taxa SELIC anualizada (% ao ano), ex: 13.25

TABELA: ipca
  Descrição: Variação mensal do IPCA (inflação oficial do Brasil), IBGE via BCB.
  Colunas:
    - data            DATE   — primeiro dia do mês de referência
    - ipca_pct_mes    DOUBLE — variação mensal (%), ex: 0.44
    - ipca_acum_12m   DOUBLE — IPCA acumulado nos últimos 12 meses (%), ex: 4.62

TABELA: cdi
  Descrição: Taxa CDI diária (Certificado de Depósito Interbancário).
  Colunas:
    - data         DATE   — data de referência
    - cdi_pct_dia  DOUBLE — taxa CDI (% ao dia), ex: 0.0519
    - cdi_pct_ano  DOUBLE — taxa CDI anualizada (% ao ano), ex: 13.15

TABELA: cambio
  Descrição: Taxa de câmbio BRL/USD (PTAX venda, fechamento diário), Banco Central.
  Colunas:
    - data            DATE   — data de referência
    - brl_usd         DOUBLE — reais por 1 dólar americano, ex: 5.12
    - brl_usd_var_pct DOUBLE — variação percentual diária, ex: 0.35

Regras importantes para gerar o SQL:
  1. Sempre filtre por data quando o usuário mencionar um período específico.
  2. Para perguntas sobre "hoje" ou "atual", use MAX(data) como proxy.
  3. Para médias de período, use AVG() com WHERE data BETWEEN ... AND ...
  4. O câmbio só tem dias úteis — não filtre por dia da semana.
  5. O IPCA é mensal — não tente agregar por dia.
  6. Datas no DuckDB: use DATE '2024-01-01' ou CAST('2024-01-01' AS DATE).
  7. Retorne no máximo 50 linhas, a menos que o usuário peça mais.
  8. Use ORDER BY data DESC por padrão.
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONEXÃO
# ══════════════════════════════════════════════════════════════════════════════

def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Retorna uma conexão DuckDB.

    read_only=True  → para queries do agente (mais seguro, bloqueia escritas acidentais)
    read_only=False → para o ETL (necessário para INSERT/CREATE)

    Sempre use como context manager:
        with get_connection() as con:
            con.execute(...)
    """
    return duckdb.connect(DB_PATH, read_only=read_only)


# ══════════════════════════════════════════════════════════════════════════════
# 2. VERIFICAÇÃO DE SAÚDE DO BANCO
# ══════════════════════════════════════════════════════════════════════════════

def db_health() -> dict:
    """
    Verifica se o banco existe e as tabelas estão populadas.

    Retorna dict com status de cada tabela:
        {"selic": 1234, "ipca": 60, "cdi": 1234, "cambio": 1200, "ok": True}

    "ok" = False se qualquer tabela estiver vazia ou ausente.
    Usado pela sidebar do Streamlit para alertar o usuário.
    """
    tabelas = ["selic", "ipca", "cdi", "cambio"]
    resultado = {}
    try:
        with get_connection() as con:
            for t in tabelas:
                try:
                    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    resultado[t] = n
                except Exception:
                    resultado[t] = 0
        resultado["ok"] = all(resultado[t] > 0 for t in tabelas)
    except Exception as e:
        log.error(f"Erro ao verificar health do banco: {e}")
        resultado = {t: 0 for t in tabelas}
        resultado["ok"] = False
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXECUÇÃO DE SQL GERADO PELO LLM
# ══════════════════════════════════════════════════════════════════════════════

def execute_sql(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Executa o SQL gerado pelo Groq no DuckDB.

    Retorna uma tupla:
        (DataFrame com resultado, None)       → sucesso
        (None, mensagem de erro)              → falha

    Por que retornar o erro em vez de lançar exceção?
    O agente SQL pode receber esse erro de volta para tentar corrigir o SQL.

    Proteções básicas:
        - Conexão read-only: impede INSERT, UPDATE, DELETE, DROP acidentais.
        - Timeout implícito da conexão DuckDB.
        - Limite de linhas retornadas (controlado pelo schema + pelo SQL).
    """
    # Whitelist: apenas comandos de leitura são permitidos
    # Mais seguro que blacklist — só aprova o que é explicitamente conhecido
    if not re.match(r'^\s*(SELECT|WITH|SHOW|DESCRIBE|EXPLAIN)\b', sql, re.I):
        return None, "Apenas comandos de leitura (SELECT, WITH) são permitidos."

    try:
        with get_connection(read_only=True) as con:
            df = con.execute(sql).df()
            if len(df) > 100:
                df = df.head(100)
            return df, None
    except duckdb.Error as e:
        log.warning(f"Erro DuckDB ao executar SQL: {e}\nSQL: {sql}")
        return None, str(e)
    except Exception as e:
        log.error(f"Erro inesperado ao executar SQL: {e}")
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSULTAS RÁPIDAS — usadas pelo app sem passar pelo agente SQL
# ══════════════════════════════════════════════════════════════════════════════

def get_snapshot_atual() -> dict:
    """
    Retorna os valores mais recentes de todos os indicadores.
    Usado para exibir métricas rápidas no topo da sidebar do Streamlit.

    Retorna dict:
        {
          "selic_ano": 13.25,   "selic_data": "2024-05-01",
          "ipca_mes": 0.44,     "ipca_data":  "2024-04-01",
          "ipca_acum": 4.62,
          "cdi_ano": 13.15,     "cdi_data":   "2024-05-01",
          "cambio": 5.12,       "cambio_data": "2024-05-02",
        }
    """
    try:
        with get_connection() as con:
            selic  = con.execute("SELECT data, selic_pct_ano FROM selic  ORDER BY data DESC LIMIT 1").fetchone()
            ipca   = con.execute("SELECT data, ipca_pct_mes, ipca_acum_12m FROM ipca ORDER BY data DESC LIMIT 1").fetchone()
            cdi    = con.execute("SELECT data, cdi_pct_ano  FROM cdi    ORDER BY data DESC LIMIT 1").fetchone()
            cambio = con.execute("SELECT data, brl_usd      FROM cambio  ORDER BY data DESC LIMIT 1").fetchone()

        return {
            "selic_ano":   round(selic[1], 2)  if selic  else None,
            "selic_data":  str(selic[0])        if selic  else None,
            "ipca_mes":    round(ipca[1], 2)    if ipca   else None,
            "ipca_acum":   round(ipca[2], 2)    if ipca   else None,
            "ipca_data":   str(ipca[0])         if ipca   else None,
            "cdi_ano":     round(cdi[1], 2)     if cdi    else None,
            "cdi_data":    str(cdi[0])           if cdi    else None,
            "cambio":      round(cambio[1], 4)  if cambio else None,
            "cambio_data": str(cambio[0])        if cambio else None,
        }
    except Exception as e:
        log.error(f"Erro ao buscar snapshot: {e}")
        return {}