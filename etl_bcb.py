"""
etl_bcb.py — Coleta e transforma dados macroeconômicos do Banco Central do Brasil.

Fluxo:
    API BCB (requests) → pandas (transformação) → DuckDB (persistência em disco)

Tabelas criadas/atualizadas no finance.db:
    • selic   — Meta da taxa SELIC por período de vigência
    • ipca    — Variação mensal do IPCA (índice de inflação)
    • cdi     — Taxa CDI diária acumulada
    • cambio  — Taxa de câmbio BRL/USD (PTAX venda, fechamento diário)

Cada função de coleta é independente: se uma API falhar, as demais continuam.
A função `run_etl()` orquestra tudo e registra a data/hora da última atualização.
"""

import datetime
import logging

import duckdb
import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
DB_PATH = "finance.db"          # DuckDB persiste tudo neste arquivo único
BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
# Séries SGS do Banco Central:
#   11   → SELIC diária (% a.d.)
#   433  → IPCA mensal (variação %)
#   12   → CDI diário (% a.d.)
#   1    → Taxa de câmbio BRL/USD PTAX venda (fechamento)
SERIES = {
    "selic":  11,
    "ipca":   433,
    "cdi":    12,
    "cambio": 1,
}
# Quantos anos de histórico puxar na primeira carga
ANOS_HISTORICO = 5


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNÇÕES DE COLETA — uma por indicador
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_sgs(codigo: int, data_ini: str, data_fim: str) -> pd.DataFrame:
    """
    Busca uma série do SGS (Sistema Gerenciador de Séries Temporais) do BCB.

    Parâmetros
    ----------
    codigo    : código numérico da série no SGS
    data_ini  : data inicial no formato dd/MM/yyyy
    data_fim  : data final no formato dd/MM/yyyy

    Retorna um DataFrame com colunas ['data', 'valor'] ou DataFrame vazio se falhar.
    """
    url = BCB_BASE.format(codigo=codigo)
    params = {
        "formato": "json",
        "dataInicial": data_ini,
        "dataFinal":   data_fim,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        # A API retorna 'data' como string "dd/MM/yyyy" e 'valor' como string
        df["data"]  = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["valor"])
        return df
    except Exception as e:
        log.error(f"Erro ao buscar série {codigo}: {e}")
        return pd.DataFrame()


def collect_selic(data_ini: str, data_fim: str) -> pd.DataFrame:
    """
    SELIC diária — taxa over (% ao dia) do Banco Central.
    Série SGS 11. Disponível a partir de 01/06/1986.

    A SELIC diária é a taxa básica de juros da economia.
    Para uso anual: (1 + selic_dia/100)^252 - 1
    """
    log.info("Coletando SELIC...")
    df = _fetch_sgs(SERIES["selic"], data_ini, data_fim)
    if df.empty:
        return df
    df = df.rename(columns={"data": "data", "valor": "selic_pct_dia"})
    # Adiciona coluna de taxa anualizada para facilitar consultas do usuário
    df["selic_pct_ano"] = (1 + df["selic_pct_dia"] / 100) ** 252 - 1
    df["selic_pct_ano"] = (df["selic_pct_ano"] * 100).round(4)
    log.info(f"SELIC: {len(df)} registros coletados.")
    return df


def collect_ipca(data_ini: str, data_fim: str) -> pd.DataFrame:
    """
    IPCA mensal — variação percentual mensal do Índice de Preços ao Consumidor Amplo.
    Série SGS 433. Principal índice de inflação oficial do Brasil.
    """
    log.info("Coletando IPCA...")
    df = _fetch_sgs(SERIES["ipca"], data_ini, data_fim)
    if df.empty:
        return df
    df = df.rename(columns={"data": "data", "valor": "ipca_pct_mes"})
    # Acumulado 12 meses — útil para consultas sobre inflação anual
    df = df.sort_values("data")
    df["ipca_acum_12m"] = (
        (1 + df["ipca_pct_mes"] / 100)
        .rolling(12)
        .apply(lambda x: x.prod() - 1, raw=True) * 100
    ).round(4)
    log.info(f"IPCA: {len(df)} registros coletados.")
    return df


def collect_cdi(data_ini: str, data_fim: str) -> pd.DataFrame:
    """
    CDI diário — Certificado de Depósito Interbancário (% ao dia).
    Série SGS 12. Referência para investimentos em renda fixa (CDBs, LCIs, etc).
    """
    log.info("Coletando CDI...")
    df = _fetch_sgs(SERIES["cdi"], data_ini, data_fim)
    if df.empty:
        return df
    df = df.rename(columns={"data": "data", "valor": "cdi_pct_dia"})
    df["cdi_pct_ano"] = (1 + df["cdi_pct_dia"] / 100) ** 252 - 1
    df["cdi_pct_ano"] = (df["cdi_pct_ano"] * 100).round(4)
    log.info(f"CDI: {len(df)} registros coletados.")
    return df


def collect_cambio(data_ini: str, data_fim: str) -> pd.DataFrame:
    """
    Câmbio BRL/USD — taxa PTAX de venda (fechamento diário).
    Série SGS 1. Publicada diariamente pelo Banco Central.
    """
    log.info("Coletando Câmbio BRL/USD...")
    df = _fetch_sgs(SERIES["cambio"], data_ini, data_fim)
    if df.empty:
        return df
    df = df.rename(columns={"data": "data", "valor": "brl_usd"})
    # Variação diária em % — útil para perguntas sobre volatilidade
    df = df.sort_values("data")
    df["brl_usd_var_pct"] = df["brl_usd"].pct_change() * 100
    df["brl_usd_var_pct"] = df["brl_usd_var_pct"].round(4)
    log.info(f"Câmbio: {len(df)} registros coletados.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNÇÕES DE PERSISTÊNCIA — gravar no DuckDB
# ══════════════════════════════════════════════════════════════════════════════

def _upsert_table(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, tabela: str, pk: str = "data"):
    """
    Insere ou atualiza registros numa tabela DuckDB.

    Estratégia:
        1. Cria a tabela se não existir (inferindo schema do DataFrame).
        2. Apaga registros com data >= data mínima do novo lote (evita duplicatas).
        3. Insere o novo lote.

    Isso garante idempotência: rodar o ETL duas vezes não duplica dados.

    Parâmetros
    ----------
    con    : conexão DuckDB já aberta
    df     : DataFrame com os dados novos
    tabela : nome da tabela destino
    pk     : coluna de data usada como chave de controle de upsert
    """
    if df.empty:
        log.warning(f"DataFrame vazio para tabela '{tabela}'. Pulando.")
        return

    # Cria a tabela a partir do schema do DataFrame na primeira execução
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {tabela} AS
        SELECT * FROM df WHERE 1=0
    """)

    # Remove registros que serão substituídos (upsert por range de data)
    data_min = df[pk].min()
    con.execute(f"DELETE FROM {tabela} WHERE {pk} >= '{data_min}'")

    # Insere os novos registros
    con.execute(f"INSERT INTO {tabela} SELECT * FROM df")
    log.info(f"Tabela '{tabela}' atualizada com {len(df)} registros.")


def _save_last_update(con: duckdb.DuckDBPyConnection):
    """
    Salva o timestamp da última execução do ETL numa tabela de metadados.
    Usado pela sidebar do Streamlit para exibir "Última atualização: ...".
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS etl_metadata (
            chave   VARCHAR PRIMARY KEY,
            valor   VARCHAR
        )
    """)
    agora = datetime.datetime.now().isoformat(timespec="seconds")
    con.execute("""
        INSERT OR REPLACE INTO etl_metadata (chave, valor)
        VALUES ('ultima_atualizacao', ?)
    """, [agora])
    log.info(f"Metadata salva: ultima_atualizacao = {agora}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ORQUESTRADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_etl(anos: int = ANOS_HISTORICO):
    """
    Executa o ETL completo: coleta todos os indicadores e persiste no DuckDB.

    Parâmetros
    ----------
    anos : janela histórica em anos a partir de hoje (padrão: 5 anos)

    Cada indicador é coletado de forma independente. Se um falhar,
    os demais continuam normalmente.
    """
    hoje    = datetime.date.today()
    inicio  = hoje - datetime.timedelta(days=anos * 365)
    data_ini = inicio.strftime("%d/%m/%Y")
    data_fim = hoje.strftime("%d/%m/%Y")

    log.info(f"Iniciando ETL — período: {data_ini} → {data_fim}")

    # Coleta todos os indicadores
    coletas = {
        "selic":  collect_selic(data_ini, data_fim),
        "ipca":   collect_ipca(data_ini, data_fim),
        "cdi":    collect_cdi(data_ini, data_fim),
        "cambio": collect_cambio(data_ini, data_fim),
    }

    # Persiste no DuckDB — uma única conexão para toda a operação
    with duckdb.connect(DB_PATH) as con:
        for tabela, df in coletas.items():
            _upsert_table(con, df, tabela)
        _save_last_update(con)

    log.info("ETL finalizado com sucesso.")


def get_last_update() -> str | None:
    """
    Lê o timestamp da última execução do ETL.
    Retorna string ISO ou None se nunca executado.
    Chamado pela sidebar do Streamlit.
    """
    try:
        with duckdb.connect(DB_PATH, read_only=True) as con:
            result = con.execute(
                "SELECT valor FROM etl_metadata WHERE chave = 'ultima_atualizacao'"
            ).fetchone()
            return result[0] if result else None
    except Exception:
        return None


# ── Execução direta (teste / cron) ────────────────────────────────────────────
# python etl_bcb.py  → roda o ETL e sai
if __name__ == "__main__":
    run_etl()