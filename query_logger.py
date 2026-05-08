"""
query_logger.py — Logger do agente SQL.

Substitui o RetrievalLogger (que era para RAG/PDF).
Agora loga: pergunta do usuário → SQL gerado → resultado ou erro.

Formato: JSONL (uma linha JSON por entrada) — fácil de parsear e analisar depois.
Útil para:
    • Depurar SQL mal gerado pelo LLM
    • Entender quais perguntas o usuário faz com mais frequência
    • Auditar o que foi consultado no banco

Arquivo de log padrão: query_logs.jsonl (na raiz do projeto)
Em Docker: montar volume para persistir fora do container.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class QueryLogger:
    """
    Logger de queries do agente SQL.

    Uso:
        logger = QueryLogger()
        logger.log(
            pergunta="Qual a Selic hoje?",
            sql="SELECT selic_pct_ano FROM selic ORDER BY data DESC LIMIT 1",
            resultado_shape=(1, 1),   # (linhas, colunas) do DataFrame
            erro=None,
            modo="especialista"       # "especialista" ou "analista"
        )
    """

    def __init__(self, log_file: str = "query_logs.jsonl"):
        self.log_file = Path(log_file)

    def log(
        self,
        pergunta: str,
        sql: str,
        resultado_shape: tuple[int, int] | None = None,
        erro: str | None = None,
        modo: str = "especialista",
    ):
        """
        Registra uma query do agente SQL.

        Parâmetros
        ----------
        pergunta        : texto original do usuário
        sql             : SQL gerado pelo LLM
        resultado_shape : tupla (n_linhas, n_colunas) do DataFrame resultado
        erro            : mensagem de erro se a query falhou, None se sucesso
        modo            : "especialista" (dados BCB) ou "analista" (CSV do usuário)
        """
        entrada = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "modo":      modo,
            "pergunta":  pergunta,
            "sql":       sql,
            "sucesso":   erro is None,
            "resultado": {
                "linhas":  resultado_shape[0] if resultado_shape else None,
                "colunas": resultado_shape[1] if resultado_shape else None,
            },
            "erro": erro,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        except Exception as e:
            # Log de log não pode travar o app — apenas avisa no terminal
            log.warning(f"Falha ao escrever no log de queries: {e}")

    def ler_recentes(self, n: int = 20) -> list[dict]:
        """
        Lê as últimas N entradas do log.
        Útil para debug na sidebar do Streamlit (futuramente).
        """
        if not self.log_file.exists():
            return []
        try:
            linhas = self.log_file.read_text(encoding="utf-8").strip().splitlines()
            entradas = [json.loads(l) for l in linhas if l.strip()]
            return entradas[-n:]
        except Exception as e:
            log.warning(f"Falha ao ler log de queries: {e}")
            return []