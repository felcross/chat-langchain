"""
query_logger.py — Log de queries e análises para debug.

Vem do FinanceBot BR, sem alteração.
Grava em JSONL — cada linha é um JSON independente, fácil de grep/parse.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

LOG_PATH = Path("query_logs.jsonl")


class QueryLogger:
    def log(
        self,
        *,
        modo: str,                          # "csv" | "pdf"
        pergunta: str | None = None,
        sql: str | None = None,
        erro: str | None = None,
        resultado_shape: tuple | None = None,
        extra: dict | None = None,
    ) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "modo": modo,
        }
        if pergunta:
            entry["pergunta"] = pergunta
        if sql:
            entry["sql"] = sql
        if erro:
            entry["erro"] = erro
        if resultado_shape:
            entry["linhas"], entry["colunas"] = resultado_shape
        if extra:
            entry.update(extra)

        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"[LOGGER] Falha ao gravar log: {e}")