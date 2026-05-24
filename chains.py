"""
chains.py — RAG chain para análise de PDF.

Adaptado do Chatbot WhatsApp:
  - Remove toda dependência de WhatsApp / buffer / Evolution API
  - Adiciona gerar_insights_pdf() que usa o vectorstore como contexto
    para produzir insights automáticos (equivalente ao gerar_sugestoes() do CSV)
  - Mantém get_conversational_rag_chain() para o chat livre (fase 2)

Fluxo PDF:
    vectorstore → retriever → history_aware_retriever (Groq)
    → stuff_documents_chain (Groq) → resposta em PT-BR
"""

import logging

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever

from config import MODEL_NAME, MODEL_TEMPERATURE, MODEL_MAX_TOKENS
from memory import get_session_history
from prompts import context_prompt, qa_prompt

log = logging.getLogger(__name__)

# Singleton da chain base — recriado quando o vectorstore muda (novo PDF)
_rag_base_chain = None
_current_vectorstore_id: str | None = None  # controla quando recriar


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_llm() -> ChatGroq:
    return ChatGroq(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS,
    )


# ── Chain base (singleton por vectorstore) ────────────────────────────────────

def get_base_rag_chain(vectorstore: Chroma, vectorstore_id: str):
    """
    Inicializa LLM + retriever + chain RAG.
    Recria o singleton apenas quando o vectorstore_id muda
    (ou seja, quando o usuário fez upload de um novo PDF).

    vectorstore_id: qualquer string que identifique unicamente o documento
                    atual (ex: hash do nome + tamanho do arquivo).
    """
    global _rag_base_chain, _current_vectorstore_id

    if _rag_base_chain is not None and _current_vectorstore_id == vectorstore_id:
        return _rag_base_chain

    log.info(f"[CHAINS] Inicializando RAG chain para vectorstore '{vectorstore_id}'...")

    llm       = _build_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    history_aware = create_history_aware_retriever(llm, retriever, context_prompt)
    qa_chain      = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    _rag_base_chain         = create_retrieval_chain(history_aware, qa_chain)
    _current_vectorstore_id = vectorstore_id

    log.info("[CHAINS] RAG chain pronta.")
    return _rag_base_chain


def get_conversational_rag_chain(
    vectorstore: Chroma, vectorstore_id: str
) -> RunnableWithMessageHistory:
    """
    Wrapper com histórico Redis isolado por session_id.
    Usado pelo chat livre (fase 2).
    """
    rag_chain = get_base_rag_chain(vectorstore, vectorstore_id)
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ── Insights automáticos do PDF ───────────────────────────────────────────────

_INSIGHTS_PROMPT = """Você é um analista de documentos especialista.
Analise o conteúdo abaixo (extraído de um PDF) e gere EXATAMENTE 4 insights
relevantes e objetivos sobre o documento.

Conteúdo:
{context}

Para cada insight retorne um objeto JSON com:
{{
  "titulo":    "nome curto do insight",
  "descricao": "explicação clara e objetiva em 2-3 frases"
}}

Retorne APENAS um array JSON válido. Sem markdown, sem texto extra.
Responda SEMPRE em português do Brasil.

JSON:"""


def gerar_insights_pdf(vectorstore: Chroma, n_chunks: int = 8) -> list[dict]:
    """
    Recupera os chunks mais representativos do PDF e pede ao Groq
    para gerar 4 insights automáticos.

    Equivalente ao gerar_sugestoes() do fluxo CSV, mas sem SQL —
    usa busca semântica com query genérica para cobrir o documento.
    """
    import json
    import re

    try:
        # Busca ampla para pegar chunks variados do documento
        docs = vectorstore.similarity_search(
            "resumo geral conteúdo principal pontos importantes",
            k=n_chunks,
        )
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        llm    = _build_llm()
        prompt = _INSIGHTS_PROMPT.format(context=context)

        resposta = llm.invoke(prompt)
        texto    = resposta.content.strip()
        texto    = re.sub(r"```(?:json)?", "", texto).replace("```", "").strip()

        insights = json.loads(texto)
        campos   = {"titulo", "descricao"}
        validos  = [i for i in insights if isinstance(i, dict) and campos.issubset(i.keys())]

        log.info(f"[CHAINS] Insights PDF gerados: {len(validos)}")
        return validos

    except Exception as e:
        log.error(f"[CHAINS] Erro ao gerar insights do PDF: {e}")
        return []