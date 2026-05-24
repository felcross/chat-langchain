"""
prompts.py — Templates de prompt para o agente RAG (PDF).

Vem do Chatbot WhatsApp, adaptado para o contexto de analista de documentos.
Os textos em si vêm do config.py (configuráveis via .env).
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import AI_CONTEXTUALIZE_PROMPT, AI_SYSTEM_PROMPT

# Usado pelo create_history_aware_retriever para reformular a pergunta
# levando em conta o histórico antes de buscar no vectorstore.
context_prompt = ChatPromptTemplate.from_messages([
    ("system", AI_CONTEXTUALIZE_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt principal — recebe {context} (chunks do PDF) + histórico + pergunta.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", AI_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])