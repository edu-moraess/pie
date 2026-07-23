"""
Memory & Learning Agent
========================
Constrói memória contextual por usuário:
- Memória SEMÂNTICA (ChromaDB): embeddings de ideias/prompts passados,
  para recuperar sessões similares e reaproveitar o que funcionou.
- Memória ESTRUTURADA (via pie_core.db): preferências de formato, estilo
  de resposta, áreas de interesse e padrões de uso, persistidas em
  SQL (SQLite por padrão, Postgres via DATABASE_URL).

Essas informações alimentam o Prompt Architect Agent em sessões futuras,
como contexto adicional.
"""
from __future__ import annotations

import os
from typing import List, Optional

from pie_core import db
from pie_core.models import PIESession, UserMemory

_CHROMA_PATH = os.getenv("CHROMA_PATH", "./pie_chroma")


class MemoryLearningAgent:
    def __init__(self, persist_path: str = _CHROMA_PATH):
        import chromadb  # lazy import

        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection("pie_sessions")

    # ---------------------------------------------------------------- write

    def store_session(self, session: PIESession) -> None:
        """Guarda a ideia bruta + prompt final como memória semântica, e
        atualiza as preferências estruturadas do usuário."""
        self._collection.upsert(
            ids=[session.session_id],
            documents=[session.raw_idea],
            metadatas=[
                {
                    "user_id": session.user_id,
                    "task_type": session.intent.task_type.value if session.intent else "outro",
                    "final_prompt": session.final_prompt[:2000],
                    "score": session.evaluation.score if session.evaluation else 0,
                }
            ],
        )
        db.record_usage(
            user_id=session.user_id,
            task_type=session.intent.task_type.value if session.intent else "outro",
            recommended_model=session.intent.recommended_model.value if session.intent else None,
        )

    def update_preferences(
        self,
        user_id: str,
        format_preferences: Optional[List[str]] = None,
        response_style: Optional[List[str]] = None,
        interest_areas: Optional[List[str]] = None,
    ) -> UserMemory:
        return db.upsert_user_memory(
            user_id=user_id,
            format_preferences=format_preferences,
            response_style=response_style,
            interest_areas=interest_areas,
        )

    # ----------------------------------------------------------------- read

    def retrieve_similar(self, user_id: str, query: str, k: int = 3) -> List[dict]:
        """Busca sessões passadas semanticamente parecidas com a ideia atual."""
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where={"user_id": user_id},
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"raw_idea": d, **m} for d, m in zip(docs, metas)]

    def get_user_memory(self, user_id: str) -> UserMemory:
        return db.get_user_memory(user_id) or UserMemory(user_id=user_id)

    def build_memory_context(self, user_id: str, query: str) -> str:
        """Gera um bloco de texto pronto para injetar como contexto extra
        no Prompt Architect Agent (personalização)."""
        mem = self.get_user_memory(user_id)
        similar = self.retrieve_similar(user_id, query)

        lines = []
        if mem.format_preferences:
            lines.append(f"Preferências de formato: {', '.join(mem.format_preferences)}")
        if mem.response_style:
            lines.append(f"Estilo de resposta preferido: {', '.join(mem.response_style)}")
        if mem.interest_areas:
            lines.append(f"Áreas de interesse recorrentes: {', '.join(mem.interest_areas)}")
        if similar:
            lines.append("Ideias semelhantes já trabalhadas antes por este usuário:")
            lines.extend(f'  - "{s["raw_idea"][:120]}"' for s in similar)

        return "\n".join(lines) if lines else ""