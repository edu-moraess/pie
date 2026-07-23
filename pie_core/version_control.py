"""
pie_core.version_control
==========================
Prompt Version Control: um "git" simplificado para prompts.

Cada `prompt_id` é uma linha do tempo lógica (ex.: um prompt que você vai
refinando ao longo do tempo). Cada `commit` guarda o texto completo do
prompt naquele momento + uma mensagem, encadeado ao commit anterior.
"""
from __future__ import annotations

import difflib
from typing import List, Optional

from pie_core import db


class PromptRepo:
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id

    def commit(self, prompt_text: str, message: str) -> str:
        history = db.log_prompt(self.prompt_id)
        parent_id = history[-1].id if history else None
        commit = db.commit_prompt(self.prompt_id, prompt_text, message, parent_id)
        return commit.id

    def log(self) -> List[dict]:
        return [
            {
                "id": c.id,
                "parent_id": c.parent_id,
                "message": c.message,
                "created_at": c.created_at.isoformat(),
            }
            for c in db.log_prompt(self.prompt_id)
        ]

    def get(self, commit_id: str) -> Optional[str]:
        for c in db.log_prompt(self.prompt_id):
            if c.id == commit_id:
                return c.prompt_text
        return None

    def diff(self, commit_id_a: str, commit_id_b: str) -> str:
        text_a = self.get(commit_id_a) or ""
        text_b = self.get(commit_id_b) or ""
        diff = difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=commit_id_a[:8],
            tofile=commit_id_b[:8],
        )
        return "".join(diff)

    def checkout(self, commit_id: str) -> Optional[str]:
        """Retorna o texto do prompt naquele commit (para 'restaurar')."""
        return self.get(commit_id)