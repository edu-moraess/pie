"""
pie_core.library
==================
Biblioteca de prompts organizada por categoria.
"""
from __future__ import annotations

from typing import List, Optional

from pie_core import db

CATEGORIES = [
    "Data Science",
    "Finance",
    "Coding",
    "Research",
    "Marketing",
    "Image Generation",
]


class PromptLibrary:
    def add(self, category: str, title: str, prompt_text: str, tags: Optional[List[str]] = None) -> dict:
        if category not in CATEGORIES:
            raise ValueError(f"Categoria inválida. Use uma de: {CATEGORIES}")
        entry = db.add_library_entry(category, title, prompt_text, tags)
        return self._serialize(entry)

    def list(self, category: Optional[str] = None) -> List[dict]:
        return [self._serialize(e) for e in db.list_library(category)]

    def search(self, query: str) -> List[dict]:
        return [self._serialize(e) for e in db.search_library(query)]

    @staticmethod
    def _serialize(entry) -> dict:
        return {
            "id": entry.id,
            "category": entry.category,
            "title": entry.title,
            "prompt_text": entry.prompt_text,
            "tags": entry.tags.split(",") if entry.tags else [],
            "created_at": entry.created_at.isoformat(),
        }