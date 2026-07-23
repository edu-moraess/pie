"""
pie_core.db
============
Dados estruturados do PIE via SQLAlchemy.

Por padrão usa SQLite local (./pie.db) para rodar sem infraestrutura
externa. Para produção, defina DATABASE_URL com uma conexão Postgres, por
exemplo:  postgresql+psycopg2://user:pass@host:5432/pie
Nenhum outro código do projeto precisa mudar — SQLAlchemy abstrai o dialeto.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from pie_core.models import UserMemory

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pie.db")

_engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# Prompt Version Control (estilo Git, simplificado)
# --------------------------------------------------------------------------

class PromptCommit(Base):
    __tablename__ = "prompt_commits"

    id = Column(String, primary_key=True)
    prompt_id = Column(String, index=True)       # identifica a "linha do tempo" (um prompt lógico)
    parent_id = Column(String, nullable=True)
    message = Column(String)
    prompt_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# --------------------------------------------------------------------------
# Prompt Library
# --------------------------------------------------------------------------

class LibraryEntry(Base):
    __tablename__ = "library_entries"

    id = Column(String, primary_key=True)
    category = Column(String, index=True)   # Data Science, Finance, Coding, Research, Marketing, Image Generation
    title = Column(String)
    prompt_text = Column(Text)
    tags = Column(String, default="")       # csv simples
    created_at = Column(DateTime, default=datetime.utcnow)


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(String, primary_key=True)
    prompt_a = Column(Text)
    prompt_b = Column(Text)
    output_a = Column(Text)
    output_b = Column(Text)
    score_a = Column(Integer)
    score_b = Column(Integer)
    winner = Column(String)
    rationale = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# --------------------------------------------------------------------------
# Memória estruturada do usuário
# --------------------------------------------------------------------------

class UserMemoryRecord(Base):
    __tablename__ = "user_memory"

    user_id = Column(String, primary_key=True)
    format_preferences = Column(Text, default="[]")   # JSON list
    response_style = Column(Text, default="[]")       # JSON list
    interest_areas = Column(Text, default="[]")       # JSON list
    usage_patterns = Column(Text, default="[]")       # JSON list
    updated_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(_engine)


# --------------------------------------------------------------------------
# API de conveniência — Version Control
# --------------------------------------------------------------------------

def commit_prompt(prompt_id: str, prompt_text: str, message: str, parent_id: Optional[str] = None) -> PromptCommit:
    with SessionLocal() as s:
        commit = PromptCommit(
            id=str(uuid.uuid4()),
            prompt_id=prompt_id,
            parent_id=parent_id,
            message=message,
            prompt_text=prompt_text,
        )
        s.add(commit)
        s.commit()
        s.refresh(commit)
        return commit


def log_prompt(prompt_id: str) -> List[PromptCommit]:
    with SessionLocal() as s:
        return (
            s.query(PromptCommit)
            .filter(PromptCommit.prompt_id == prompt_id)
            .order_by(PromptCommit.created_at.asc())
            .all()
        )


# --------------------------------------------------------------------------
# API de conveniência — Library
# --------------------------------------------------------------------------

def add_library_entry(category: str, title: str, prompt_text: str, tags: Optional[List[str]] = None) -> LibraryEntry:
    with SessionLocal() as s:
        entry = LibraryEntry(
            id=str(uuid.uuid4()),
            category=category,
            title=title,
            prompt_text=prompt_text,
            tags=",".join(tags or []),
        )
        s.add(entry)
        s.commit()
        s.refresh(entry)
        return entry


def list_library(category: Optional[str] = None) -> List[LibraryEntry]:
    with SessionLocal() as s:
        q = s.query(LibraryEntry)
        if category:
            q = q.filter(LibraryEntry.category == category)
        return q.order_by(LibraryEntry.created_at.desc()).all()


def search_library(query: str) -> List[LibraryEntry]:
    with SessionLocal() as s:
        like = f"%{query}%"
        return (
            s.query(LibraryEntry)
            .filter((LibraryEntry.title.ilike(like)) | (LibraryEntry.prompt_text.ilike(like)))
            .all()
        )


# --------------------------------------------------------------------------
# API de conveniência — Benchmark
# --------------------------------------------------------------------------

def save_benchmark(
    prompt_a: str,
    prompt_b: str,
    output_a: str,
    output_b: str,
    score_a: int,
    score_b: int,
    winner: str,
    rationale: str,
) -> BenchmarkRun:
    with SessionLocal() as s:
        run = BenchmarkRun(
            id=str(uuid.uuid4()),
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            output_a=output_a,
            output_b=output_b,
            score_a=score_a,
            score_b=score_b,
            winner=winner,
            rationale=rationale,
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        return run


# --------------------------------------------------------------------------
# API de conveniência — Memória estruturada
# --------------------------------------------------------------------------

def get_user_memory(user_id: str) -> Optional[UserMemory]:
    with SessionLocal() as s:
        rec = s.get(UserMemoryRecord, user_id)
        if not rec:
            return None
        return UserMemory(
            user_id=rec.user_id,
            format_preferences=json.loads(rec.format_preferences),
            response_style=json.loads(rec.response_style),
            interest_areas=json.loads(rec.interest_areas),
            usage_patterns=json.loads(rec.usage_patterns),
            updated_at=rec.updated_at,
        )


def upsert_user_memory(
    user_id: str,
    format_preferences: Optional[List[str]] = None,
    response_style: Optional[List[str]] = None,
    interest_areas: Optional[List[str]] = None,
) -> UserMemory:
    with SessionLocal() as s:
        rec = s.get(UserMemoryRecord, user_id)
        if rec is None:
            rec = UserMemoryRecord(user_id=user_id)
            s.add(rec)

        def _merge(existing_json: str, new_items: Optional[List[str]]) -> str:
            if not new_items:
                return existing_json
            existing = set(json.loads(existing_json))
            existing.update(new_items)
            return json.dumps(sorted(existing), ensure_ascii=False)

        rec.format_preferences = _merge(rec.format_preferences, format_preferences)
        rec.response_style = _merge(rec.response_style, response_style)
        rec.interest_areas = _merge(rec.interest_areas, interest_areas)
        rec.updated_at = datetime.utcnow()
        s.commit()
        s.refresh(rec)
        return UserMemory(
            user_id=rec.user_id,
            format_preferences=json.loads(rec.format_preferences),
            response_style=json.loads(rec.response_style),
            interest_areas=json.loads(rec.interest_areas),
            usage_patterns=json.loads(rec.usage_patterns),
            updated_at=rec.updated_at,
        )


def record_usage(user_id: str, task_type: str, recommended_model: Optional[str]) -> None:
    """Registra um padrão de uso simples (tipo de tarefa) na memória estruturada."""
    with SessionLocal() as s:
        rec = s.get(UserMemoryRecord, user_id)
        if rec is None:
            rec = UserMemoryRecord(user_id=user_id)
            s.add(rec)
        patterns = json.loads(rec.usage_patterns)
        label = f"tarefa:{task_type}" + (f"|modelo:{recommended_model}" if recommended_model else "")
        patterns.append(label)
        rec.usage_patterns = json.dumps(patterns[-50:], ensure_ascii=False)  # mantém as últimas 50
        rec.updated_at = datetime.utcnow()
        s.commit()