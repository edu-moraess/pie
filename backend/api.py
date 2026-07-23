"""
backend.api
============
API FastAPI do Prompt Intelligence Engine.

Rodar com:  uvicorn backend.api:app --reload --port 8000
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pie_core import db
from pie_core.agents.memory_learning import MemoryLearningAgent
from pie_core.benchmark import PromptBenchmark
from pie_core.graph import run_pipeline
from pie_core.library import CATEGORIES, PromptLibrary
from pie_core.llm_providers import get_provider
from pie_core.models import PIESession
from pie_core.version_control import PromptRepo

app = FastAPI(title="Prompt Intelligence Engine (PIE)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()
_memory_agent = MemoryLearningAgent()
_library = PromptLibrary()
_benchmark = PromptBenchmark()


# --------------------------------------------------------------------------
# Schemas de request
# --------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    idea: str
    user_id: str = "default"
    provider: Optional[str] = None
    model: Optional[str] = None


class LibraryEntryRequest(BaseModel):
    category: str
    title: str
    prompt_text: str
    tags: List[str] = []


class CommitRequest(BaseModel):
    prompt_id: str
    prompt_text: str
    message: str


class BenchmarkRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    test_input: str
    provider: Optional[str] = None
    model: Optional[str] = None


# --------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------

@app.post("/pipeline/run", response_model=PIESession)
def pipeline_run(req: PipelineRequest) -> PIESession:
    try:
        llm = get_provider(req.provider, req.model)
        return run_pipeline(req.idea, llm, user_id=req.user_id, memory_agent=_memory_agent)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Biblioteca
# --------------------------------------------------------------------------

@app.get("/library/categories")
def library_categories() -> List[str]:
    return CATEGORIES


@app.post("/library")
def library_add(req: LibraryEntryRequest) -> dict:
    return _library.add(req.category, req.title, req.prompt_text, req.tags)


@app.get("/library")
def library_list(category: Optional[str] = None) -> List[dict]:
    return _library.list(category)


@app.get("/library/search")
def library_search(q: str) -> List[dict]:
    return _library.search(q)


# --------------------------------------------------------------------------
# Version Control
# --------------------------------------------------------------------------

@app.post("/versions/commit")
def versions_commit(req: CommitRequest) -> dict:
    repo = PromptRepo(req.prompt_id)
    commit_id = repo.commit(req.prompt_text, req.message)
    return {"commit_id": commit_id}


@app.get("/versions/{prompt_id}/log")
def versions_log(prompt_id: str) -> List[dict]:
    return PromptRepo(prompt_id).log()


@app.get("/versions/{prompt_id}/diff")
def versions_diff(prompt_id: str, a: str, b: str) -> dict:
    return {"diff": PromptRepo(prompt_id).diff(a, b)}


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

@app.post("/benchmark")
def benchmark_run(req: BenchmarkRequest) -> dict:
    llm = get_provider(req.provider, req.model)
    return _benchmark.compare(req.prompt_a, req.prompt_b, req.test_input, llm)


# --------------------------------------------------------------------------
# Memória
# --------------------------------------------------------------------------

@app.get("/memory/{user_id}")
def memory_get(user_id: str) -> dict:
    mem = _memory_agent.get_user_memory(user_id)
    return mem.model_dump()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}