"""
pie_core.models
================
Todos os contratos de dados (Pydantic) trocados entre os agentes do
Prompt Intelligence Engine (PIE). Manter esses modelos estáveis é o que
permite que Intent Analysis -> Prompt Architect -> Prompt Critic ->
Prompt Optimization -> Memory & Learning conversem entre si sem
ambiguidade.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class TaskType(str, Enum):
    TEXT = "texto"
    CODE = "codigo"
    IMAGE = "imagem"
    DATA_ANALYSIS = "analise_de_dados"
    RESEARCH = "pesquisa"
    MARKETING = "marketing"
    BUSINESS = "negocios"
    EDUCATION = "educacao"
    OTHER = "outro"


class RecommendedModel(str, Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    MIDJOURNEY = "midjourney"
    OPEN_SOURCE = "open_source"


class ModelTarget(str, Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    MIDJOURNEY = "midjourney"
    OPEN_SOURCE = "open_source"


# --------------------------------------------------------------------------
# 1. Intent Analysis Agent
# --------------------------------------------------------------------------

class IntentAnalysis(BaseModel):
    goal: str = Field(..., description="Objetivo principal identificado na solicitação do usuário")
    task_type: TaskType
    missing_information: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    required_context: List[str] = Field(default_factory=list)
    recommended_model: RecommendedModel
    task_requirements: List[str] = Field(default_factory=list)
    confidence: float = Field(0.7, ge=0.0, le=1.0)


# --------------------------------------------------------------------------
# 2. Prompt Architect Agent
# --------------------------------------------------------------------------

class PromptStructure(BaseModel):
    role: str
    context: str
    objective: str
    task: str
    constraints: List[str] = Field(default_factory=list)
    output_format: str
    quality_criteria: List[str] = Field(default_factory=list)

    def render(self) -> str:
        """Renderiza a estrutura em um prompt final em texto corrido."""
        constraints = "\n".join(f"- {c}" for c in self.constraints) or "- (nenhuma restrição adicional)"
        criteria = "\n".join(f"- {c}" for c in self.quality_criteria) or "- (critérios padrão de qualidade)"
        return (
            f"ROLE:\n{self.role}\n\n"
            f"CONTEXT:\n{self.context}\n\n"
            f"OBJECTIVE:\n{self.objective}\n\n"
            f"TASK:\n{self.task}\n\n"
            f"CONSTRAINTS:\n{constraints}\n\n"
            f"OUTPUT FORMAT:\n{self.output_format}\n\n"
            f"QUALITY CRITERIA:\n{criteria}"
        )


# --------------------------------------------------------------------------
# 3. Prompt Critic Agent
# --------------------------------------------------------------------------

class CriticEvaluation(BaseModel):
    score: int = Field(..., ge=0, le=100)
    clarity: int = Field(..., ge=0, le=100)
    specificity: int = Field(..., ge=0, le=100)
    context_quality: int = Field(..., ge=0, le=100)
    efficiency: int = Field(..., ge=0, le=100)
    ambiguities: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    verdict: str = ""


# --------------------------------------------------------------------------
# 4. Prompt Optimization Agent
# --------------------------------------------------------------------------

class OptimizedVersions(BaseModel):
    basic: str
    professional: str
    expert: str
    model_specific: Dict[ModelTarget, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 5. Memory & Learning Agent
# --------------------------------------------------------------------------

class UserMemory(BaseModel):
    user_id: str
    format_preferences: List[str] = Field(default_factory=list)
    response_style: List[str] = Field(default_factory=list)
    interest_areas: List[str] = Field(default_factory=list)
    usage_patterns: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------
# Explainable AI / trace de raciocínio
# --------------------------------------------------------------------------

class ReasoningStep(BaseModel):
    step: str
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------
# Sessão completa do pipeline
# --------------------------------------------------------------------------

class PIESession(BaseModel):
    session_id: str
    user_id: str = "default"
    raw_idea: str
    intent: Optional[IntentAnalysis] = None
    prompt_structure: Optional[PromptStructure] = None
    final_prompt: str = ""
    evaluation: Optional[CriticEvaluation] = None
    versions: Optional[OptimizedVersions] = None
    reasoning_log: List[ReasoningStep] = Field(default_factory=list)
    refinement_iterations: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)