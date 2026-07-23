"""
pie_core.graph
================
Orquestra os 5 agentes do PIE com LangGraph:

    intent_node -> architect_node -> critic_node --(score baixo)--> architect_node (refino)
                                              \\--(score ok / limite de iterações)--> optimizer_node -> memory_node -> END

Isso implementa, ao mesmo tempo, o "Auto Refinement" (o prompt é
reconstruído automaticamente quando o Critic aponta score baixo) e a
"Explainable AI" (cada transição de nó grava um ReasoningStep explicando
por que aquela decisão foi tomada).
"""
from __future__ import annotations

import uuid
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from pie_core.agents.intent_analysis import IntentAnalysisAgent
from pie_core.agents.memory_learning import MemoryLearningAgent
from pie_core.agents.prompt_architect import PromptArchitectAgent
from pie_core.agents.prompt_critic import PromptCriticAgent
from pie_core.agents.prompt_optimizer import PromptOptimizationAgent
from pie_core.llm_providers import LLMProvider
from pie_core.models import (
    CriticEvaluation,
    IntentAnalysis,
    OptimizedVersions,
    PIESession,
    PromptStructure,
    ReasoningStep,
)

SCORE_THRESHOLD = 80
MAX_REFINEMENTS = 2


class PIEState(TypedDict, total=False):
    raw_idea: str
    user_id: str
    llm: LLMProvider
    intent: IntentAnalysis
    structure: PromptStructure
    critic: CriticEvaluation
    versions: OptimizedVersions
    reasoning_log: List[ReasoningStep]
    iterations: int
    memory_context: str


def _log(state: PIEState, step: str, reasoning: str) -> None:
    state.setdefault("reasoning_log", []).append(ReasoningStep(step=step, reasoning=reasoning))


def _build_graph(memory_agent: Optional[MemoryLearningAgent] = None):
    intent_agent = IntentAnalysisAgent()
    architect_agent = PromptArchitectAgent()
    critic_agent = PromptCriticAgent()
    optimizer_agent = PromptOptimizationAgent()

    def intent_node(state: PIEState) -> PIEState:
        llm = state["llm"]
        memory_ctx = ""
        if memory_agent is not None:
            memory_ctx = memory_agent.build_memory_context(state.get("user_id", "default"), state["raw_idea"])
        state["memory_context"] = memory_ctx

        intent = intent_agent.analyze(state["raw_idea"], llm)
        state["intent"] = intent
        _log(
            state,
            "Intent Analysis",
            f"Objetivo identificado: '{intent.goal}'. Classificado como '{intent.task_type.value}'. "
            f"Modelo recomendado: {intent.recommended_model.value}.",
        )
        return state

    def architect_node(state: PIEState) -> PIEState:
        llm = state["llm"]
        critic_feedback = state["critic"].improvements if state.get("critic") else None
        intent = state["intent"]

        # injeta memória do usuário como contexto adicional, se houver
        if state.get("memory_context"):
            intent = intent.model_copy(
                update={"required_context": intent.required_context + [f"Memória do usuário: {state['memory_context']}"]}
            )

        structure = architect_agent.build(intent, llm, critic_feedback=critic_feedback)
        state["structure"] = structure
        iteration = state.get("iterations", 0)
        if critic_feedback:
            _log(
                state,
                f"Prompt Architect (refinamento {iteration})",
                "Reconstruiu o prompt incorporando as melhorias apontadas pelo Critic: "
                + "; ".join(critic_feedback),
            )
        else:
            _log(state, "Prompt Architect", "Construiu o prompt estruturado inicial nos 7 blocos padrão do PIE.")
        return state

    def critic_node(state: PIEState) -> PIEState:
        llm = state["llm"]
        rendered = state["structure"].render()
        evaluation = critic_agent.evaluate(rendered, llm)
        state["critic"] = evaluation
        _log(
            state,
            "Prompt Critic",
            f"Score {evaluation.score}/100. Veredito: {evaluation.verdict or 'sem observação adicional'}.",
        )
        return state

    def should_refine(state: PIEState) -> str:
        iterations = state.get("iterations", 0)
        if state["critic"].score < SCORE_THRESHOLD and iterations < MAX_REFINEMENTS:
            state["iterations"] = iterations + 1
            return "refine"
        return "proceed"

    def optimizer_node(state: PIEState) -> PIEState:
        llm = state["llm"]
        versions = optimizer_agent.generate_versions(state["structure"], llm, critic=state.get("critic"))
        state["versions"] = versions
        _log(
            state,
            "Prompt Optimization",
            "Gerou as versões Básica, Profissional, Expert e as variantes por modelo "
            "(GPT, Claude, Gemini, Midjourney, open-source).",
        )
        return state

    def memory_node(state: PIEState) -> PIEState:
        if memory_agent is not None:
            # a persistência efetiva da sessão acontece em run_pipeline() (precisa do PIESession completo)
            _log(state, "Memory & Learning", "Sessão registrada na memória semântica e nos padrões de uso do usuário.")
        return state

    graph = StateGraph(PIEState)
    graph.add_node("intent", intent_node)
    graph.add_node("architect", architect_node)
    graph.add_node("critic", critic_node)
    graph.add_node("optimizer", optimizer_node)
    graph.add_node("memory", memory_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "architect")
    graph.add_edge("architect", "critic")
    graph.add_conditional_edges("critic", should_refine, {"refine": "architect", "proceed": "optimizer"})
    graph.add_edge("optimizer", "memory")
    graph.add_edge("memory", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph(memory_agent: Optional[MemoryLearningAgent] = None):
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph(memory_agent)
    return _compiled_graph


def run_pipeline(
    raw_idea: str,
    llm: LLMProvider,
    user_id: str = "default",
    memory_agent: Optional[MemoryLearningAgent] = None,
) -> PIESession:
    """Executa o pipeline completo do PIE e retorna uma PIESession pronta para exibição/API."""
    compiled = get_compiled_graph(memory_agent)
    initial_state: PIEState = {"raw_idea": raw_idea, "user_id": user_id, "llm": llm, "iterations": 0}
    final_state = compiled.invoke(initial_state)

    session = PIESession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        raw_idea=raw_idea,
        intent=final_state["intent"],
        prompt_structure=final_state["structure"],
        final_prompt=final_state["structure"].render(),
        evaluation=final_state["critic"],
        versions=final_state["versions"],
        reasoning_log=final_state["reasoning_log"],
        refinement_iterations=final_state.get("iterations", 0),
    )

    if memory_agent is not None:
        memory_agent.store_session(session)

    return session