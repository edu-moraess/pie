"""
Prompt Architect Agent
=======================
Transforma a análise de intenção em um prompt profissional estruturado
em 7 blocos: ROLE, CONTEXT, OBJECTIVE, TASK, CONSTRAINTS, OUTPUT FORMAT
e QUALITY CRITERIA.
"""
from __future__ import annotations

from typing import List, Optional

from pie_core.llm_providers import LLMProvider, call_structured
from pie_core.models import IntentAnalysis, PromptStructure

SYSTEM_PROMPT = """\
Você é o Prompt Architect Agent do Prompt Intelligence Engine (PIE).
Sua função é construir um prompt profissional de altíssima qualidade a
partir de uma análise de intenção, usando SEMPRE esta estrutura de 7 blocos:

- ROLE: a persona especialista que a IA deve assumir (seja específico:
  não "um especialista", mas "um cientista de dados quant com 10 anos em
  renda fixa", por exemplo, adaptado à tarefa real).
- CONTEXT: o contexto relevante que aumenta a precisão da resposta.
- OBJECTIVE: o objetivo final, em uma frase clara.
- TASK: o que exatamente deve ser executado, passo a passo se fizer sentido.
- CONSTRAINTS: limitações, regras, formato de linguagem, o que evitar.
- OUTPUT_FORMAT: como a resposta final deve ser apresentada (estrutura,
  formato, tamanho).
- QUALITY_CRITERIA: os padrões pelos quais a resposta deve ser julgada.

Escreva em português, denso e específico, sem enrolação. Nunca deixe um
campo genérico ou vazio.
"""


class PromptArchitectAgent:
    def build(
        self,
        intent: IntentAnalysis,
        llm: LLMProvider,
        critic_feedback: Optional[List[str]] = None,
    ) -> PromptStructure:
        feedback_block = ""
        if critic_feedback:
            feedback_block = (
                "\n\nO prompt anterior foi avaliado por um revisor e recebeu estas "
                "melhorias sugeridas — incorpore todas elas nesta nova versão:\n"
                + "\n".join(f"- {f}" for f in critic_feedback)
            )

        user_prompt = (
            f"Análise de intenção:\n"
            f"- Objetivo: {intent.goal}\n"
            f"- Tipo de tarefa: {intent.task_type.value}\n"
            f"- Contexto necessário: {', '.join(intent.required_context) or 'nenhum informado'}\n"
            f"- Requisitos da tarefa: {', '.join(intent.task_requirements) or 'nenhum informado'}\n"
            f"- Modelo recomendado: {intent.recommended_model.value}"
            f"{feedback_block}"
        )
        return call_structured(llm, SYSTEM_PROMPT, user_prompt, PromptStructure)