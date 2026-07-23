"""
pie_core.benchmark
====================
Prompt Benchmark: roda dois prompts concorrentes contra a mesma entrada de
teste no mesmo LLM, e usa um juiz (LLM) para decidir qual produz o
resultado de melhor qualidade — registrando o resultado no banco.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from pie_core import db
from pie_core.llm_providers import LLMProvider, call_structured

JUDGE_SYSTEM_PROMPT = """\
Você é um juiz imparcial de qualidade de respostas de IA. Vai receber duas
respostas (A e B), geradas por dois prompts diferentes para a mesma
entrada de teste. Avalie cada uma de 0 a 100 quanto a: aderência ao que foi
pedido, profundidade, clareza e utilidade prática. Declare um vencedor
("A", "B" ou "empate") e explique o motivo em até 3 frases.
"""


class _JudgeVerdict(BaseModel):
    score_a: int = Field(..., ge=0, le=100)
    score_b: int = Field(..., ge=0, le=100)
    winner: str
    rationale: str


class PromptBenchmark:
    def compare(self, prompt_a: str, prompt_b: str, test_input: str, llm: LLMProvider) -> dict:
        output_a = llm.generate(prompt_a, test_input)
        output_b = llm.generate(prompt_b, test_input)

        judge_user_prompt = (
            f"Entrada de teste usada para ambos os prompts:\n\"\"\"\n{test_input}\n\"\"\"\n\n"
            f"Resposta A:\n\"\"\"\n{output_a}\n\"\"\"\n\n"
            f"Resposta B:\n\"\"\"\n{output_b}\n\"\"\""
        )
        verdict = call_structured(llm, JUDGE_SYSTEM_PROMPT, judge_user_prompt, _JudgeVerdict)

        run = db.save_benchmark(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            output_a=output_a,
            output_b=output_b,
            score_a=verdict.score_a,
            score_b=verdict.score_b,
            winner=verdict.winner,
            rationale=verdict.rationale,
        )
        return {
            "id": run.id,
            "output_a": output_a,
            "output_b": output_b,
            "score_a": verdict.score_a,
            "score_b": verdict.score_b,
            "winner": verdict.winner,
            "rationale": verdict.rationale,
        }