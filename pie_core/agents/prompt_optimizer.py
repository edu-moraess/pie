"""
Prompt Optimization Agent
==========================
Gera múltiplas versões do prompt final: Básica, Profissional, Expert, e
versões adaptadas a modelos específicos (GPT, Claude, Gemini, Midjourney,
open-source).
"""
from __future__ import annotations

from typing import List, Optional

from pie_core.llm_providers import LLMProvider, call_structured
from pie_core.models import CriticEvaluation, ModelTarget, OptimizedVersions, PromptStructure

SYSTEM_PROMPT = """\
Você é o Prompt Optimization Agent do Prompt Intelligence Engine (PIE).
A partir de um prompt já estruturado (e, se houver, do feedback de um
revisor), gere 3 níveis de versão MAIS versões específicas por modelo:

- basic: versão simples e direta, poucas linhas, para quem só quer algo
  rápido e funcional.
- professional: versão otimizada, mantém a estrutura de blocos, boa para
  usuários avançados que vão usar em produção.
- expert: versão de nível empresarial — inclui metodologia explícita,
  critérios de validação, passos de verificação/auto-checagem, e tratamento
  de casos-limite. É a versão mais longa e rigorosa.
- model_specific: um dicionário com as chaves "gpt", "claude", "gemini",
  "midjourney", "open_source", cada uma adaptando o MESMO prompt para as
  particularidades daquele modelo:
    * gpt: aproveita bem system+user separados, boa em seguir listas numeradas.
    * claude: responde bem a tags XML e a raciocínio passo a passo explícito.
    * gemini: pode explorar contexto longo e multimodalidade quando fizer sentido.
    * midjourney: reescreva como um prompt de geração de imagem (parâmetros
      visuais, estilo, iluminação, composição, --ar, --v quando aplicável);
      se a tarefa não for de imagem, adapte descrevendo visualmente o
      resultado desejado como um brief criativo.
    * open_source: versão mais explícita e "à prova de falhas", já que
      modelos open-source menores seguem instruções ambíguas pior.

Todas as versões devem preservar a intenção e o objetivo originais.
"""


class PromptOptimizationAgent:
    def generate_versions(
        self,
        structure: PromptStructure,
        llm: LLMProvider,
        critic: Optional[CriticEvaluation] = None,
        targets: Optional[List[ModelTarget]] = None,
    ) -> OptimizedVersions:
        feedback_block = ""
        if critic:
            feedback_block = (
                f"\n\nFeedback do revisor (score atual: {critic.score}/100):\n"
                + "\n".join(f"- {i}" for i in critic.improvements)
            )
        user_prompt = f"Prompt estruturado base:\n\n{structure.render()}{feedback_block}"
        return call_structured(llm, SYSTEM_PROMPT, user_prompt, OptimizedVersions, max_tokens=4000)