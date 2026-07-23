"""
pie_core.llm_providers
=======================
Abstração fina sobre múltiplos provedores de LLM para que os agentes do
PIE nunca precisem saber qual API está por trás. Cada provider expõe:

    generate(system: str, user: str) -> str

As chaves de API são lidas de variáveis de ambiente (ver .env.example) ou
de st.secrets (via Streamlit). Nunca hardcode chaves aqui.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str, max_tokens: int = 2000) -> str:
        ...


class AnthropicProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class OpenAIProvider(LLMProvider):
    name = "gpt"

    def __init__(self, model: str = "gpt-4.1", api_key: Optional[str] = None):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self._model_name = model

    def generate(self, system: str, user: str, max_tokens: int = 2000) -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(self._model_name, system_instruction=system)
        resp = model.generate_content(user, generation_config={"max_output_tokens": max_tokens})
        return resp.text or ""


class GroqProvider(LLMProvider):
    """
    Provedor específico para a API da Groq (modelos open-source como Llama, Mixtral).
    Utiliza a SDK oficial da Groq.
    """
    name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        from groq import Groq

        self.model = model
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def generate(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class OpenSourceProvider(LLMProvider):
    """
    Provider genérico para qualquer endpoint compatível com a API da OpenAI
    (Ollama, vLLM, LM Studio, Together.ai, etc). Configure via
    OPENSOURCE_BASE_URL e OPENSOURCE_MODEL no .env.
    """

    name = "open_source"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENSOURCE_MODEL", "llama3")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENSOURCE_API_KEY", "not-needed"),
            base_url=base_url or os.getenv("OPENSOURCE_BASE_URL", "http://localhost:11434/v1"),
        )

    def generate(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# Lista atualizada de modelos compatíveis com a Groq (setembro 2025)
GROQ_MODELS = {
    # Meta Llama
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama4-scout-17b-16e",
    "llama4-maverick-17b-128e",
    # Mistral
    "mixtral-8x7b-32768",
    # Google
    "gemma2-9b-it",
    "gemma-7b-it",
    # OpenAI OSS
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    # Qwen
    "qwen/qwen3.6-27b",
    "qwen/qwen3.5-14b",
    # DeepSeek
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    # Outros
    "gemma3-12b-it",
    "gemma3-27b-it",
}


def _should_use_groq(model: Optional[str]) -> bool:
    """Verifica se o modelo é compatível com a Groq e a chave está presente."""
    if not model:
        return False
    if model not in GROQ_MODELS:
        return False
    return bool(os.getenv("GROQ_API_KEY"))


_PROVIDERS = {
    "claude": AnthropicProvider,
    "anthropic": AnthropicProvider,
    "gpt": OpenAIProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "groq": GroqProvider,          # Agora registrado e visível no frontend
    "open_source": OpenSourceProvider,
    "opensource": OpenSourceProvider,
}


def get_provider(name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """Factory central: retorna uma instância do provider pedido (ou o default do .env)."""
    if name is None:
        name = os.getenv("DEFAULT_PROVIDER", "claude")
    key = name.lower()

    # Se o usuário escolheu "groq" diretamente, usa GroqProvider
    if key == "groq":
        return GroqProvider(model=model or "llama-3.3-70b-versatile")

    # Fallback: se escolheu "open_source" e modelo é da Groq
    if key in ("open_source", "opensource") and _should_use_groq(model):
        return GroqProvider(model=model or "llama-3.3-70b-versatile")

    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(f"Provider desconhecido: {key}. Opções: {list(_PROVIDERS)}")
    kwargs = {"model": model} if model else {}
    return cls(**kwargs)


# --------------------------------------------------------------------------
# Helper: pedir JSON estruturado a um LLM e validar contra um schema Pydantic
# --------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Remove cercas de código markdown e texto solto ao redor do JSON."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()
    # fallback: pega do primeiro '{' ao último '}'
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def call_structured(
    llm: LLMProvider,
    system: str,
    user: str,
    schema: Type[T],
    max_tokens: int = 2000,
    retries: int = 1,
) -> T:
    """
    Chama o LLM pedindo explicitamente JSON puro compatível com `schema`,
    faz o parse e valida. Em caso de erro, tenta novamente incluindo o erro
    de validação no próximo prompt (auto-correção).
    """
    schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    full_system = (
        f"{system}\n\n"
        "Responda ESTRITAMENTE com um objeto JSON válido, sem nenhum texto antes ou depois, "
        "sem cercas de markdown, seguindo este JSON Schema:\n"
        f"{schema_hint}"
    )

    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        prompt = user if last_error is None else f"{user}\n\nSeu JSON anterior falhou a validação com este erro, corrija:\n{last_error}"
        raw = llm.generate(full_system, prompt, max_tokens=max_tokens)
        try:
            data = json.loads(_extract_json(raw))
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
    raise ValueError(f"Falha ao obter saída estruturada válida após {retries + 1} tentativas: {last_error}")