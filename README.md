# PIE — Prompt Intelligence Engine

Laboratório de engenharia de prompts baseado em arquitetura multiagente:
transforma uma ideia crua em prompts profissionais, avaliados e otimizados
para diferentes modelos de IA (GPT, Claude, Gemini, Midjourney, open-source).

## Arquitetura

Ideia do usuário
│
▼
[1] Intent Analysis Agent      → objetivo, tipo de tarefa, contexto faltante
│
▼
[2] Prompt Architect Agent     → ROLE / CONTEXT / OBJECTIVE / TASK /
│                          CONSTRAINTS / OUTPUT FORMAT / QUALITY CRITERIA
▼
[3] Prompt Critic Agent        → score 0-100 + melhorias
│
├──(score < 80 e < 2 refinamentos)──► volta para [2] (Auto Refinement)
│
▼
[4] Prompt Optimization Agent  → versões Básica / Profissional / Expert
│                          + variantes por modelo
▼
[5] Memory & Learning Agent    → memória semântica (ChromaDB) +
preferências estruturadas (SQL)
Orquestrado com **LangGraph** (`pie_core/graph.py`), com um log de
raciocínio (`reasoning_log`) explicando cada decisão — a base do recurso
de **Explainable AI**.

## Estrutura do projeto

pie/
├── pie_core/              núcleo do sistema (reutilizado por API e UI)
│   ├── models.py          contratos Pydantic entre os agentes
│   ├── llm_providers.py   abstração Claude / GPT / Gemini / open-source
│   ├── graph.py           orquestração LangGraph + auto-refinamento
│   ├── db.py              persistência estruturada (SQLite/Postgres)
│   ├── version_control.py Prompt Version Control (estilo Git)
│   ├── library.py         Prompt Library por categorias
│   ├── benchmark.py       Prompt Benchmark (A/B com juiz LLM)
│   └── agents/            os 5 agentes do PIE
├── backend/api.py         API FastAPI
├── frontend/app.py        interface Streamlit ("AI Terminal")
├── requirements.txt
└── .env.example

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # preencha ao menos uma chave de API (ex: ANTHROPIC_API_KEY)


