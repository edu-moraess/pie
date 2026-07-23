"""
frontend.app
=============
Interface Streamlit do Prompt Intelligence Engine — visual "AI Terminal".

Rodar com (a partir da raiz do projeto):
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# permite `streamlit run frontend/app.py` a partir da raiz do projeto
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from pie_core import db
from pie_core.agents.memory_learning import MemoryLearningAgent
from pie_core.benchmark import PromptBenchmark
from pie_core.graph import run_pipeline
from pie_core.library import CATEGORIES, PromptLibrary
from pie_core.llm_providers import get_provider
from pie_core.version_control import PromptRepo

st.set_page_config(page_title="PIE — Prompt Intelligence Engine", page_icon="🧠", layout="wide")

db.init_db()

if "memory_agent" not in st.session_state:
    st.session_state.memory_agent = MemoryLearningAgent()
if "library" not in st.session_state:
    st.session_state.library = PromptLibrary()
if "benchmark" not in st.session_state:
    st.session_state.benchmark = PromptBenchmark()
if "session" not in st.session_state:
    st.session_state.session = None

# --------------------------------------------------------------------------
# CSS — estética de terminal de IA (dark, monoespaçado, glow verde/ciano)
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0a0e12;
        color: #d8f5e3;
    }
    * { font-family: 'JetBrains Mono', 'Courier New', monospace !important; }
    h1, h2, h3 { color: #35f2a8 !important; text-shadow: 0 0 8px rgba(53,242,168,0.35); }
    .pie-panel {
        background: #0f151b;
        border: 1px solid #1f3b30;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .pie-score {
        font-size: 2.2rem;
        font-weight: bold;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #0f151b !important;
        color: #d8f5e3 !important;
        border: 1px solid #1f3b30 !important;
    }
    .stButton button {
        background-color: #123527;
        color: #35f2a8;
        border: 1px solid #35f2a8;
        border-radius: 4px;
    }
    .stButton button:hover { background-color: #1a4d36; color: #ffffff; }
    code { color: #7ef7c7 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 PIE")
    st.caption("Prompt Intelligence Engine")

    page = st.radio("Navegação", ["Studio", "Biblioteca", "Versionamento", "Benchmark", "Memória"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Provedor de IA**")
    provider = st.selectbox("Provider", ["claude", "gpt", "gemini", "open_source"], label_visibility="collapsed")
    model = st.text_input("Modelo (opcional)", placeholder="ex: claude-sonnet-4-6")
    user_id = st.text_input("User ID", value="default")

    st.markdown("---")
    st.caption("As chaves de API são lidas do arquivo .env (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).")


def _llm():
    return get_provider(provider, model or None)


# --------------------------------------------------------------------------
# Página: Studio (pipeline principal)
# --------------------------------------------------------------------------

if page == "Studio":
    st.title("🧠 Prompt Intelligence Engine")
    st.caption("Transforme uma ideia simples em um prompt profissional, avaliado e otimizado.")

    idea = st.text_area("Descreva sua ideia", height=140, placeholder="Ex: quero um prompt para gerar um relatório semanal de risco de crédito...")

    run = st.button("▶ Processar", type="primary")

    if run and idea.strip():
        steps = ["Análise da intenção", "Construção do prompt", "Avaliação", "Otimização"]
        progress = st.status("Processando pipeline PIE...", expanded=True)
        with progress:
            for s in steps:
                st.write(f"→ {s}")
        try:
            session = run_pipeline(idea, _llm(), user_id=user_id, memory_agent=st.session_state.memory_agent)
            st.session_state.session = session
            progress.update(label="Pipeline concluído", state="complete")
        except Exception as e:  # noqa: BLE001
            progress.update(label="Erro no pipeline", state="error")
            st.error(f"Falha ao rodar o pipeline: {e}")

    session = st.session_state.session
    if session:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Prompt final")
            st.code(session.final_prompt, language="markdown")

        with col2:
            st.subheader("Score")
            score = session.evaluation.score if session.evaluation else 0
            color = "#35f2a8" if score >= 80 else ("#f2c335" if score >= 60 else "#f24f4f")
            st.markdown(f"<div class='pie-score' style='color:{color}'>{score}/100</div>", unsafe_allow_html=True)
            st.progress(score / 100)
            if session.refinement_iterations:
                st.caption(f"Refinado automaticamente {session.refinement_iterations}x")

        st.subheader("Melhorias sugeridas")
        if session.evaluation and session.evaluation.improvements:
            for imp in session.evaluation.improvements:
                st.markdown(f"- {imp}")
        else:
            st.caption("Nenhuma melhoria pendente relevante.")

        st.subheader("Versões alternativas")
        if session.versions:
            tabs = st.tabs(["Básica", "Profissional", "Expert", "GPT", "Claude", "Gemini", "Midjourney", "Open-source"])
            with tabs[0]:
                st.code(session.versions.basic, language="markdown")
            with tabs[1]:
                st.code(session.versions.professional, language="markdown")
            with tabs[2]:
                st.code(session.versions.expert, language="markdown")
            model_tabs = {"gpt": tabs[3], "claude": tabs[4], "gemini": tabs[5], "midjourney": tabs[6], "open_source": tabs[7]}
            for key, tab in model_tabs.items():
                with tab:
                    text = session.versions.model_specific.get(key, "(não gerado)")
                    st.code(text, language="markdown")

        with st.expander("🔍 Explainable AI — por que cada decisão foi tomada"):
            for step in session.reasoning_log:
                st.markdown(f"**{step.step}**  \n{step.reasoning}")

        with st.expander("💾 Salvar como versão / na biblioteca"):
            c1, c2 = st.columns(2)
            with c1:
                prompt_id = st.text_input("ID do prompt (para versionamento)", value=session.session_id[:8])
                msg = st.text_input("Mensagem do commit", value="Versão gerada pelo PIE")
                if st.button("Commitar versão"):
                    PromptRepo(prompt_id).commit(session.final_prompt, msg)
                    st.success("Commit registrado.")
            with c2:
                cat = st.selectbox("Categoria", CATEGORIES)
                title = st.text_input("Título na biblioteca", value=(session.intent.goal if session.intent else "Prompt PIE"))
                if st.button("Adicionar à biblioteca"):
                    st.session_state.library.add(cat, title, session.final_prompt)
                    st.success("Adicionado à biblioteca.")

# --------------------------------------------------------------------------
# Página: Biblioteca
# --------------------------------------------------------------------------

elif page == "Biblioteca":
    st.title("📚 Prompt Library")
    cat_filter = st.selectbox("Categoria", ["Todas"] + CATEGORIES)
    entries = st.session_state.library.list(None if cat_filter == "Todas" else cat_filter)
    if not entries:
        st.caption("Nenhum prompt salvo ainda nesta categoria.")
    for e in entries:
        with st.expander(f"[{e['category']}] {e['title']}"):
            st.code(e["prompt_text"], language="markdown")

# --------------------------------------------------------------------------
# Página: Versionamento
# --------------------------------------------------------------------------

elif page == "Versionamento":
    st.title("🕘 Prompt Version Control")
    prompt_id = st.text_input("ID do prompt")
    if prompt_id:
        repo = PromptRepo(prompt_id)
        log = repo.log()
        if not log:
            st.caption("Nenhum commit encontrado para este ID.")
        else:
            for c in log:
                st.markdown(f"`{c['id'][:8]}` — {c['message']} — {c['created_at']}")
            st.markdown("---")
            ids = [c["id"] for c in log]
            a = st.selectbox("Versão A", ids, index=0)
            b = st.selectbox("Versão B", ids, index=len(ids) - 1)
            if st.button("Comparar (diff)"):
                st.code(repo.diff(a, b) or "(sem diferenças)", language="diff")

# --------------------------------------------------------------------------
# Página: Benchmark
# --------------------------------------------------------------------------

elif page == "Benchmark":
    st.title("⚖️ Prompt Benchmark")
    prompt_a = st.text_area("Prompt A", height=150)
    prompt_b = st.text_area("Prompt B", height=150)
    test_input = st.text_area("Entrada de teste (aplicada aos dois prompts)", height=100)
    if st.button("Rodar benchmark") and prompt_a and prompt_b and test_input:
        with st.spinner("Rodando os dois prompts e julgando os resultados..."):
            result = st.session_state.benchmark.compare(prompt_a, prompt_b, test_input, _llm())
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Score A", result["score_a"])
            st.text_area("Saída A", result["output_a"], height=200)
        with c2:
            st.metric("Score B", result["score_b"])
            st.text_area("Saída B", result["output_b"], height=200)
        st.success(f"Vencedor: {result['winner']} — {result['rationale']}")

# --------------------------------------------------------------------------
# Página: Memória
# --------------------------------------------------------------------------

elif page == "Memória":
    st.title("🧬 Memory & Learning")
    mem = st.session_state.memory_agent.get_user_memory(user_id)
    st.json(mem.model_dump(mode="json"))
    st.markdown("#### Ensinar uma preferência manualmente")
    c1, c2, c3 = st.columns(3)
    with c1:
        fmt = st.text_input("Preferência de formato", placeholder="ex: bullet points")
    with c2:
        style = st.text_input("Estilo de resposta", placeholder="ex: direto, técnico")
    with c3:
        interest = st.text_input("Área de interesse", placeholder="ex: renda fibra")
    if st.button("Salvar preferências"):
        st.session_state.memory_agent.update_preferences(
            user_id,
            format_preferences=[fmt] if fmt else None,
            response_style=[style] if style else None,
            interest_areas=[interest] if interest else None,
        )
        st.success("Preferências atualizadas.")