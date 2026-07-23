"""
frontend.app
=============
Interface Streamlit do Prompt Intelligence Engine — visual profissional e clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

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

# Inicializa session state
if "memory_agent" not in st.session_state:
    st.session_state.memory_agent = MemoryLearningAgent()
if "library" not in st.session_state:
    st.session_state.library = PromptLibrary()
if "benchmark" not in st.session_state:
    st.session_state.benchmark = PromptBenchmark()
if "session" not in st.session_state:
    st.session_state.session = None

# ==========================================================================
# CSS – Tema claro profissional (fixo)
# ==========================================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f5;
        color: #2c3e50;
    }
    .main-header {
        color: #003366;
        font-weight: 600;
    }
    .card {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        padding: 1.8rem 2rem;
        border: 1px solid #e4e7ec;
    }
    .btn-primary {
        background: #003366;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.8rem;
        font-weight: 500;
        transition: background 0.2s;
    }
    .btn-primary:hover {
        background: #004c99;
    }
    .btn-secondary {
        background: #ffffff;
        color: #003366;
        border: 1px solid #003366;
        border-radius: 8px;
        padding: 0.5rem 1.8rem;
    }
    .btn-secondary:hover {
        background: #eef5fa;
    }
    .pie-score {
        color: #003366;
        font-size: 2.8rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.2rem;
        border-bottom: 2px solid #e4e7ec;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #5a6c7e;
        border-radius: 0;
        padding: 0.5rem 1rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #003366;
        border-bottom: 2px solid #003366;
    }
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        color: #2c3e50;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #003366;
        box-shadow: 0 0 0 2px rgba(0,51,102,0.15);
    }
    .stButton button {
        background: #ffffff;
        color: #003366;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
    }
    .stButton button:hover {
        background: #f0f2f5;
        border-color: #003366;
    }
    .stButton button[data-testid="baseButton-primary"] {
        background: #003366;
        color: #ffffff;
        border: none;
    }
    .stButton button[data-testid="baseButton-primary"]:hover {
        background: #004c99;
    }
    .env-badge {
        background: #e4e7ec;
        color: #2c3e50;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e4e7ec;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Sidebar
# ==========================================================================

with st.sidebar:
    st.markdown("### 🧠 PIE")
    st.caption("Prompt Intelligence Engine")

    page = st.radio(
        "Navegação",
        ["Studio", "Biblioteca", "Versionamento", "Benchmark", "Memória"],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("⚙️ Provedor & Modelo", expanded=False):
        provider = st.selectbox(
            "Provider",
            ["claude", "gpt", "gemini", "open_source"],
            label_visibility="collapsed",
        )
        model = st.text_input("Modelo (opcional)", placeholder="ex: claude-sonnet-4-6", label_visibility="collapsed")
        user_id = st.text_input("User ID", value="default", label_visibility="collapsed")

    st.markdown(
        f"<div class='env-badge'>🔑 API keys do .env</div>",
        unsafe_allow_html=True,
    )

def _llm():
    return get_provider(provider, model or None)

# ==========================================================================
# PÁGINA: Studio (principal)
# ==========================================================================

if page == "Studio":
    st.markdown("<h1 class='main-header'>🧠 Prompt Intelligence Engine</h1>", unsafe_allow_html=True)
    st.caption("Transforme uma ideia simples em um prompt profissional, avaliado e otimizado.")

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        idea = st.text_area(
            "Descreva sua ideia",
            height=100,
            placeholder="Ex: quero um prompt para gerar um relatório semanal de risco de crédito...",
            label_visibility="collapsed",
        )
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            run = st.button("▶ Processar", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

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
        except Exception as e:
            progress.update(label="Erro no pipeline", state="error")
            st.error(f"Falha ao rodar o pipeline: {e}")

    session = st.session_state.session
    if session:
        st.divider()

        score = session.evaluation.score if session.evaluation else 0
        col_score, col_info = st.columns([1, 3])
        with col_score:
            color = "#00a65a" if score >= 80 else ("#f39c12" if score >= 60 else "#e74c3c")
            st.markdown(f"<div class='pie-score' style='color:{color}'>{score}</div>", unsafe_allow_html=True)
            st.caption("Score / 100")
            st.progress(score / 100)
            if session.refinement_iterations:
                st.caption(f"Refinado automaticamente {session.refinement_iterations}x")
        with col_info:
            st.subheader("Prompt final")
            st.code(session.final_prompt, language="markdown", line_numbers=False)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Avaliação", "📂 Versões", "💾 Salvar", "🔍 Explicação"])

        with tab1:
            if session.evaluation and session.evaluation.improvements:
                st.markdown("**Melhorias sugeridas**")
                for imp in session.evaluation.improvements:
                    st.markdown(f"- {imp}")
            else:
                st.caption("Nenhuma melhoria pendente relevante.")

        with tab2:
            if session.versions:
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("**Básica**")
                    st.code(session.versions.basic, language="markdown")
                with cols[1]:
                    st.markdown("**Profissional**")
                    st.code(session.versions.professional, language="markdown")
                with cols[2]:
                    st.markdown("**Expert**")
                    st.code(session.versions.expert, language="markdown")

                st.markdown("**Versões por modelo**")
                model_tabs = st.tabs(["GPT", "Claude", "Gemini", "Midjourney", "Open-source"])
                model_keys = ["gpt", "claude", "gemini", "midjourney", "open_source"]
                for tab, key in zip(model_tabs, model_keys):
                    with tab:
                        text = session.versions.model_specific.get(key, "(não gerado)")
                        st.code(text, language="markdown")
            else:
                st.caption("Nenhuma versão alternativa gerada.")

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                prompt_id = st.text_input("ID do prompt (para versionamento)", value=session.session_id[:8])
                msg = st.text_input("Mensagem do commit", value="Versão gerada pelo PIE")
                if st.button("💾 Commit versão"):
                    PromptRepo(prompt_id).commit(session.final_prompt, msg)
                    st.success("Commit registrado.")
            with c2:
                cat = st.selectbox("Categoria", CATEGORIES)
                title = st.text_input("Título", value=(session.intent.goal if session.intent else "Prompt PIE"))
                if st.button("📚 Adicionar à biblioteca"):
                    st.session_state.library.add(cat, title, session.final_prompt)
                    st.success("Adicionado à biblioteca.")

        with tab4:
            st.markdown("**Raciocínio por etapa**")
            for step in session.reasoning_log:
                st.markdown(f"**{step.step}**  \n{step.reasoning}")

# ==========================================================================
# PÁGINA: Biblioteca
# ==========================================================================

elif page == "Biblioteca":
    st.markdown("<h1 class='main-header'>📚 Biblioteca de Prompts</h1>", unsafe_allow_html=True)
    cat_filter = st.selectbox("Filtrar por categoria", ["Todas"] + CATEGORIES)
    entries = st.session_state.library.list(None if cat_filter == "Todas" else cat_filter)
    if not entries:
        st.caption("Nenhum prompt salvo ainda.")
    for e in entries:
        with st.expander(f"[{e['category']}] {e['title']}"):
            st.code(e["prompt_text"], language="markdown")

# ==========================================================================
# PÁGINA: Versionamento
# ==========================================================================

elif page == "Versionamento":
    st.markdown("<h1 class='main-header'>🕘 Versionamento</h1>", unsafe_allow_html=True)
    prompt_id = st.text_input("ID do prompt")
    if prompt_id:
        repo = PromptRepo(prompt_id)
        log = repo.log()
        if not log:
            st.caption("Nenhum commit encontrado.")
        else:
            for c in log:
                st.markdown(f"`{c['id'][:8]}` — {c['message']} — {c['created_at']}")
            st.divider()
            ids = [c["id"] for c in log]
            col_a, col_b = st.columns(2)
            with col_a:
                a = st.selectbox("Versão A", ids, index=0)
            with col_b:
                b = st.selectbox("Versão B", ids, index=len(ids)-1)
            if st.button("Comparar (diff)"):
                st.code(repo.diff(a, b) or "(sem diferenças)", language="diff")

# ==========================================================================
# PÁGINA: Benchmark
# ==========================================================================

elif page == "Benchmark":
    st.markdown("<h1 class='main-header'>⚖️ Benchmark</h1>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        prompt_a = st.text_area("Prompt A", height=150)
    with col_b:
        prompt_b = st.text_area("Prompt B", height=150)
    test_input = st.text_area("Entrada de teste", height=100)
    if st.button("Rodar benchmark") and prompt_a and prompt_b and test_input:
        with st.spinner("Processando..."):
            result = st.session_state.benchmark.compare(prompt_a, prompt_b, test_input, _llm())
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Score A", result["score_a"])
            st.text_area("Saída A", result["output_a"], height=200)
        with c2:
            st.metric("Score B", result["score_b"])
            st.text_area("Saída B", result["output_b"], height=200)
        st.success(f"Vencedor: {result['winner']} — {result['rationale']}")

# ==========================================================================
# PÁGINA: Memória
# ==========================================================================

elif page == "Memória":
    st.markdown("<h1 class='main-header'>🧬 Memória</h1>", unsafe_allow_html=True)
    mem = st.session_state.memory_agent.get_user_memory(user_id)
    st.json(mem.model_dump(mode="json"))
    st.markdown("#### Ensinar preferências")
    with st.form("memory_form"):
        fmt = st.text_input("Formato preferido", placeholder="ex: bullet points")
        style = st.text_input("Estilo de resposta", placeholder="ex: direto, técnico")
        interest = st.text_input("Área de interesse", placeholder="ex: renda fibra")
        if st.form_submit_button("Salvar preferências"):
            st.session_state.memory_agent.update_preferences(
                user_id,
                format_preferences=[fmt] if fmt else None,
                response_style=[style] if style else None,
                interest_areas=[interest] if interest else None,
            )
            st.success("Preferências atualizadas.")