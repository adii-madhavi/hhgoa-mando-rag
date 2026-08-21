import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_desktop_shell_uses_full_width_grid_and_full_height_sidebar():
    assert "grid-template-columns: 220px minmax(0, 1fr)" in FRONTEND
    assert "height: 100dvh" in FRONTEND
    assert "min-height: 100vh" in FRONTEND
    assert "max-width: 100vw" in FRONTEND
    assert "overflow-x: clip" in FRONTEND
    assert 'class="mando-mobile-spotlight home-mando-companion"' in FRONTEND
    assert "max-width: 860px" in FRONTEND


def test_explore_opens_curated_supported_details_before_querying():
    topic_keys = re.findall(r"openExploreTopic\('([^']+)'\)", FRONTEND)
    assert topic_keys == [
        "basilica", "xit-kodi", "palolem", "old-goa",
        "shigmo", "dudhsagar", "ponda-temples", "architecture",
    ]
    assert "const EXPLORE_TOPICS" in FRONTEND
    assert "currentExploreQuery = topic.query" in FRONTEND
    assert "textContent = topic.text" in FRONTEND
    assert "askQuestion(currentExploreQuery)" in FRONTEND


def test_clickable_prompts_do_not_offer_unsupported_specifics():
    clickable_queries = re.findall(r"askQuestion\('([^']+)'\)", FRONTEND)
    joined = " ".join(clickable_queries).casefold()
    unsupported = [
        "agonda", "kopel", "siolim", "ghumot", "spice plantation",
        "oyster", "balc", "romi", "devanagari", "kadamba",
        "satavahana", "ghode modni", "usgalimal", "st francis xavier",
        "best shigmo", "festival foods", "top must-visit",
    ]
    assert not [term for term in unsupported if term in joined]


def test_followups_use_same_real_query_path_and_preserve_preferences():
    assert 'id="followup-chips"' in FRONTEND
    assert "container.replaceChildren()" in FRONTEND
    assert "button.textContent = followUpQuery" in FRONTEND
    assert "button.addEventListener('click', () => askQuestion(followUpQuery))" in FRONTEND
    assert "renderSupportedFollowUps(query)" in FRONTEND
    assert "switchNavTab('mando')" in FRONTEND
    assert "executeQuery(q)" in FRONTEND
    assert (
        "JSON.stringify({ query: query, language: activeLanguage, "
        "answer_mode: answerMode })"
    ) in FRONTEND


def test_followups_wrap_and_response_integrity_guards_remain():
    assert ".followup-box .prompt-chips-scroll" in FRONTEND
    assert "flex-wrap: wrap" in FRONTEND
    assert "currentSources = data.sources || []" in FRONTEND
    assert "sourcesBar.hidden = isExternal || refused" in FRONTEND
    assert "renderSourcesInSheet()" in FRONTEND
    assert "escapeHtml(currentAnswer)" in FRONTEND
    assert "escapeHtml(query)" in FRONTEND


def test_no_fake_answer_or_unrelated_fallback_is_introduced():
    assert 'let currentAnswer = ""' in FRONTEND
    assert "const MOCK" not in FRONTEND
    assert "const fallbackQuestion" not in FRONTEND
    assert "fetch('/api/text/query'" in FRONTEND


def test_followup_reuses_normal_query_and_response_state_updates():
    ask_flow = FRONTEND[FRONTEND.index("function askQuestion"):FRONTEND.index("function submitComposerQuery")]
    assert "user-question-text" in ask_flow
    assert "rag-step-q-label" in ask_flow
    assert "executeQuery(q)" in ask_flow

    render_flow = FRONTEND[FRONTEND.index("function renderBackendResponse"):FRONTEND.index("function renderPerformance")]
    assert "answer-provenance" in render_flow
    assert "renderSourcesInSheet()" in render_flow
    assert "renderPerformance(data)" in render_flow


def test_responsive_breakpoints_keep_layout_bounded():
    assert "@media (min-width: 1024px)" in FRONTEND
    assert "grid-template-columns: repeat(4, 1fr)" in FRONTEND
    assert "grid-template-columns: repeat(2, 1fr)" in FRONTEND
    assert "min-width: 0" in FRONTEND
