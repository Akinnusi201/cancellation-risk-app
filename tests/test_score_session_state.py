from pathlib import Path


def test_simulation_score_is_defensively_initialized():
    text = Path("views/2_Score_Order.py").read_text(encoding="utf-8")
    assert 'st.session_state.get("current_score")' in text
    assert 'not isinstance(cached_score, dict)' in text
    assert '"probability" not in cached_score' in text
    assert "st.session_state.current_score" not in text


def test_logout_clears_both_simulation_queue_indices():
    text = Path("src/auth.py").read_text(encoding="utf-8")
    assert '"live_queue_index"' in text
    assert '"historical_queue_index"' in text
