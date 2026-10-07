from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def app():
    result = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not result.exception
    return result


def analyze(result):
    result.button[0].click().run()
    assert not result.exception
    return result


def test_dashboard_complete_flow_and_input_invalidation():
    result = analyze(app())
    assert len(result.dataframe) >= 1
    assert result.selectbox(key="method_choice").options
    result.selectbox(key="method_choice").set_value("cholesky").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception
    assert result.session_state["decomposition_result"].method == "cholesky"
    assert result.session_state["solution_result"] is not None
    assert result.success
    result.text_area[0].set_value("1 2\n2 4").run()
    assert not result.exception
    # The old factors are not rendered for a changed matrix.
    assert not result.tabs
    assert any("Analisar" in x.value for x in result.info)


def test_example_singular_and_solver_error_preserves_factors():
    result = app()
    result.selectbox(key="example").set_value("Singular consistente (3 × 3)").run()
    result = analyze(result)
    assert "Cholesky • LLᵀ" not in result.selectbox(key="method_choice").options
    result.selectbox(key="method_choice").set_value("lu_partial").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception
    assert result.session_state["decomposition_result"] is not None
    assert result.session_state["solution_error"]
    assert result.warning


def test_invalid_input():
    result = app()
    result.text_area[0].set_value("1 2\n3").run()
    assert not result.exception
    assert result.error


def test_grid_mode_and_svd_wide():
    result = app()
    result.selectbox(key="example").set_value("Subdeterminado (2 × 3)").run()
    result.radio[0].set_value("Grade").run()
    result = analyze(result)
    result.selectbox(key="method_choice").set_value("svd").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception
    assert result.session_state["solution_result"].relative_residual < 1e-10
    assert result.session_state["analysis_result"]["properties"]["rows"] == 2
    assert result.session_state["analysis_result"]["properties"]["columns"] == 3


def test_text_grid_roundtrip_and_b_toggle_preserve_user_data():
    result = app()
    result.text_area[0].set_value("2 0\n0 3").run()
    result.radio[0].set_value("Grade").run()
    result = analyze(result)
    assert result.session_state["analysis_result"]["properties"]["rows"] == 2
    result.radio[0].set_value("Texto").run()
    assert result.text_area[0].value == "2 0\n0 3"
    result.checkbox[0].check().run()
    result.text_area[1].set_value("5 6").run()
    result.checkbox[0].uncheck().run()
    result.checkbox[0].check().run()
    assert result.text_area[1].value == "5 6"


def test_button_navigation_has_boundaries_and_separate_solution_state():
    result = analyze(app())
    result.selectbox(key="method_choice").set_value("qr_modified").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception
    assert result.button(key="step_index_previous").disabled
    assert result.button(key="step_index_first").disabled
    assert all("passo" not in slider.label.lower() for slider in result.slider)
    result.button(key="step_index_next").click().run()
    assert not result.exception
    assert result.session_state["step_index"] == 2
    assert result.session_state["solution_step_index"] == 1
    result.button(key="solution_step_index_next").click().run()
    assert result.session_state["solution_step_index"] == 2
    assert result.session_state["step_index"] == 2
    result.button(key="step_index_last").click().run()
    assert result.session_state["step_index"] == len(
        result.session_state["decomposition_result"].steps
    )
    assert result.button(key="step_index_next").disabled
    result.button(key="step_index_previous").click().run()
    assert not result.button(key="step_index_next").disabled
    result.button(key="step_index_first").click().run()
    assert result.session_state["step_index"] == 1
    # A new decomposition starts from the first step in both navigators.
    result.button[1].click().run()
    assert result.session_state["step_index"] == 1
    assert result.session_state["solution_step_index"] == 1


def test_svd_lesson_and_button_navigation_for_wide_matrix():
    result = app()
    result.selectbox(key="example").set_value("Subdeterminado (2 × 3)").run()
    result = analyze(result)
    result.selectbox(key="method_choice").set_value("svd").run()
    result.button[1].click().run()
    assert not result.exception
    assert any("Aula 18" in x.value for x in result.markdown)
    result.button(key="step_index_last").click().run()
    assert not result.exception
    result.button(key="step_index_first").click().run()
    assert result.session_state["step_index"] == 1


def test_theory_sections_and_handout_lu_example():
    result = app()
    result.selectbox(key="example").set_value("LU • exemplo da Seção 1.3 (3 × 3)").run()
    result = analyze(result)
    result.selectbox(key="method_choice").set_value("lu_doolittle").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception

    assert result.session_state["solution_result"].x.tolist() == [2.0, 1.0, -1.0]
    headings = [x.label for x in result.expander]
    for fragment in [
        "1 · Fundamentos",
        "2 · LU",
        "3 · Cholesky",
        "4 · QR",
        "5 · SVD",
        "6 · Aplicações",
        "7 · Precisão",
    ]:
        assert any(fragment in heading for heading in headings)
    assert any("Seção 1.3" in item.value for item in result.markdown)
    assert any("Exemplo simples" in item.value for item in result.markdown)
    result.button(key="step_index_next").click().run()
    assert not result.exception


def test_integer_example_and_theory_use_separate_latex_blocks():
    result = app()
    result.selectbox(key="example").set_value("LU • fatores inteiros (3 × 3)").run()
    result = analyze(result)
    result.selectbox(key="method_choice").set_value("lu_doolittle").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception

    assert result.session_state["solution_result"].x.tolist() == [1.0, 2.0, 3.0]
    assert set(result.session_state["decomposition_result"].factors) == {"L", "U"}
    guide = result.tabs[4]
    assert any(r"2a+c=5" in item.value for item in guide.latex)
    assert any(r"\begin{bmatrix}5\\11\end{bmatrix}" in item.value for item in guide.latex)
    assert any("caderno" in item.value for item in guide.markdown)
    assert all("$$" not in item.value for item in guide.markdown)
    assert any("| Propriedade |" in item.value for item in guide.markdown)
    result.button(key="solution_step_index_last").click().run()
    assert not result.exception


def test_svd_dashboard_shows_spectral_stages_and_numeric_calculations():
    result = app()
    result.selectbox(key="example").set_value("SVD • autovetores e fatores simples (2 × 2)").run()
    result = analyze(result)
    result.selectbox(key="method_choice").set_value("svd").run()
    result.checkbox[0].check().run()
    result.button[1].click().run()
    assert not result.exception
    assert "autovalores e autovetores" in result.selectbox(key="method_choice").options[-1]
    decomposition = result.session_state["decomposition_result"]
    assert any("autovalores_escalados" in step.matrices for step in decomposition.steps)
    result.button(key="step_index_next").click().run()
    assert any("Contas deste passo" in item.label for item in result.expander)
    assert any(r"B_{1,1}" in item.value for item in result.latex)
    assert any("calculations" in step for step in decomposition.to_dict()["steps"])
