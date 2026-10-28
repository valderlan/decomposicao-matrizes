"""Execute com: uv run streamlit run app.py."""

import json
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from matrixlab import METHODS, decompose, inspect_matrix, solve
from matrixlab.examples import EXAMPLES
from matrixlab.models import MethodError
from matrixlab.theory import family_for, theory_sections
from matrixlab.validation import parse_matrix, parse_vector, validate_matrix

st.set_page_config(
    page_title="Decomposições | Álgebra Linear", page_icon="🧮", layout="wide"
)


def latex_matrix(a, digits=5):
    array = np.asarray(a)
    if array.ndim == 1:
        array = array[:, None]

    def format_number(x):
        text = f"{x:.{digits}g}"
        if "e" in text:
            mantissa, exponent = text.split("e")
            return rf"{mantissa}\times 10^{{{int(exponent)}}}"
        return text

    rows = [" & ".join(format_number(x) for x in row) for row in array]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


ROLE_COLORS = {
    "pivot": "#b42318",
    "target": "#175cd3",
    "active": "#b54708",
    "updated": "#027a48",
    "reference": "#344054",
}


def highlight_map(entries):
    return {
        (int(item["row"]), int(item["col"])): item
        for item in entries
        if "row" in item and "col" in item
    }


def latex_matrix_highlighted(a, digits=5, entries=None):
    entries = entries or []
    marked = highlight_map(entries)
    array = np.asarray(a)
    if array.ndim == 1:
        array = array[:, None]

    def format_number(x):
        text = f"{x:.{digits}g}"
        if "e" in text:
            mantissa, exponent = text.split("e")
            return rf"{mantissa}\times 10^{{{int(exponent)}}}"
        return text

    rows = []
    for i, row in enumerate(array):
        row_entries = []
        for j, value in enumerate(row):
            text = format_number(value)
            marker = marked.get((i, j))
            if marker:
                color = ROLE_COLORS.get(
                    marker.get("role", "reference"), ROLE_COLORS["reference"]
                )
                text = rf"\color{{{color}}}{{\mathbf{{{text}}}}}"
            row_entries.append(text)
        rows.append(" & ".join(row_entries))
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def display_matrix(name, value, digits, highlights):
    clean_name = name.replace("_", " ")
    st.markdown(f"**{clean_name}** · {value.shape[0]} × {value.shape[1]}")
    st.latex(latex_matrix_highlighted(value, digits, highlights.get(name, [])))
    with st.expander(f"Valores de {clean_name} para copiar"):
        st.dataframe(pd.DataFrame(value), hide_index=True, width="stretch")


def sort_matrices(matrices, matrix_order):
    keys = list(matrices)
    if not matrix_order:
        return keys
    ordered = [name for name in matrix_order if name in matrices]
    ordered.extend(name for name in keys if name not in ordered)
    return ordered


def matrix_state(name):
    if name == "antes" or name.endswith("_antes"):
        return "before"
    if name == "depois" or name.endswith("_depois"):
        return "after"
    return "other"


def pair_name(name):
    if name == "antes":
        return "depois"
    if name.endswith("_antes"):
        return name[:-6] + "_depois"
    return None


def show_matrices(matrices, digits, highlights=None, matrix_order=None):
    highlights = highlights or {}
    ordered = sort_matrices(matrices, matrix_order or [])
    before_names = [name for name in ordered if matrix_state(name) == "before"]
    after_names = [name for name in ordered if matrix_state(name) == "after"]
    other_names = [name for name in ordered if matrix_state(name) == "other"]

    if before_names:
        st.markdown("#### Matriz antes")
        for name in before_names:
            display_matrix(name, matrices[name], digits, highlights)

    paired = []
    used_after = set()
    for before_name in before_names:
        expected = pair_name(before_name)
        if expected and expected in matrices:
            paired.append((before_name, expected))
            used_after.add(expected)

    remaining_after = [name for name in after_names if name not in used_after]

    if paired:
        st.markdown("#### Comparação antes × depois")
        for before_name, after_name in paired:
            left, right = st.columns(2)
            with left:
                display_matrix(before_name, matrices[before_name], digits, highlights)
            with right:
                display_matrix(after_name, matrices[after_name], digits, highlights)

    if after_names:
        st.markdown("#### Matriz depois")
        for name in (remaining_after if paired else after_names):
            display_matrix(name, matrices[name], digits, highlights)

    if other_names:
        st.markdown("#### Matrizes auxiliares")
        for name in other_names:
            display_matrix(name, matrices[name], digits, highlights)


def show_step(step, number, total, digits):
    st.markdown(f"### Passo {number} de {total} · {step.title}")
    if step.objective:
        st.markdown("**Objetivo**")
        st.write(step.objective)
    if step.operation:
        st.markdown("**Operação aplicada**")
        st.latex(step.operation)
    st.write(step.explanation)
    if step.formula and not step.operation:
        st.latex(step.formula)
    if step.calculations:
        with st.expander("Contas deste passo · valores substituídos", expanded=True):
            for calculation in step.calculations:
                st.latex(calculation)
    show_matrices(step.matrices, digits, step.highlights, step.matrix_order)
    if step.legend:
        st.markdown("**Legenda dos destaques**")
        for item in step.legend:
            st.write(f"- {item}")


def show_theory(body):
    """Renderize blocos matemáticos separadamente, preservando o LaTeX original."""
    for index, block in enumerate(re.split(r"\$\$(.*?)\$\$", body, flags=re.DOTALL)):
        if not block.strip():
            continue
        if index % 2:
            st.latex(block.strip())
        else:
            st.markdown(block)


def move_step(key, target, total):
    st.session_state[key] = max(1, min(target, total))


def step_navigator(steps, key, digits):
    total = len(steps)
    current = max(1, min(st.session_state.get(key, 1), total))
    st.session_state[key] = current
    first, previous, counter, next_step, last = st.columns([1, 1.2, 1.8, 1.2, 1])
    first.button(
        "Primeiro",
        key=f"{key}_first",
        disabled=current == 1,
        on_click=move_step,
        args=(key, 1, total),
        width="stretch",
    )
    previous.button(
        "← Anterior",
        key=f"{key}_previous",
        disabled=current == 1,
        on_click=move_step,
        args=(key, current - 1, total),
        width="stretch",
    )
    counter.markdown(f"**Passo {current} de {total}**")
    next_step.button(
        "Próximo →",
        key=f"{key}_next",
        disabled=current == total,
        on_click=move_step,
        args=(key, current + 1, total),
        type="primary",
        width="stretch",
    )
    last.button(
        "Último",
        key=f"{key}_last",
        disabled=current == total,
        on_click=move_step,
        args=(key, total, total),
        width="stretch",
    )
    show_step(steps[current - 1], current, total, digits)


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinito" if value > 0 else "não finito"
    return value


def latex_number(value, digits=10):
    text = f"{float(value):.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return text


def product_expression(values, digits=10):
    terms = [f"({latex_number(v, digits)})" for v in values]
    return r"\cdot".join(terms) if terms else "1"


def permutation_sign(matrix):
    return int(np.rint(np.linalg.det(matrix)))


def show_determinant_walkthrough(a, decomposition, digits, rtol):
    rows, cols = a.shape
    if rows != cols:
        st.info(
            "det(A) não existe para matrizes retangulares. Nesses casos, QR e SVD continuam "
            "válidas para fatoração e mínimos quadrados."
        )
        return

    n = rows
    u = a.astype(float).copy()
    sign = 1
    tol = rtol * max(1.0, float(np.linalg.norm(a, ord=np.inf)))

    st.markdown("**Cálculo de det(A) diretamente da matriz A**")
    st.write(
        "Usamos eliminação para triangularizar A. As operações de linha usadas são: "
        "troca de linhas (muda o sinal do determinante) e soma de múltiplo de linha "
        "(não altera o determinante)."
    )
    st.latex(r"L_i\leftrightarrow L_j\Rightarrow \det\to-\det")
    st.latex(r"L_i\leftarrow L_i-\lambda L_j\Rightarrow \det\ \text{inalterado}")
    st.markdown("Matriz inicial A")
    st.latex(latex_matrix(a, digits))

    singular = False
    step = 1
    for k in range(n):
        pivot_row = k + int(np.argmax(np.abs(u[k:, k])))
        pivot_candidate = float(u[pivot_row, k])

        with st.expander(f"Passo {step} · escolher pivô na coluna {k + 1}", expanded=step <= 2):
            st.write(
                f"No bloco ativo, escolhemos o pivô em ({pivot_row + 1}, {k + 1}) = {pivot_candidate:.10g}."
            )
            st.write(f"Limiar numérico para pivô: {tol:.3g}.")
            st.latex(
                rf"|u_{{{pivot_row + 1},{k + 1}}}|={latex_number(abs(pivot_candidate), digits)}"
            )
            st.latex(latex_matrix(u, digits))
        step += 1

        if abs(pivot_candidate) <= tol:
            singular = True
            break

        if pivot_row != k:
            before = u.copy()
            u[[k, pivot_row], :] = u[[pivot_row, k], :]
            sign *= -1
            with st.expander(
                f"Passo {step} · trocar linhas {k + 1} e {pivot_row + 1}",
                expanded=step <= 2,
            ):
                st.write("Troca de linhas muda o sinal do determinante.")
                st.latex(r"\det\leftarrow -\det")
                st.latex(rf"\text{{sinal acumulado}}={sign}")
                show_matrices(
                    {"U_antes": before, "U_depois": u},
                    digits,
                    matrix_order=["U_antes", "U_depois"],
                )
            step += 1

        pivot = float(u[k, k])
        for i in range(k + 1, n):
            if abs(u[i, k]) <= tol:
                continue
            before = u.copy()
            multiplier = float(u[i, k] / pivot)
            u[i, k:] = u[i, k:] - multiplier * u[k, k:]
            u[i, k] = 0.0
            with st.expander(
                f"Passo {step} · eliminar entrada ({i + 1}, {k + 1})",
                expanded=step <= 2,
            ):
                st.write(
                    f"Aplicamos L{i + 1} ← L{i + 1} − ({multiplier:.10g})L{k + 1}. "
                    "Esta operação não altera o determinante."
                )
                st.latex(
                    rf"L_{{{i + 1}}}\leftarrow L_{{{i + 1}}}-({latex_number(multiplier, digits)})L_{{{k + 1}}}"
                )
                st.latex(r"\det\ \text{inalterado}")
                show_matrices(
                    {"U_antes": before, "U_depois": u},
                    digits,
                    matrix_order=["U_antes", "U_depois"],
                )
            step += 1

    det_direct = float(np.linalg.det(a))
    sign_direct, logabs_direct = np.linalg.slogdet(a)

    if singular:
        det_elimination = 0.0
        st.warning(
            "Encontramos pivô numericamente nulo durante a eliminação. Assim, det(A)=0 "
            "no critério numérico adotado."
        )
    else:
        diagonal = np.diag(u)
        det_triangular = float(np.prod(diagonal))
        det_elimination = float(sign * det_triangular)
        st.markdown("**Passo final · produto da diagonal da triangular**")
        st.latex(r"\det(A)=\text{sinal}\cdot\prod_{i=1}^{n}u_{ii}")
        st.latex(
            rf"\det(U)={product_expression(diagonal, digits)}\approx {latex_number(det_triangular, digits)}"
        )
        st.latex(
            rf"\det(A)\approx ({sign})\cdot {latex_number(det_triangular, digits)}={latex_number(det_elimination, digits)}"
        )

    st.markdown("**Conferência com NumPy**")
    st.latex(
        rf"\det(A)_{{eliminacao}}\approx {latex_number(det_elimination, digits)},\quad "
        rf"\det(A)_{{numpy}}\approx {latex_number(det_direct, digits)}"
    )
    if sign_direct != 0:
        st.latex(
            rf"\operatorname{{sign}}(\det(A))={int(sign_direct)},\quad "
            rf"\log\left(|\det(A)|\right)\approx {latex_number(logabs_direct, digits)}"
        )
    abs_error = abs(det_elimination - det_direct)
    rel_error = abs_error / max(abs(det_direct), np.finfo(float).tiny)
    st.latex(
        rf"|\Delta|\approx {latex_number(abs_error, digits)},\quad "
        rf"\text{{erro relativo}}\approx {latex_number(rel_error, digits)}"
    )

    st.markdown("**Propriedades do determinante e impacto nas decomposições**")
    st.write("- det(A)=0 implica matriz singular: não há inversa e não há solução única garantida para Ax=b.")
    st.write("- det(A)≠0 (com A quadrada) implica invertibilidade e solução única para Ax=b.")
    st.write("- Em LU com pivotamento, cada troca de linha muda o sinal do determinante.")
    st.write(
        "- Em Cholesky/LDLᵀ desta aplicação, a hipótese é positiva definida: isso força det(A)>0 "
        "(e, teoricamente, todos os menores principais líderes positivos)."
    )
    st.write(
        "- QR e SVD existem mesmo quando det(A)=0; por isso continuam úteis em posto deficiente."
    )
    st.write(
        "- |det(A)| muito pequeno sugere proximidade de singularidade, mas condicionamento (κ₂) e "
        "valores singulares descrevem melhor a sensibilidade numérica."
    )


def load_example():
    example = EXAMPLES[st.session_state.example]
    st.session_state.matrix_source_text = example["matrix"]
    st.session_state.b_source_text = example["b"]
    st.session_state.matrix_text = example["matrix"]
    st.session_state.b_text = example["b"]
    array = parse_matrix(example["matrix"])
    st.session_state.grid_rows, st.session_state.grid_cols = array.shape
    st.session_state.grid_base = array
    st.session_state.grid_last = array
    st.session_state.grid_generation = st.session_state.get("grid_generation", 0) + 1
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("decomposition_result", None)


def remember_matrix():
    st.session_state.matrix_source_text = st.session_state.matrix_text


def remember_b():
    st.session_state.b_source_text = st.session_state.b_text


def switch_mode():
    if st.session_state.input_mode == "Grade":
        try:
            array = parse_matrix(st.session_state.matrix_source_text)
        except ValueError:
            array = st.session_state.grid_last
        st.session_state.grid_base = array
        st.session_state.grid_rows, st.session_state.grid_cols = array.shape
        st.session_state.grid_generation += 1
    else:
        array = st.session_state.grid_last
        text = "\n".join(" ".join(f"{x:.17g}" for x in row) for row in array)
        st.session_state.matrix_source_text = text
        st.session_state.matrix_text = text


# default_example = next(iter(EXAMPLES))
default_example = "LU • exemplo da Seção 1.3 (3 × 3)"
if "initialized" not in st.session_state:
    st.session_state.example = default_example
    load_example()
    st.session_state.initialized = True

st.session_state.setdefault("matrix_text", st.session_state.matrix_source_text)
st.session_state.setdefault("b_text", st.session_state.b_source_text)
st.session_state.setdefault("grid_rows", st.session_state.grid_base.shape[0])
st.session_state.setdefault("grid_cols", st.session_state.grid_base.shape[1])

st.title("Laboratório de decomposições matriciais")
st.caption(
    "Álgebra Linear Numérica · Entrada → diagnóstico → escolha do método → passos → verificação"
)

with st.sidebar:
    st.subheader("Configuração")
    st.selectbox("Exemplo", list(EXAMPLES), key="example", on_change=load_example)
    st.caption(EXAMPLES[st.session_state.example]["description"])
    exponent = st.select_slider(
        "Tolerância relativa · 10⁻ᵖ", options=list(range(4, 16)), value=12
    )
    rtol = 10.0 ** (-exponent)
    digits = st.slider("Algarismos exibidos", 3, 12, 6)
    st.caption(
        "A tolerância afeta os testes numéricos. O arredondamento da exibição não altera os cálculos."
    )
    st.info(
        "Toda matriz real válida admite SVD. Falha de um método não significa que a matriz não possa ser fatorada."
    )

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("1 · Informe A")
    mode = st.radio(
        "Forma de entrada",
        ["Texto", "Grade"],
        horizontal=True,
        key="input_mode",
        on_change=switch_mode,
    )
    if mode == "Texto":
        st.text_area(
            "Matriz A", key="matrix_text", height=160, on_change=remember_matrix
        )
        st.caption(
            "Uma linha por linha; elementos separados por espaços ou vírgulas. Decimais com ponto. Frações como 1/3 e JSON são aceitos."
        )
    else:
        c1, c2 = st.columns(2)
        c1.number_input("Linhas", 1, 10, key="grid_rows")
        c2.number_input("Colunas", 1, 10, key="grid_cols")
        shape = (st.session_state.grid_rows, st.session_state.grid_cols)
        grid_base = st.session_state.grid_base
        if grid_base.shape != shape:
            grid_base = np.zeros(shape)
        grid = st.data_editor(
            pd.DataFrame(grid_base, columns=[f"c{i + 1}" for i in range(shape[1])]),
            key=f"grid_{shape}_{st.session_state.grid_generation}",
            num_rows="fixed",
            hide_index=True,
            width="stretch",
        )
        st.session_state.grid_last = grid.to_numpy(dtype=float)
    with_b = st.checkbox("Resolver também Ax=b", value=False)
    st.text_area(
        "Vetor b", key="b_text", height=70, on_change=remember_b, disabled=not with_b
    )
    analyze = st.button("Analisar matriz", type="primary", width="stretch")

try:
    a = (
        parse_matrix(st.session_state.matrix_text)
        if mode == "Texto"
        else validate_matrix(grid.to_numpy())
    )
except ValueError as exc:
    right.error(str(exc))
    st.stop()

signature = (a.shape, a.tobytes(), rtol)
with right:
    st.subheader("Prévia da matriz")
    st.latex(latex_matrix(a, digits))
    st.caption(
        f"{a.shape[0]} equações · {a.shape[1]} variáveis · limite didático de 10 × 10"
    )

if analyze:
    with st.spinner("Verificando propriedades e condições dos algoritmos…"):
        st.session_state.analysis_result = inspect_matrix(a, rtol)
        st.session_state.analysis_signature = signature
        st.session_state.pop("decomposition_result", None)

if (
    st.session_state.get("analysis_signature") != signature
    or "analysis_result" not in st.session_state
):
    st.info(
        "Clique em Analisar matriz para verificar os métodos aplicáveis à entrada atual."
    )
    st.stop()

diagnostic = st.session_state.analysis_result
props = diagnostic["properties"]
st.divider()
st.subheader("2 · Diagnóstico e escolha")
metrics = st.columns(5)
metrics[0].metric("Dimensão", f"{props['rows']} × {props['columns']}")
metrics[1].metric("Posto numérico", props["rank"])
metrics[2].metric("Simétrica", "Sim" if props["symmetric"] else "Não")
metrics[3].metric(
    "Condicionamento κ₂",
    (
        f"{props['condition_2']:.3g}"
        if math.isfinite(props["condition_2"])
        else "∞ (numérico)"
    ),
)
det_value = props.get("determinant")
metrics[4].metric(
    "det(A)",
    (f"{det_value:.3g}" if det_value is not None else "N/A (retangular)"),
)
with st.expander("Propriedades e interpretação"):
    st.write(
        f"Quadrada: {'sim' if props['square'] else 'não'} · Positiva definida (teste numérico): {'sim' if props['positive_definite'] else 'não'}."
    )
    st.write(
        f"Norma Frobenius: {props['norm_frobenius']:.8g} · Norma infinito: {props['norm_inf']:.8g}."
    )
    st.write(
        f"Posto: contamos valores singulares maiores que {props['rank_cutoff']:.5g}."
    )
    for message in diagnostic["warnings"]:
        st.warning(message)

with st.expander("Determinante: cálculo e relação com decomposições"):
    if det_value is None:
        st.info(
            "det(A) é definido apenas para matrizes quadradas. Para matrizes retangulares, "
            "QR e SVD continuam válidas, mas sem determinante clássico."
        )
    else:
        det_sign = props.get("det_sign")
        det_logabs = props.get("det_logabs")
        st.write(
            "O painel calcula det(A) com np.linalg.det e também mostra sign/log|det| "
            "com np.linalg.slogdet para interpretação numérica estável."
        )
        st.latex(rf"\det(A)\approx {det_value:.10g}")
        if det_sign == 0:
            st.write(
                "Sign = 0 indica singularidade numérica: não há inversa e pode não existir solução única para Ax=b."
            )
        elif det_logabs is not None:
            st.latex(
                rf"\operatorname{{sign}}(\det(A))={int(det_sign)},\quad \log\left(|\det(A)|\right)\approx {det_logabs:.10g}"
            )

        st.markdown("**Como calcular via decomposição**")
        st.latex(r"A=LU\Rightarrow\det(A)=\det(L)\det(U)")
        st.latex(r"PA=LU\Rightarrow\det(A)=\det(P)\det(L)\det(U),\ \det(P)\in\{-1,+1\}")
        st.latex(r"PAC=LU\Rightarrow\det(A)=\det(P)\det(C)\det(L)\det(U)")
        st.latex(r"A=LL^T\Rightarrow\det(A)=\det(L)^2>0")
        st.latex(r"A=LDL^T\Rightarrow\det(A)=\det(D)\ \text{(L unitária)}")
        st.latex(r"A=QR\Rightarrow\det(A)=\det(Q)\det(R),\ |\det(Q)|=1\ \text{(A quadrada)}")
        st.latex(r"A=U\Sigma V^T\Rightarrow|\det(A)|=\prod_i\sigma_i\ \text{(A quadrada)}")

        st.markdown("**Propriedades importantes**")
        st.write("- det(A)=0 implica matriz singular e não invertível.")
        st.write("- det(A)≠0 implica matriz invertível e solução única para Ax=b.")
        st.write("- Cada troca de linha/coluna em pivotamento muda o sinal do determinante.")
        st.write("- Em Cholesky de matriz positiva definida, o determinante é sempre positivo.")
        st.write("- Em SVD quadrada, valores singulares pequenos tornam |det(A)| pequeno.")
table = pd.DataFrame(
    [
        {
            "Método": entry["name"],
            "Situação": "Disponível" if entry["available"] else "Bloqueado",
            "Explicação": entry["reason"],
        }
        for entry in diagnostic["methods"]
    ]
)
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={"Explicação": st.column_config.TextColumn(width="large")},
)
available = [entry["id"] for entry in diagnostic["methods"] if entry["available"]]
if not available:
    st.error(
        "Nenhum algoritmo desta implementação conseguiu prosseguir sob as condições atuais. "
        "Veja as explicações acima. Isso não prova inexistência matemática: SVD existe para "
        "uma matriz real válida, mas uma execução pode falhar numericamente."
    )
    st.stop()

method = st.selectbox(
    "Método de decomposição", available, format_func=METHODS.get, key="method_choice"
)
current_b = st.session_state.b_text if with_b else None
result_signature = (signature, method, current_b)
if st.button("Decompor e mostrar os passos", type="primary"):
    st.session_state.pop("decomposition_result", None)
    try:
        b = parse_vector(current_b, len(a)) if with_b else None
        result = decompose(a, method, rtol)
        solution = None
        solution_error = None
        if b is not None:
            try:
                solution = solve(a, b, result, rtol)
            except MethodError as exc:
                solution_error = str(exc)
        st.session_state.decomposition_result = result
        st.session_state.solution_result = solution
        st.session_state.solution_error = solution_error
        st.session_state.result_signature = result_signature
        st.session_state.step_index = 1
        st.session_state.solution_step_index = 1
    except (ValueError, MethodError) as exc:
        st.error(str(exc))

if (
    st.session_state.get("result_signature") != result_signature
    or "decomposition_result" not in st.session_state
):
    st.stop()

result = st.session_state.decomposition_result
solution = st.session_state.solution_result
st.divider()
st.subheader("3 · Resultado e explicação")
st.latex(result.identity)
st.caption(
    f"{METHODS[method]} · {len(result.steps)} passos registrados · erro relativo de reconstrução {result.relative_error:.3g}"
)
for warning in result.warnings:
    st.warning(warning)
if not result.warnings:
    st.success(
        "Decomposição concluída. A matriz foi reconstruída a partir dos fatores para verificar o resultado."
    )

factors_tab, steps_tab, solution_tab, compare_tab, theory_tab = st.tabs(
    [
        "Fatores finais",
        "Passo a passo",
        "Solução de Ax=b",
        "Comparação",
        "Guia teórico",
    ]
)
with factors_tab:
    show_matrices(result.factors, digits)
    if result.orthogonality_error is not None:
        st.write(
            f"Erro de ortogonalidade (Frobenius): {result.orthogonality_error:.6g}."
        )
    if method == "lu_total":
        st.info(
            "No pivotamento total: PAC=LU. A matriz original é PᵀLUCᵀ e a solução original é x=Cz."
        )
    st.divider()
    st.markdown("### Determinante · passo a passo")
    st.caption(
        "Contas do determinante diretamente da matriz A e propriedades para interpretar as decomposições."
    )
    show_determinant_walkthrough(a, result, digits, rtol)
with steps_tab:
    st.caption(
        "Clique em Próximo para acompanhar os cálculos; Anterior permite revisar. Primeiro e Último levam aos extremos."
    )
    step_navigator(result.steps, "step_index", digits)
    with st.expander("Lista de todas as operações"):
        for index, step in enumerate(result.steps, 1):
            st.write(f"{index}. {step.title}: {step.explanation}")
with solution_tab:
    if not with_b:
        st.info(
            "Habilite Resolver também Ax=b e execute novamente para acompanhar a solução."
        )
    elif st.session_state.solution_error:
        st.warning(st.session_state.solution_error)
        st.info(
            "A fatoração acima permanece válida. A restrição é do procedimento de resolução escolhido."
        )
    elif solution:
        st.write(solution.kind)
        st.latex("x=" + latex_matrix(solution.x, digits))
        c1, c2 = st.columns(2)
        c1.metric("‖Ax−b‖₂", f"{solution.residual_norm:.6g}")
        c2.metric("Resíduo relativo escalado", f"{solution.relative_residual:.6g}")
        if solution.relative_residual <= 100 * rtol:
            st.success(
                "A solução satisfaz Ax≈b sob o critério numérico de resíduo usado."
            )
        else:
            st.info(
                "Há resíduo significativo. Em mínimos quadrados, isto pode indicar que não há solução exata, ou que a tolerância descartou direções pequenas."
            )
        step_navigator(solution.steps, "solution_step_index", digits)
with compare_tab:
    st.write(
        "Comparação teórica para matrizes densas. Não são tempos medidos neste computador."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Família": "LU",
                    "Hipótese usual para solução única": "Quadrada; pivôs não nulos, com pivotamento se necessário",
                    "Custo dominante": "≈ 2n³/3",
                    "Uso": "Sistemas quadrados; reutilizar fatores para vários b",
                },
                {
                    "Família": "Cholesky / LDLᵀ SPD",
                    "Hipótese usual para solução única": "Simétrica positiva definida",
                    "Custo dominante": "≈ n³/3",
                    "Uso": "Aproveitar simetria; menos operações que LU",
                },
                {
                    "Família": "QR Householder",
                    "Hipótese usual para solução única": "m≥n e posto completo de colunas",
                    "Custo dominante": "≈ 2mn² − 2n³/3",
                    "Uso": "Mínimos quadrados; boa estabilidade",
                },
                {
                    "Família": "SVD",
                    "Hipótese usual para solução única": "Nenhuma para fatorar; unicidade depende do posto",
                    "Custo dominante": "O(mn·min(m,n)) em algoritmos padrão",
                    "Uso": "Posto deficiente, pseudoinversa, norma mínima",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Os custos clássicos não incluem registrar matrizes, construir Q completo ou interface. "
        "Esta QR usa matrizes ortogonais explícitas por clareza e pode custar mais. "
        "A SVD implementada forma AᵀA em O(mn²) e resolve o problema simétrico de ordem n "
        "em O(n³); esse caminho didático pode perder precisão nas direções pequenas. "
        "O polinômio e o registro das contas acrescentam trabalho à execução com passos."
    )
    st.write(
        "Um resíduo pequeno verifica o sistema calculado, mas o condicionamento determina a sensibilidade da solução. Gram–Schmidt clássico e SVD pela gramiana podem perder ortogonalidade em casos difíceis; examine os avisos e as verificações."
    )
with theory_tab:
    st.caption(
        "Abra as seções para estudar conceitos, cálculos e exemplos. Fundamentos e a família "
        "do método escolhido começam abertos. QR e SVD adaptam as dimensões à matriz atual; "
        "na seção QR, a referência é o método QR escolhido ou Householder quando outra família está selecionada."
    )
    for family, title, body in theory_sections(method, a.shape):
        with st.expander(title, expanded=family in {"Fundamentos", family_for(method)}):
            show_theory(body)

payload = {
    "format_version": "1.0",
    "matrix": a.tolist(),
    "b": parse_vector(current_b, len(a)).tolist() if with_b else None,
    "diagnostic": diagnostic,
    "decomposition": result.to_dict(),
    "solution": solution.to_dict() if solution else None,
    "solution_error": st.session_state.solution_error,
}
st.download_button(
    "Baixar resultado e todos os passos (JSON)",
    data=json.dumps(safe_json(payload), ensure_ascii=False, indent=2, allow_nan=False),
    file_name=f"decomposicao_{method}.json",
    mime="application/json",
)
