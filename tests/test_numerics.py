import numpy as np
import pytest

from matrixlab import METHODS, decompose, inspect_matrix, solve
from matrixlab.models import MethodError
from matrixlab.validation import parse_matrix, parse_vector

SPD = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])


def rebuild(result):
    f, name = result.factors, result.method
    if name.startswith("lu_"):
        identity = np.eye(f["L"].shape[0])
        return f.get("P", identity).T @ f["L"] @ f["U"] @ f.get("C", identity).T
    if name == "cholesky":
        return f["L"] @ f["L"].T
    if name == "ldlt":
        return f["L"] @ f["D"] @ f["L"].T
    if name.startswith("qr_"):
        return f["Q"] @ f["R"]
    return f["U"] @ f["Sigma"] @ f["Vt"]


@pytest.mark.parametrize("method", METHODS)
def test_spd_reconstruction_structure_and_solution(method):
    result = decompose(SPD, method)
    np.testing.assert_allclose(rebuild(result), SPD, rtol=1e-10, atol=1e-10)
    assert result.relative_error < 1e-10
    assert result.steps
    f = result.factors
    if "L" in f:
        np.testing.assert_allclose(f["L"], np.tril(f["L"]))
    if method.startswith("lu_"):
        np.testing.assert_allclose(f["U"], np.triu(f["U"]))
        if method == "lu_crout":
            np.testing.assert_allclose(np.diag(f["U"]), 1)
        else:
            np.testing.assert_allclose(np.diag(f["L"]), 1)
    if method.startswith("qr_"):
        np.testing.assert_allclose(
            f["Q"].T @ f["Q"], np.eye(f["Q"].shape[1]), atol=1e-10
        )
        np.testing.assert_allclose(f["R"], np.triu(f["R"]), atol=1e-10)
    x = np.array([1.0, 2.0, 3.0])
    solution = solve(SPD, SPD @ x, result)
    np.testing.assert_allclose(solution.x, x, atol=1e-9)
    assert solution.relative_residual < 1e-10


@pytest.mark.parametrize(
    "shape", [(1, 1), (2, 2), (3, 3), (5, 3), (3, 5), (1, 4), (4, 1), (10, 10)]
)
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_svd_general(shape, rank_deficient):
    rng = np.random.default_rng(42)
    a = rng.normal(size=shape)
    if rank_deficient:
        if min(shape) == 1:
            a[:] = 0
        else:
            a[:, -1] = a[:, 0]
            if shape[0] < shape[1]:
                a[-1, :] = a[0, :]
    result = decompose(a, "svd")
    f = result.factors
    np.testing.assert_allclose(rebuild(result), a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.diag(f["Sigma"]), np.linalg.svd(a, compute_uv=False), atol=1e-10
    )
    k = min(shape)
    np.testing.assert_allclose(f["U"].T @ f["U"], np.eye(k), atol=1e-10)
    np.testing.assert_allclose(f["Vt"] @ f["Vt"].T, np.eye(k), atol=1e-10)
    b = rng.normal(size=shape[0])
    solution = solve(a, b, result)
    expected = np.linalg.pinv(a, rcond=1e-12) @ b
    np.testing.assert_allclose(solution.x, expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(a.T @ (a @ solution.x - b), 0, atol=1e-9)


@pytest.mark.parametrize("method", ["qr_householder", "qr_givens"])
@pytest.mark.parametrize(
    "a",
    [
        np.zeros((3, 2)),
        np.zeros((2, 3)),
        np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
        np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
    ],
)
def test_qr_rank_deficient_and_rectangular(method, a):
    result = decompose(a, method)
    np.testing.assert_allclose(rebuild(result), a, atol=1e-12)
    np.testing.assert_allclose(
        result.factors["Q"].T @ result.factors["Q"], np.eye(a.shape[0]), atol=1e-12
    )
    if a.shape[0] < a.shape[1]:
        with pytest.raises(MethodError, match="m ≥ n"):
            solve(a, np.ones(a.shape[0]), result)


@pytest.mark.parametrize(
    "method", ["qr_classical", "qr_modified", "qr_householder", "qr_givens", "svd"]
)
def test_least_squares(method):
    a = np.column_stack([np.ones(4), np.arange(4.0)])
    b = np.array([1.0, 2.0, 2.0, 4.0])
    result = decompose(a, method)
    actual = solve(a, b, result)
    np.testing.assert_allclose(
        actual.x, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-10
    )
    assert actual.residual_norm > 0


def test_lu_pivoting_and_variable_permutation():
    a = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    for method in ["lu_doolittle", "lu_crout"]:
        with pytest.raises(MethodError, match="pivotamento"):
            decompose(a, method)
    x = np.array([2.0, 3.0, 4.0])
    for method in ["lu_partial", "lu_total"]:
        result = decompose(a, method)
        np.testing.assert_allclose(rebuild(result), a, atol=1e-12)
        np.testing.assert_allclose(solve(a, a @ x, result).x, x, atol=1e-12)
    assert not np.array_equal(decompose(a, "lu_total").factors["C"], np.eye(3))


@pytest.mark.parametrize(
    "method", ["lu_doolittle", "lu_crout", "lu_partial", "lu_total"]
)
def test_random_lu(method):
    rng = np.random.default_rng(17)
    for n in [2, 4, 8, 10]:
        for _ in range(3):
            a = rng.normal(size=(n, n))
            result = decompose(a, method)
            np.testing.assert_allclose(rebuild(result), a, rtol=1e-9, atol=1e-9)
            b = rng.normal(size=n)
            np.testing.assert_allclose(
                solve(a, b, result).x, np.linalg.solve(a, b), rtol=1e-8, atol=1e-8
            )


def test_singular_lu_is_factorable_but_not_uniquely_solvable():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    result = decompose(a, "lu_partial")
    np.testing.assert_allclose(rebuild(result), a)
    with pytest.raises(MethodError, match="nulo"):
        solve(a, np.array([1.0, 2.0]), result)


@pytest.mark.parametrize("method", ["cholesky", "ldlt"])
@pytest.mark.parametrize(
    "a",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.zeros((2, 2)),
        np.ones((2, 3)),
    ],
)
def test_cholesky_failure_reasons(method, a):
    with pytest.raises(MethodError):
        decompose(a, method)


def test_diagnostics_distinguish_methods():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    report = inspect_matrix(a)
    entries = {item["id"]: item for item in report["methods"]}
    assert report["properties"]["rank"] == 1
    assert not report["properties"]["positive_definite"]
    assert entries["svd"]["available"]
    assert entries["qr_householder"]["available"]
    assert entries["lu_partial"]["available"]
    assert not entries["qr_modified"]["available"]
    assert all(item["reason"] for item in report["methods"])


@pytest.mark.parametrize("text", ["1 2\n3 4", "1,2;3,4", "[[1,2],[3,4]]", "1/1 2\n3 4"])
def test_parser(text):
    np.testing.assert_array_equal(parse_matrix(text), [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(parse_vector("1 2", 2), [1.0, 2.0])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 2\n3",
        "nan 1",
        "inf",
        "1/0",
        "__import__('os')",
        "1e101",
        "1e-101",
        "[1,2,3]",
        "1+2j",
    ],
)
def test_invalid_input(text):
    with pytest.raises(ValueError):
        parse_matrix(text)


@pytest.mark.parametrize("scale", [1e-80, 1.0, 1e80])
@pytest.mark.parametrize(
    "method", ["lu_partial", "cholesky", "ldlt", "qr_householder", "svd"]
)
def test_scale_invariance(scale, method):
    a = SPD * scale
    result = decompose(a, method)
    assert result.relative_error < 1e-10
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(solve(a, a @ expected, result).x, expected, atol=1e-8)


def test_trace_states_do_not_alias():
    result = decompose(SPD, "lu_partial")
    initial = result.steps[0].matrices["U"].copy()
    result.factors["U"][:] = 0
    np.testing.assert_array_equal(result.steps[0].matrices["U"], initial)


def test_lu_elimination_records_before_after_objective_and_operation():
    a = np.array([[2.0, 1.0, 1.0], [4.0, 5.0, 3.0], [2.0, 7.0, 7.0]])
    result = decompose(a, "lu_doolittle")
    elimination = next(step for step in result.steps if "E_inversa" in step.matrices)
    assert elimination.objective
    assert elimination.operation
    assert "U_antes" in elimination.matrices
    assert "U_depois" in elimination.matrices
    assert "L_antes" in elimination.matrices
    assert "L_depois" in elimination.matrices
    np.testing.assert_allclose(
        elimination.matrices["E"] @ elimination.matrices["U_antes"],
        elimination.matrices["U_depois"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        elimination.matrices["U"], elimination.matrices["U_depois"], atol=1e-12
    )
    before_l = elimination.matrices["L_antes"]
    after_l = elimination.matrices["L_depois"]
    changed = np.argwhere(~np.isclose(before_l, after_l))
    assert changed.shape[0] == 1


def test_pseudoinverse_moore_penrose_and_inconsistent_system():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    result = decompose(a, "svd")
    f = result.factors
    values = np.diag(f["Sigma"])
    inverse = np.zeros_like(values)
    kept = values > values[0] * 1e-12
    inverse[kept] = 1 / values[kept]
    pinv = f["Vt"].T @ np.diag(inverse) @ f["U"].T
    np.testing.assert_allclose(a @ pinv @ a, a, atol=1e-12)
    np.testing.assert_allclose(pinv @ a @ pinv, pinv, atol=1e-12)
    np.testing.assert_allclose((a @ pinv).T, a @ pinv, atol=1e-12)
    np.testing.assert_allclose((pinv @ a).T, pinv @ a, atol=1e-12)
    solution = solve(a, np.array([1.0, 3.0]), result)
    assert solution.residual_norm > 0.1


def test_ill_conditioned_svd_and_tolerance_cutoff():
    a = np.diag([1.0, 1e-10, 1e-14])
    result = decompose(a, "svd")
    b = np.ones(3)
    x = solve(a, b, result).x
    np.testing.assert_allclose(x, [1.0, 1e10, 0.0], rtol=1e-10)
    assert inspect_matrix(a)["properties"]["rank"] == 2


def test_extreme_solution_scale_has_finite_residual():
    a = np.array([[1e-100]])
    b = np.array([1e100])
    result = decompose(a, "svd")
    solution = solve(a, b, result)
    np.testing.assert_allclose(solution.x / 1e200, [1.0])
    assert np.isfinite(solution.relative_residual)


@pytest.mark.parametrize("method", ["qr_classical", "qr_modified"])
def test_gram_schmidt_matches_aula10_worked_example(method):
    a = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    result = decompose(a, method)
    expected_q = np.column_stack(
        [
            np.array([0.0, 1.0, 1.0]) / np.sqrt(2),
            np.array([2.0, -1.0, 1.0]) / np.sqrt(6),
            np.array([1.0, 1.0, -1.0]) / np.sqrt(3),
        ]
    )
    expected_r = np.array(
        [
            [np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0.0, np.sqrt(6) / 2, 1 / np.sqrt(6)],
            [0.0, 0.0, 2 / np.sqrt(3)],
        ]
    )
    np.testing.assert_allclose(result.factors["Q"], expected_q, atol=1e-12)
    np.testing.assert_allclose(result.factors["R"], expected_r, atol=1e-12)
    projection_steps = [s for s in result.steps if "projecao" in s.matrices]
    assert projection_steps
    for step in projection_steps:
        np.testing.assert_allclose(
            step.matrices["antes"] - step.matrices["projecao"],
            step.matrices["depois"],
            atol=1e-12,
        )


@pytest.mark.parametrize("shape", [(4, 2), (2, 4), (3, 3)])
def test_svd_explanatory_states_obey_aula18_identities(shape):
    a = np.random.default_rng(14).normal(size=shape)
    result = decompose(a, "svd")
    relation = next(s for s in result.steps if "YtY" in s.matrices)
    np.testing.assert_allclose(
        relation.matrices["Y"], relation.matrices["U_Sigma"], atol=1e-12
    )
    np.testing.assert_allclose(
        relation.matrices["YtY"], relation.matrices["Sigma_quadrado"], atol=1e-11
    )
    terms = [s for s in result.steps if "termo" in s.matrices]
    np.testing.assert_allclose(sum(s.matrices["termo"] for s in terms), a, atol=1e-12)
    np.testing.assert_allclose(terms[-1].matrices["soma_parcial"], a, atol=1e-12)


@pytest.mark.parametrize("method", ["lu_doolittle", "lu_crout"])
def test_lu_handout_factors_elementary_matrices_and_solution(method):
    a = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 7.0], [-2.0, 2.0, 5.0]])
    result = decompose(a, method)
    expected_l = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-2.0, 3.0, 1.0]])
    expected_u = np.array([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, -1.0]])
    if method == "lu_crout":
        expected_l = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [-2.0, 6.0, -1.0]])
        expected_u = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(result.factors["L"], expected_l)
    np.testing.assert_allclose(result.factors["U"], expected_u)
    elementary_steps = [s for s in result.steps if "E_inversa" in s.matrices]
    assert len(elementary_steps) == 3
    for step in elementary_steps:
        m = step.matrices
        np.testing.assert_allclose(m["E"] @ m["U_antes"], m["U"], atol=1e-12)
        np.testing.assert_allclose(m["E"] @ m["E_inversa"], np.eye(3), atol=1e-12)
    product = next(s for s in result.steps if "produto_das_inversas" in s.matrices)
    np.testing.assert_allclose(
        product.matrices["produto_das_inversas"], product.matrices["L"]
    )
    solution = solve(a, np.array([1.0, -1.0, -7.0]), result)
    np.testing.assert_allclose(solution.x, [2.0, 1.0, -1.0])


@pytest.mark.parametrize("method", ["lu_partial", "lu_total"])
def test_lu_elementary_trace_with_permutations(method):
    a = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    result = decompose(a, method)
    for step in result.steps:
        if "E_inversa" in step.matrices:
            m = step.matrices
            np.testing.assert_allclose(m["E"] @ m["U_antes"], m["U"], atol=1e-12)
    assert any("troca" in s.matrices for s in result.steps)
    np.testing.assert_allclose(rebuild(result), a)


@pytest.mark.parametrize(
    "a,b,method,x",
    [
        ([[2.0, 1.0], [4.0, 3.0]], [5.0, 11.0], "lu_doolittle", [2.0, 1.0]),
        ([[4.0, 2.0], [2.0, 3.0]], [8.0, 8.0], "cholesky", [1.0, 2.0]),
        ([[1.0, 1.0], [1.0, 0.0]], [3.0, 1.0], "qr_modified", [1.0, 2.0]),
        ([[3.0, 0.0], [0.0, 1.0]], [6.0, 2.0], "svd", [2.0, 2.0]),
    ],
)
def test_theory_simple_worked_examples(a, b, method, x):
    a = np.asarray(a)
    result = decompose(a, method)
    np.testing.assert_allclose(solve(a, np.asarray(b), result).x, x, atol=1e-12)


@pytest.mark.parametrize(
    "method,identity,keys,lower,upper",
    [
        ("lu_doolittle", "A=LU", {"L", "U"}, "Ly=b", "Ux=y"),
        ("lu_crout", "A=LU", {"L", "U"}, "Ly=b", "Ux=y"),
        ("lu_partial", "PA=LU", {"L", "U", "P"}, "Ly=Pb", "Ux=y"),
        ("lu_total", "PAC=LU", {"L", "U", "P", "C"}, "Ly=Pb", "Uz=y"),
    ],
)
def test_lu_identity_and_solution_trace_match_the_selected_variant(
    method, identity, keys, lower, upper
):
    # This input produces permutations for both pivoting variants.
    a = np.array([[2.0, 1.0, 1.0], [4.0, 5.0, 3.0], [2.0, 7.0, 7.0]])
    result = decompose(a, method)
    assert result.identity == identity
    assert set(result.factors) == keys
    assert result.steps[0].formula.startswith(identity + ",")
    solution = solve(a, [7.0, 23.0, 37.0], result)
    np.testing.assert_allclose(solution.x, [1.0, 2.0, 3.0], atol=1e-12)
    assert lower in solution.steps[0].formula
    assert upper in solution.steps[0].formula
    for step in result.steps + solution.steps:
        if "P" not in keys:
            assert "P" not in step.formula
            assert "Pb" not in step.explanation
            assert "matriz_permutada" not in step.matrices
        if "C" not in keys:
            assert "PAC" not in step.formula
            assert "P=C=I" not in step.formula
            assert "x=Cz" not in step.formula
            assert "x=Cz" not in step.explanation
    assert any(s.title == "Permutar b" for s in solution.steps) == ("P" in keys)
    assert any(s.title == "Recuperar ordem das variáveis" for s in solution.steps) == (
        "C" in keys
    )
    if method == "lu_doolittle":
        np.testing.assert_array_equal(
            result.factors["L"], [[1, 0, 0], [2, 1, 0], [1, 2, 1]]
        )
        np.testing.assert_array_equal(
            result.factors["U"], [[2, 1, 1], [0, 3, 1], [0, 0, 4]]
        )


@pytest.mark.parametrize(
    "a",
    [
        [[2.0, 1.0], [1.0, 2.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        [[1.0, 1.0], [2.0, 2.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
)
def test_spectral_svd_uses_and_verifies_gram_eigenpairs(a, monkeypatch):
    a = np.array(a)

    def forbidden(*args, **kwargs):
        raise AssertionError("A construção espectral não deve chamar SVD pronta.")

    monkeypatch.setattr(np.linalg, "svd", forbidden)
    result = decompose(a, "svd")
    s = float(np.max(np.abs(a))) or 1.0
    gram = (a / s).T @ (a / s)
    eigen_step = next(
        step for step in result.steps if "autovalores_escalados" in step.matrices
    )
    e = eigen_step.matrices["autovalores_escalados"].ravel()
    v = eigen_step.matrices["autovetores"]
    np.testing.assert_allclose(gram @ v, v * e, atol=1e-12)
    np.testing.assert_allclose(v.T @ v, np.eye(a.shape[1]), atol=1e-12)
    f = result.factors
    np.testing.assert_allclose(a @ f["Vt"].T, f["U"] @ f["Sigma"], atol=1e-12)
    assert any("numpy.linalg.eigh" in step.explanation for step in result.steps)
    assert not any("Jacobi" in step.title for step in result.steps)


def test_spectral_polynomial_and_worked_non_diagonal_example():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    result = decompose(a, "svd")
    polynomial = next(
        step
        for step in result.steps
        if step.title == "Equação que determina os autovalores"
    )
    np.testing.assert_allclose(
        polynomial.matrices["coeficientes"], [[1.0, -2.5, 0.5625]]
    )
    np.testing.assert_allclose(np.diag(result.factors["Sigma"]), [3.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(solve(a, [4.0, 5.0], result).x, [1.0, 2.0], atol=1e-12)


def test_spectral_svd_explains_precision_limit_for_rotated_ill_conditioning():
    q = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    a = np.diag([1.0, 1e-10]) @ q.T
    result = decompose(a, "svd")
    assert any("direções pequenas" in warning for warning in result.warnings)
    assert result.relative_error < 1e-8


@pytest.mark.parametrize("method", ["qr_householder", "qr_givens"])
def test_qr_intermediate_products_preserve_original_matrix(method):
    a = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])
    result = decompose(a, method)
    stored = [
        step for step in result.steps if step.title.startswith("Guardar os fatores")
    ]
    assert stored
    for step in stored:
        np.testing.assert_allclose(
            step.matrices["Q"] @ step.matrices["R"], a, atol=1e-12
        )
        q = step.matrices["Q"]
        np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-12)
