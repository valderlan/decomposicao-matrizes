import numpy as np

from .decompositions import METHODS, decompose
from .models import MethodError
from .validation import DEFAULT_RTOL, threshold, validate_matrix, validate_rtol


def inspect_matrix(value, rtol=DEFAULT_RTOL):
    a = validate_matrix(value)
    validate_rtol(rtol)
    m, n = a.shape
    scale = float(np.max(np.abs(a))) or 1.0
    singular = np.linalg.svd(a / scale, compute_uv=False)
    cutoff = rtol * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    full_rank = rank == min(m, n)
    condition = float(singular[0] / singular[-1]) if full_rank else float("inf")
    symmetric = m == n and np.max(np.abs(a - a.T)) <= threshold(a, rtol)
    eigenvalues = np.linalg.eigvalsh((a + a.T) / (2 * scale)) * scale if symmetric else None
    properties = {
        "rows": m,
        "columns": n,
        "square": m == n,
        "symmetric": bool(symmetric),
        "rank": rank,
        "full_rank": full_rank,
        "positive_definite": bool(symmetric and eigenvalues.min() > threshold(a, rtol)),
        "condition_2": condition,
        "norm_frobenius": float(np.linalg.norm(a)),
        "norm_inf": float(np.linalg.norm(a, ord=np.inf)),
        "rtol": rtol,
        "rank_cutoff": float(cutoff * scale),
        "singular_values": (singular * scale).tolist(),
    }
    availability = []
    for method, name in METHODS.items():
        try:
            result = decompose(a, method, rtol, record_steps=False)
            reason = (
                "Aplicável a matrizes reais, inclusive retangulares e singulares."
                if method == "svd"
                else "Reflexões/rotações permitem QR mesmo com colunas dependentes."
                if method in {"qr_householder", "qr_givens"}
                else "As colunas podem ser ortonormalizadas sob a tolerância escolhida."
                if method.startswith("qr_")
                else "Simetria e pivôs positivos foram verificados."
                if method in {"cholesky", "ldlt"}
                else "A eliminação e as condições dos pivôs foram verificadas."
            )
            if m == n and rank < n and method.startswith("lu_"):
                reason += (
                    " Fatoração possível, mas há posto deficiente e não há solução única garantida."
                )
            if result.warnings:
                reason += " " + " ".join(result.warnings)
            availability.append({"id": method, "name": name, "available": True, "reason": reason})
        except MethodError as exc:
            availability.append(
                {"id": method, "name": name, "available": False, "reason": str(exc)}
            )
    warnings = []
    if not full_rank:
        warnings.append(
            "Posto numérico deficiente. Isso não impede QR por Householder/Givens ou SVD. "
            "A existência de solução de Ax=b depende também de b; a pseudoinversa fornece "
            "uma solução de mínimos quadrados com norma mínima."
        )
    if not np.isfinite(condition) or condition > 1e8:
        warnings.append(
            "Matriz singular numericamente ou mal condicionada. Pequenas perturbações nos "
            "dados podem produzir grandes mudanças na solução. Examine os avisos de cada método, "
            "a ortogonalidade e o resíduo. A SVD pela gramiana também pode perder precisão."
        )
    warnings.append(
        "Aplicabilidade, posto e positividade são diagnósticos numéricos dependentes da "
        "tolerância; uma falha de algoritmo não prova inexistência de fatoração matemática."
    )
    return {"properties": properties, "methods": availability, "warnings": warnings}
