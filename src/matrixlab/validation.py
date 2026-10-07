import json
import re
from fractions import Fraction

import numpy as np

MAX_DIMENSION = 10
DEFAULT_RTOL = 1e-12


def validate_matrix(value):
    if np.iscomplexobj(value):
        raise ValueError("Esta versão aceita apenas matrizes reais.")
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Informe uma matriz retangular de números reais.") from exc
    if a.ndim != 2 or min(a.shape) < 1:
        raise ValueError("A matriz precisa ter pelo menos uma linha e uma coluna.")
    if max(a.shape) > MAX_DIMENSION:
        raise ValueError(f"Limite didático: {MAX_DIMENSION} linhas e {MAX_DIMENSION} colunas.")
    if not np.isfinite(a).all():
        raise ValueError("NaN e infinito não são números válidos para esta ferramenta.")
    nonzero = np.abs(a[a != 0])
    if nonzero.size and (nonzero.max() > 1e100 or nonzero.min() < 1e-100):
        raise ValueError("Use valores entre 1e-100 e 1e100 em módulo, ou zero. Reescale a matriz.")
    return a.copy()


def validate_rtol(rtol):
    if not np.isfinite(rtol) or not 1e-15 <= rtol <= 1e-4:
        raise ValueError("A tolerância relativa deve estar entre 1e-15 e 1e-4.")


def parse_matrix(text):
    """Linhas por newline/; e valores por espaço/vírgula; JSON também aceito."""
    text = text.strip()
    if not text:
        raise ValueError("Digite uma matriz antes de analisar.")
    if len(text) > 20000:
        raise ValueError("Entrada muito longa. O limite é uma matriz 10 × 10.")
    try:
        if text.startswith("["):
            return validate_matrix(json.loads(text))
        rows = []
        for line in re.split(r"[;\n]+", text):
            if line.strip():
                tokens = re.split(r"[\s,]+", line.strip())
                rows.append([float(Fraction(x)) if "/" in x else float(x) for x in tokens])
        return validate_matrix(rows)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(
            "Entrada inválida: use linhas de mesmo tamanho e valores como 2, -1, 0.5 ou 1/3. "
            "Use ponto como separador decimal; vírgula separa elementos. " + str(exc)
        ) from exc


def parse_vector(text, length):
    a = parse_matrix(text)
    if 1 not in a.shape or a.size != length:
        raise ValueError(
            f"O vetor b deve conter exatamente {length} números em uma linha ou coluna."
        )
    return a.ravel()


def threshold(a, rtol):
    return rtol * max(float(np.linalg.norm(a, ord=np.inf)), np.finfo(float).tiny)


def relative_error(a, b):
    scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
    denom = np.linalg.norm(a / scale)
    return float(np.linalg.norm((a - b) / scale) / max(denom, np.finfo(float).tiny))


def stable_norm(value):
    """Norma euclidiana/Frobenius sem elevar a magnitude original ao quadrado."""
    value = np.asarray(value)
    scale = float(np.max(np.abs(value)))
    if scale == 0:
        return 0.0
    return float(scale * np.linalg.norm(value / scale))
