import numpy as np


def number(value):
    text = f"{float(value):.10g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return text


def products(*vectors):
    return (
        "+".join(r"\cdot".join(f"({number(value)})" for value in term) for term in zip(*vectors))
        or "0"
    )


def norm_calculation(vector, label, value=None):
    value = np.linalg.norm(vector) if value is None else value
    squares = "+".join(f"({number(x)})^2" for x in vector) or "0"
    return rf"{label}=\sqrt{{{squares}}}\approx {number(value)}"


def divisions(vector, denominator, label):
    return [
        rf"{label}_{{{i + 1}}}=\frac{{{number(x)}}}{{{number(denominator)}}}\approx {number(x / denominator)}"
        for i, x in enumerate(vector)
    ]


def product_calculations(left, right, result, label):
    return [
        rf"\left({label}\right)_{{{i + 1},{j + 1}}}={products(left[i, :], right[:, j])}\approx {number(result[i, j])}"
        for i in range(result.shape[0])
        for j in range(result.shape[1])
    ]


def record_product(trace, left, right, label, expression, explanation=""):
    result = left @ right
    if trace.enabled:
        trace.add(
            f"Multiplicar {expression}",
            explanation + " Cada entrada é o produto interno de uma linha do fator à esquerda "
            "com uma coluna do fator à direita. Os números exibidos estão arredondados; "
            "o cálculo usa os valores completos.",
            rf"{label}={expression}",
            calculations=product_calculations(left, right, result, label),
            fator_esquerdo=left,
            fator_direito=right,
            produto=result,
        )
    return result
