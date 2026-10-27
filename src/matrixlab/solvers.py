import numpy as np

from .arithmetic import norm_calculation, number, products, record_product
from .models import MethodError, Solution, Trace
from .validation import DEFAULT_RTOL, stable_norm, threshold, validate_matrix, validate_rtol


def _triangular(t, b, lower, trace, label, rtol, unknown="x", rhs_symbol="b"):
    n = len(b)
    x = np.zeros(n)
    tol = threshold(t, rtol)
    indices = range(n) if lower else range(n - 1, -1, -1)
    for i in indices:
        pivot = t[i, i]
        if abs(pivot) <= tol:
            raise MethodError(
                f"{label}: pivô triangular {i + 1} é numericamente nulo. "
                "Não há solução única garantida por este procedimento. Use SVD."
            )
        part = t[i, :i] @ x[:i] if lower else t[i, i + 1 :] @ x[i + 1 :]
        known_indices = range(i) if lower else range(i + 1, n)
        known_terms = (
            " + ".join(f"({t[i, j]:.8g} × {x[j]:.8g})" for j in known_indices)
            or "0 (nenhuma componente conhecida é necessária)"
        )
        x[i] = (b[i] - part) / pivot
        trace.add(
            f"{label}: componente {i + 1}",
            f"A soma das parcelas já conhecidas é {known_terms} = {part:.10g}. "
            f"A equação desta linha é ({pivot:.10g}) × {unknown}{i + 1} + "
            f"({part:.10g}) = {b[i]:.10g}. Isolando a incógnita: "
            f"{unknown}{i + 1} = ({b[i]:.10g} − {part:.10g}) / ({pivot:.10g}) = {x[i]:.10g}. "
            + (
                "Usamos as componentes anteriores."
                if lower
                else "Usamos as componentes posteriores."
            ),
            rf"{unknown}_i=({rhs_symbol}_i-\sum_{{j<i}}t_{{ij}}{unknown}_j)/t_{{ii}}"
            if lower
            else rf"{unknown}_i=({rhs_symbol}_i-\sum_{{j>i}}t_{{ij}}{unknown}_j)/t_{{ii}}",
            calculations=[
                rf"S={products(t[i, :i], x[:i]) if lower else products(t[i, i + 1 :], x[i + 1 :])}\approx {number(part)}",
                rf"{unknown}_{{{i + 1}}}=\frac{{({number(b[i])})-({number(part)})}}{{{number(pivot)}}}\approx {number(x[i])}",
            ],
            parcial=x[:, None],
        )
    return x


def solve(value, b, decomposition, rtol=DEFAULT_RTOL):
    a = validate_matrix(value)
    validate_rtol(rtol)
    if np.iscomplexobj(b):
        raise ValueError("b precisa ser real.")
    b = np.asarray(b, dtype=float)
    if b.ndim != 1 or len(b) != a.shape[0] or not np.isfinite(b).all():
        raise ValueError("b precisa ser um vetor finito com uma entrada para cada linha de A.")
    nonzero = np.abs(b[b != 0])
    if nonzero.size and (nonzero.max() > 1e100 or nonzero.min() < 1e-100):
        raise ValueError(
            "Reescale b: valores não nulos devem estar entre 1e-100 e 1e100 em módulo."
        )
    factors, method = decomposition.factors, decomposition.method
    trace = Trace()
    m, n = a.shape
    kind = "Solução única por substituições triangulares"
    if method.startswith("lu_"):
        total_pivoting = method == "lu_total"
        row_pivoting = method in {"lu_partial", "lu_total"}
        pb = factors["P"] @ b if row_pivoting else b
        lower_system = "Ly=Pb" if row_pivoting else "Ly=b"
        upper_system = "Uz=y" if total_pivoting else "Ux=y"
        explanation = (
            "Como PAC=LU e x=Cz, o sistema original se torna LUz=Pb. "
            "Primeiro resolvemos Ly=Pb, depois Uz=y e recuperamos x=Cz. "
            if total_pivoting
            else "Como PA=LU, aplicamos as mesmas trocas de linhas a b. "
            "Primeiro resolvemos Ly=Pb e depois Ux=y. A ordem das variáveis não muda. "
            if row_pivoting
            else "Como A=LU, temos LUx=b. Definimos y=Ux: primeiro resolvemos Ly=b "
            "de cima para baixo e depois Ux=y de baixo para cima. "
        )
        formula = rf"Ax=b\quad\Longrightarrow\quad {lower_system},\quad {upper_system}"
        if total_pivoting:
            formula += r",\quad x=Cz"
        trace.add(
            "Organizar os dois sistemas triangulares",
            explanation
            + "As inversas explicam a fatoração, mas não são calculadas para resolver o sistema.",
            formula,
            L=factors["L"],
            U=factors["U"],
            b=b[:, None],
        )
        if row_pivoting:
            trace.add(
                "Permutar b",
                "As trocas de linhas de A também são aplicadas a b.",
                r"\tilde b=Pb",
                calculations=[
                    rf"\tilde b_{{{i + 1}}}={products(factors['P'][i, :], b)}={number(pb[i])}"
                    for i in range(m)
                ],
                Pb=pb[:, None],
            )
        y = _triangular(
            factors["L"],
            pb,
            True,
            trace,
            f"Substituição sucessiva {lower_system}",
            rtol,
            unknown="y",
            rhs_symbol=r"\tilde b" if row_pivoting else "b",
        )
        z = _triangular(
            factors["U"],
            y,
            False,
            trace,
            f"Substituição regressiva {upper_system}",
            rtol,
            unknown="z" if total_pivoting else "x",
            rhs_symbol="y",
        )
        x = factors["C"] @ z if total_pivoting else z
        if total_pivoting:
            trace.add(
                "Recuperar ordem das variáveis",
                "Desfazemos a permutação de colunas.",
                "x=Cz",
                calculations=[
                    rf"x_{{{i + 1}}}={products(factors['C'][i, :], z)}={number(x[i])}"
                    for i in range(n)
                ],
                x=x[:, None],
            )
    elif method == "cholesky":
        y = _triangular(
            factors["L"], b, True, trace, "Substituição sucessiva Ly=b", rtol, unknown="y"
        )
        x = _triangular(
            factors["L"].T, y, False, trace, "Substituição regressiva Lᵀx=y", rtol, rhs_symbol="y"
        )
    elif method == "ldlt":
        y = _triangular(
            factors["L"], b, True, trace, "Substituição sucessiva Ly=b", rtol, unknown="y"
        )
        d = np.diag(factors["D"])
        if np.any(np.abs(d) <= threshold(factors["D"], rtol)):
            raise MethodError("D tem pivô numericamente nulo. Use SVD.")
        z = y / d
        trace.add(
            "Resolver Dz=y",
            "D é diagonal: dividimos cada componente pelo pivô correspondente.",
            "z_i=y_i/d_i",
            calculations=[
                rf"z_{{{i + 1}}}=\frac{{{number(y[i])}}}{{{number(d[i])}}}={number(z[i])}"
                for i in range(n)
            ],
            z=z[:, None],
        )
        x = _triangular(
            factors["L"].T, z, False, trace, "Substituição regressiva Lᵀx=z", rtol, rhs_symbol="z"
        )
    elif method.startswith("qr_"):
        if m < n:
            raise MethodError(
                "Esta rotina QR de resolução usa um bloco R quadrado com posto completo de colunas "
                "e exige m ≥ n. A fatoração QR continua válida; use SVD para norma mínima em sistemas largos."
            )
        qt_b = factors["Q"].T @ b
        q1 = factors["Q"][:, :n]
        projection_b = q1 @ (q1.T @ b)
        projection_coordinates = q1.T @ b
        trace.add(
            "Calcular a projeção de b na base Q₁",
            "Primeiro calculamos c=Q₁ᵀb, depois Q₁c. Subtrair esse vetor de b fornece a componente perpendicular ao espaço das colunas.",
            r"c=Q_1^Tb,\quad \widehat b=Q_1c,\quad r_{perp}=b-\widehat b",
            calculations=[
                rf"c_{{{i + 1}}}={products(q1[:, i], b)}\approx {number(projection_coordinates[i])}"
                for i in range(n)
            ]
            + [
                rf"\widehat b_{{{i + 1}}}={products(q1[i, :], projection_coordinates)}\approx {number(projection_b[i])},\quad (r_{{perp}})_{{{i + 1}}}=({number(b[i])})-({number(projection_b[i])})\approx {number(b[i] - projection_b[i])}"
                for i in range(m)
            ],
            projecao=projection_b[:, None],
        )
        trace.add(
            "Por que resolver um sistema triangular?",
            "Na Aula 10, mínimos quadrados é interpretado como projetar b no espaço "
            "das colunas de A. Usamos Q₁ com n colunas ortonormais e R₁ triangular "
            "de ordem n. A parte b−Q₁Q₁ᵀb é perpendicular a esse espaço e não pode "
            "ser removida pela escolha de x. Basta anular a parte R₁x−Q₁ᵀb.",
            r"\|Ax-b\|_2^2=\|R_1x-Q_1^Tb\|_2^2+\|b-Q_1Q_1^Tb\|_2^2",
            Q1=q1,
            R1=factors["R"][:n, :n],
            b=b[:, None],
            projecao_b=projection_b[:, None],
            complemento=(b - projection_b)[:, None],
        )
        trace.add(
            "Transformar o segundo membro",
            "Cada componente de Qᵀb é um produto interno com uma coluna de Q. "
            "As primeiras n componentes são usadas na substituição regressiva. "
            "Com Q completa e m>n, as componentes restantes descrevem o residual "
            "ortogonal; com Q reduzida, esse residual é b−Q₁Q₁ᵀb.",
            r"c=Q^Tb,\qquad R_1x=c_{1:n}",
            calculations=[
                rf"c_{{{i + 1}}}={products(factors['Q'][:, i], b)}\approx {number(qt_b[i])}"
                for i in range(len(qt_b))
            ],
            c=qt_b[:, None],
        )
        x = _triangular(
            factors["R"][:n, :n], qt_b[:n], False, trace, "Resolver Rx=c", rtol, rhs_symbol="c"
        )
        if m > n:
            kind = "Mínimos quadrados por QR (solução única com posto completo de colunas)"
    elif method == "svd":
        u, sigma, vt = factors["U"], factors["Sigma"], factors["Vt"]
        values = np.diag(sigma)
        cutoff = rtol * (values[0] if values.size else 0)
        inverse = np.zeros_like(values)
        retained = values > cutoff
        inverse[retained] = 1 / values[retained]
        trace.add(
            "Construir Σ⁺",
            f"Invertemos apenas σ > {cutoff:.6g}; os demais recebem zero. "
            "Com tolerância positiva, tratamos valores pequenos como nulos: é uma pseudoinversa numérica truncada.",
            r"\sigma_i^+=\begin{cases}1/\sigma_i&\sigma_i>\tau\\0&\text{caso contrário}\end{cases}",
            calculations=[rf"\tau=({number(values[0])})({number(rtol)})={number(cutoff)}"]
            + [
                rf"\sigma_{{{i + 1}}}^+="
                + (
                    rf"\frac{{1}}{{{number(value)}}}\approx {number(inverse[i])}"
                    if value > cutoff
                    else rf"0\quad({number(value)}\leq {number(cutoff)})"
                )
                for i, value in enumerate(values)
            ],
            Sigma_plus=np.diag(inverse),
        )
        intermediate = record_product(trace, vt.T, np.diag(inverse), "T", r"V\Sigma^+")
        pinv = record_product(trace, intermediate, u.T, "A^+", "TU^T")
        trace.add(
            "Pseudoinversa de A",
            "Montamos a pseudoinversa a partir dos fatores reduzidos.",
            r"A^+=V\Sigma^+U^T",
            A_plus=pinv,
        )
        projection = u.T @ b
        trace.add(
            "Projetar b nos vetores singulares",
            "Cada componente cᵢ=uᵢᵀb é a coordenada de b em uma direção singular "
            "esquerda. Direções que a matriz não consegue atingir permanecerão no "
            "residual. Para σᵢ não nulo, Avᵢ=σᵢuᵢ permite recuperar a coordenada de x "
            "dividindo cᵢ por σᵢ.",
            "c=U^Tb",
            calculations=[
                rf"c_{{{i + 1}}}={products(u[:, i], b)}\approx {number(projection[i])}"
                for i in range(len(projection))
            ],
            c=projection[:, None],
        )
        coordinates = inverse * projection
        trace.add(
            "Escalar as coordenadas e escolher norma mínima",
            "Calculamos zᵢ=cᵢ/σᵢ para os valores acima do corte e zᵢ=0 para os "
            "descartados. Acrescentar componentes em direções do núcleo não melhora "
            "Ax e aumenta a norma de x; escolher zero nessas direções fornece a "
            "solução de norma mínima do problema numérico considerado.",
            r"z=\Sigma^+c,\qquad x=Vz",
            calculations=[
                rf"z_{{{i + 1}}}=({number(inverse[i])})({number(projection[i])})\approx {number(coordinates[i])}"
                for i in range(len(coordinates))
            ],
            c=projection[:, None],
            z=coordinates[:, None],
        )
        x = vt.T @ coordinates
        kind = "Mínimos quadrados de norma mínima pela pseudoinversa numérica"
        trace.add(
            "Construir x",
            "Escalamos as coordenadas pelos inversos dos valores singulares e retornamos pela base V.",
            r"x=V\Sigma^+U^Tb",
            calculations=[
                rf"x_{{{i + 1}}}={products(vt[:, i], coordinates)}\approx {number(x[i])}"
                for i in range(n)
            ],
            x=x[:, None],
        )
    else:
        raise ValueError("Método desconhecido.")
    if not np.isfinite(x).all():
        raise MethodError("A solução excedeu a precisão numérica. Reescale A e b.")
    residual = a @ x - b
    residual_norm = stable_norm(residual)
    denominator = float(stable_norm(a) * stable_norm(x) + stable_norm(b))
    relative_residual = residual_norm / max(denominator, np.finfo(float).tiny)
    trace.add(
        "Verificar a solução",
        f"Norma de Ax−b: {residual_norm:.10g}. Resíduo relativo escalado: {relative_residual:.10g}. "
        "Um resíduo não nulo pode ser esperado em mínimos quadrados. Resíduo pequeno não garante "
        "erro pequeno em x se A for mal condicionada.",
        r"\eta=\frac{\|Ax-b\|_2}{\|A\|_F\|x\|_2+\|b\|_2}",
        calculations=[
            rf"(Ax)_{{{i + 1}}}={products(a[i, :], x)}\approx {number((a @ x)[i])},\quad r_{{{i + 1}}}=({number((a @ x)[i])})-({number(b[i])})\approx {number(residual[i])}"
            for i in range(m)
        ]
        + [
            norm_calculation(residual, r"\|r\|_2", residual_norm),
            rf"\eta=\frac{{{number(residual_norm)}}}{{{number(max(denominator, np.finfo(float).tiny))}}}\approx {number(relative_residual)}",
        ],
        Ax=(a @ x)[:, None],
        residuo=residual[:, None],
    )
    return Solution(x, residual_norm, relative_residual, kind, trace.steps)
