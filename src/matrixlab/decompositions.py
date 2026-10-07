import numpy as np

from .arithmetic import divisions, norm_calculation, number, products, record_product
from .models import Decomposition, MethodError, Trace
from .validation import (
    DEFAULT_RTOL,
    relative_error,
    stable_norm,
    threshold,
    validate_matrix,
    validate_rtol,
)

METHODS = {
    "lu_doolittle": "LU • Doolittle sem pivotamento",
    "lu_crout": "LU • Crout sem pivotamento",
    "lu_partial": "LU • pivotamento parcial",
    "lu_total": "LU • pivotamento total",
    "cholesky": "Cholesky • LLᵀ",
    "ldlt": "Cholesky sem raízes • LDLᵀ (positiva definida)",
    "qr_classical": "QR • Gram–Schmidt clássico",
    "qr_modified": "QR • Gram–Schmidt modificado",
    "qr_householder": "QR • Householder",
    "qr_givens": "QR • Givens",
    "svd": "SVD • autovalores e autovetores",
}


def _square(a):
    if a.shape[0] != a.shape[1]:
        raise MethodError("Esta variante exige matriz quadrada; a entrada é retangular.")


def _lu(a, variant, trace, rtol):
    _square(a)
    n = a.shape[0]
    l, u, p, c = np.eye(n), a.copy(), np.eye(n), np.eye(n)
    tol = threshold(a, rtol)
    identity = "PAC=LU" if variant == "lu_total" else "PA=LU" if variant == "lu_partial" else "A=LU"
    permutation_intro = (
        "P registra trocas de linhas e C registra trocas de colunas. "
        if variant == "lu_total"
        else "P registra trocas de linhas. Não trocamos colunas nesta variante. "
        if variant == "lu_partial"
        else "Nesta variante não trocamos linhas nem colunas. "
    )
    initial_formula = identity + r",\qquad L=I,\quad U=A"
    if variant == "lu_total":
        initial_formula += r",\quad P=C=I"
    elif variant == "lu_partial":
        initial_formula += r",\quad P=I"
    trace.add(
        "Inicialização",
        "U recebe A e será transformada por eliminação de Gauss. L começa como identidade "
        "e guardará os multiplicadores abaixo da diagonal. "
        + permutation_intro
        + (
            "Em Crout, esta eliminação produz fatores temporários de Doolittle; "
            "a normalização final transfere a diagonal de U para L."
            if variant == "lu_crout"
            else "A diagonal de L permanece igual a 1."
        ),
        initial_formula,
        L=l,
        U=u,
    )
    trace.add(
        "Ideia da Seção 1.3: operações e matrizes elementares",
        "Uma operação de linha equivale a multiplicar a matriz por uma matriz elementar E "
        "à esquerda. Para anular uma entrada, usamos linhaᵢ←linhaᵢ−mᵢₖlinhaₖ. "
        "E contém −mᵢₖ; sua inversa contém +mᵢₖ. É por isso que L guarda o multiplicador "
        "com o sinal original, não o coeficiente negativo usado na eliminação.",
        r"E=I-m_{ik}e_ie_k^T,\qquad E^{-1}=I+m_{ik}e_ie_k^T,\qquad U_{novo}=EU_{antes}",
    )
    inverse_product = np.eye(n)
    used_permutations = False
    for k in range(n):
        row, col = k, k
        if variant == "lu_partial":
            row = k + int(np.argmax(np.abs(u[k:, k])))
        elif variant == "lu_total":
            i, j = np.unravel_index(np.argmax(np.abs(u[k:, k:])), (n - k, n - k))
            row, col = k + i, k + j
        pivot_rule = (
            "No pivotamento parcial, escolhemos o elemento de maior módulo na coluna ativa. "
            "Isso reduz multiplicadores grandes nesta etapa, sem garantir bom condicionamento de A."
            if variant == "lu_partial"
            else "No pivotamento total, escolhemos o elemento de maior módulo no bloco ainda não eliminado. "
            "Podemos trocar linhas e colunas, reordenando também as variáveis."
            if variant == "lu_total"
            else "Sem pivotamento, usamos o elemento diagonal atual. Ele precisa ser significativo "
            "para dividir entradas abaixo dele; um zero pode impedir esta variante."
        )
        trace.add(
            f"Escolher o pivô da etapa {k + 1}",
            pivot_rule + f" O candidato está na posição ({row + 1},{col + 1}) e vale "
            f"{u[row, col]:.10g}. O limiar numérico desta matriz é {tol:.4g}.",
            rf"m_{{i,{k + 1}}}=u_{{i,{k + 1}}}/u_{{{k + 1},{k + 1}}}",
            bloco_ativo=u[k:, k:],
        )
        if row != k:
            before_swap = u.copy()
            swap = np.eye(n)
            swap[[k, row], :] = swap[[row, k], :]
            used_permutations = True
            u[[k, row], :] = u[[row, k], :]
            p[[k, row], :] = p[[row, k], :]
            l[[k, row], :k] = l[[row, k], :k]
            trace.add(
                f"Troca de linhas {k + 1} e {row + 1}",
                "A matriz de permutação elementar S é obtida trocando essas linhas na identidade. "
                "Multiplicar S por U aplica a troca. Acumulamos P←SP e trocamos, em L, "
                "apenas os multiplicadores de colunas já processadas. Essa atualização mantém "
                "o vínculo entre as linhas eliminadas e os multiplicadores que as produziram.",
                r"U\leftarrow SU,\qquad P\leftarrow SP",
                antes=before_swap,
                troca=swap,
                P=p,
                L=l,
                U=u,
            )
        if col != k:
            before_swap = u.copy()
            swap = np.eye(n)
            swap[:, [k, col]] = swap[:, [col, k]]
            used_permutations = True
            u[:, [k, col]] = u[:, [col, k]]
            c[:, [k, col]] = c[:, [col, k]]
            trace.add(
                f"Troca de colunas {k + 1} e {col + 1}",
                "Uma troca de colunas é uma multiplicação por T à direita. Acumulamos C←CT. "
                "As colunas representam variáveis: resolveremos na nova ordem e recuperaremos "
                "a ordem original por x=Cz. Esta T é uma permutação, não a transposta de A.",
                r"U\leftarrow UT,\qquad C\leftarrow CT,\qquad PAC=LU",
                antes=before_swap,
                troca=swap,
                C=c,
                U=u,
            )
        pivot = u[k, k]
        if abs(pivot) <= tol:
            if np.any(np.abs(u[k + 1 :, k]) > tol):
                raise MethodError(
                    f"Pivô U[{k + 1},{k + 1}]={pivot:.6g} é zero ou pequeno "
                    f"(limiar {tol:.3g}) e há valores abaixo dele. É necessário pivotamento."
                )
            u[k + 1 :, k] = 0.0
            trace.add(
                f"Pivô numericamente nulo na coluna {k + 1}",
                "Não há elemento significativo abaixo deste pivô. Valores abaixo dele dentro da "
                "tolerância são zerados, produzindo uma fatoração numérica aproximada. "
                "Podemos manter a fatoração, "
                "mas não garantir uma solução única por substituição triangular.",
                U=u,
            )
            continue
        for i in range(k + 1, n):
            before = u.copy()
            multiplier = u[i, k] / pivot
            trace.add(
                f"Calcular o multiplicador m[{i + 1},{k + 1}]",
                f"Queremos anular {u[i, k]:.10g} usando o pivô {pivot:.10g}. "
                f"O quociente é ({u[i, k]:.10g}) / ({pivot:.10g}) = {multiplier:.10g}. "
                "Esse multiplicador será guardado em L. Na matriz elementar de eliminação "
                "entra o seu oposto; na inversa, entra o próprio multiplicador.",
                rf"m_{{{i + 1},{k + 1}}}=\frac{{u_{{{i + 1},{k + 1}}}}}{{u_{{{k + 1},{k + 1}}}}},\qquad l_{{{i + 1},{k + 1}}}=m_{{{i + 1},{k + 1}}}",
                calculations=[
                    rf"m_{{{i + 1},{k + 1}}}=\frac{{{number(u[i, k])}}}{{{number(pivot)}}}={number(multiplier)}"
                ]
                if trace.enabled
                else [],
                linha_pivo=u[k : k + 1, :],
                linha_a_eliminar=u[i : i + 1, :],
            )
            l[i, k] = multiplier
            elementary, inverse = np.eye(n), np.eye(n)
            elementary[i, k] = -multiplier
            inverse[i, k] = multiplier
            if trace.enabled:
                inverse_product = inverse_product @ inverse
            u[i, k:] -= multiplier * u[k, k:]
            u[i, k] = 0.0
            calculations = ""
            if trace.enabled:
                calculations = " ".join(
                    f"Coluna {j + 1}: ({before[i, j]:.8g}) − ({multiplier:.8g}) × "
                    f"({before[k, j]:.8g}) = {u[i, j]:.8g}."
                    for j in range(k, n)
                )
            trace.add(
                f"Eliminar U[{i + 1},{k + 1}]",
                f"Subtraímos {multiplier:.10g} vezes a linha {k + 1} da linha {i + 1}. "
                + (
                    "Como o multiplicador é negativo, subtrair esse múltiplo equivale a somar "
                    "seu módulo vezes a linha pivô. "
                    if multiplier < 0
                    else ""
                )
                + calculations
                + " A matriz E abaixo realiza exatamente essa operação. "
                "E⁻¹ desfaz a operação; L registra os multiplicadores das operações inversas.",
                rf"L_{{{i + 1},{k + 1}}}=\frac{{U_{{{i + 1},{k + 1}}}}}{{U_{{{k + 1},{k + 1}}}}},\quad "
                rf"\mathrm{{linha}}_{{{i + 1}}}\leftarrow\mathrm{{linha}}_{{{i + 1}}}-L_{{{i + 1},{k + 1}}}\mathrm{{linha}}_{{{k + 1}}}",
                calculations=[
                    rf"u_{{{i + 1},{j + 1}}}^{{novo}}=({number(before[i, j])})-({number(multiplier)})({number(before[k, j])})\approx {number(u[i, j])}"
                    for j in range(k, n)
                ]
                if trace.enabled
                else [],
                L=l,
                U=u,
                U_antes=before,
                E=elementary,
                E_inversa=inverse,
            )
    if trace.enabled and not used_permutations:
        trace.add(
            "Por que as inversas formam L?",
            "Sem trocas, as eliminações produzem Eₛ…E₂E₁A=U. Para recuperar A, "
            "desfazemos as operações: A=E₁⁻¹E₂⁻¹…Eₛ⁻¹U. O produto dessas inversas "
            "é triangular inferior e coincide com a L de Doolittle construída pelos "
            "multiplicadores. A ordem do produto importa, como destacado na Seção 1.3.",
            r"U=E_s\cdots E_2E_1A,\qquad A=(E_1^{-1}E_2^{-1}\cdots E_s^{-1})U=LU",
            produto_das_inversas=inverse_product,
            L=l,
            U=u,
        )
    if variant == "lu_crout":
        diagonal = np.diag(u).copy()
        if np.any(np.abs(diagonal) <= tol):
            raise MethodError(
                "Crout nesta implementação requer pivôs não nulos para normalizar diag(U)=1. "
                "Tente LU com pivotamento ou SVD. Uma matriz singular pode admitir outras variantes LU."
            )

        before_l, before_u = l.copy(), u.copy()
        l = l @ np.diag(diagonal)
        u = u / diagonal[:, None]
        trace.add(
            "Normalização de Crout",
            "Após a eliminação, transferimos os pivôs de U para as colunas de L. "
            "Se D é a diagonal da U de Doolittle, multiplicamos cada coluna j de L por dⱼ "
            "e dividimos cada linha j de U por dⱼ. O produto se mantém, pois DD⁻¹=I. "
            "Assim Crout termina com diagonal de U igual a 1; Doolittle tem diagonal de L igual a 1.",
            r"L_C=L_DD,\quad U_C=D^{-1}U_D",
            calculations=[
                rf"(L_C)_{{{i + 1},{j + 1}}}=({number(before_l[i, j])})({number(diagonal[j])})={number(l[i, j])},\quad (U_C)_{{{i + 1},{j + 1}}}=\frac{{{number(before_u[i, j])}}}{{{number(diagonal[i])}}}={number(u[i, j])}"
                for i in range(n)
                for j in range(n)
            ]
            if trace.enabled
            else [],
            L=l,
            U=u,
            D=np.diag(diagonal),
        )
    factors = {"L": l, "U": u}
    if variant in {"lu_partial", "lu_total"}:
        factors["P"] = p
    if variant == "lu_total":
        factors["C"] = c
    trace.add(
        "Concluir os fatores e preparar a solução",
        "L é triangular inferior e U é triangular superior. "
        + (
            "Nesta variante com pivotamento total, temos PAC=LU: resolver Ly=Pb, Uz=y e x=Cz."
            if variant == "lu_total"
            else "Com pivotamento parcial, temos PA=LU: resolver Ly=Pb e Ux=y."
            if variant == "lu_partial"
            else "Sem pivotamento, temos A=LU: resolver Ly=b e Ux=y."
        )
        + " Não precisamos calcular a inversa de A. Se U tiver pivôs numericamente "
        "nulos, a fatoração pode existir sem que estas substituições determinem solução única.",
        identity,
        L=l,
        U=u,
        produto_LU=l @ u,
        **({"matriz_permutada": p @ a @ c} if "P" in factors else {"A": a}),
    )
    return factors, identity, p.T @ l @ u @ c.T, None


def _cholesky(a, variant, trace, rtol):
    _square(a)
    tol = threshold(a, rtol)
    asymmetry = np.max(np.abs(a - a.T))
    if asymmetry > tol:
        raise MethodError(f"A não é simétrica: max|A−Aᵀ|={asymmetry:.6g}, limiar={tol:.3g}.")
    w = (a + a.T) / 2
    trace.add(
        "Verificar simetria",
        "A deve ser simétrica e positiva definida. Se houver assimetria abaixo da tolerância, "
        "fatoramos (A+Aᵀ)/2 e verificamos o erro em relação à entrada original.",
        "A=A^T",
        A=w,
    )
    n = len(a)
    l = np.eye(n) if variant == "ldlt" else np.zeros_like(a)
    d = np.zeros(n)
    for j in range(n):
        correction = np.sum(l[j, :j] ** 2 * d[:j]) if variant == "ldlt" else l[j, :j] @ l[j, :j]
        diagonal_terms = (
            products(l[j, :j], l[j, :j], d[:j])
            if variant == "ldlt"
            else products(l[j, :j], l[j, :j])
        )
        pivot = w[j, j] - correction
        if pivot <= tol:
            raise MethodError(
                f"Na etapa {j + 1}, o pivô de Schur é {pivot:.10g} ≤ {tol:.3g}. "
                "Não é positiva definida numericamente sob a tolerância escolhida. "
                "Esta versão LDLᵀ também é restrita a matrizes positivas definidas."
            )
        if variant == "ldlt":
            d[j] = pivot
            trace.add(
                f"Calcular D[{j + 1},{j + 1}]",
                f"Subtraímos a contribuição anterior: {w[j, j]:.10g} − {correction:.10g} = {pivot:.10g}.",
                r"d_j=a_{jj}-\sum_{k<j} l_{jk}^2d_k",
                calculations=[
                    rf"d_{{{j + 1}}}=({number(w[j, j])})-({diagonal_terms})={number(pivot)}"
                ]
                if trace.enabled
                else [],
                L=l,
                D=np.diag(d),
            )
        else:
            l[j, j] = np.sqrt(pivot)
            trace.add(
                f"Calcular L[{j + 1},{j + 1}]",
                f"A raiz do pivô positivo {pivot:.10g} é {l[j, j]:.10g}.",
                r"l_{jj}=\sqrt{a_{jj}-\sum_{k<j}l_{jk}^2}",
                calculations=[
                    rf"l_{{{j + 1},{j + 1}}}=\sqrt{{({number(w[j, j])})-({diagonal_terms})}}=\sqrt{{{number(pivot)}}}\approx {number(l[j, j])}"
                ]
                if trace.enabled
                else [],
                L=l,
            )
        for i in range(j + 1, n):
            correction = (
                np.sum(l[i, :j] * l[j, :j] * d[:j]) if variant == "ldlt" else l[i, :j] @ l[j, :j]
            )
            denominator = d[j] if variant == "ldlt" else l[j, j]
            correction_terms = (
                products(l[i, :j], l[j, :j], d[:j])
                if variant == "ldlt"
                else products(l[i, :j], l[j, :j])
            )
            l[i, j] = (w[i, j] - correction) / denominator
            trace.add(
                f"Calcular L[{i + 1},{j + 1}]",
                f"({w[i, j]:.10g} − {correction:.10g}) / {denominator:.10g} = {l[i, j]:.10g}.",
                r"l_{ij}=\frac{a_{ij}-\sum_{k<j}l_{ik}l_{jk}d_k}{d_j}"
                if variant == "ldlt"
                else r"l_{ij}=\frac{a_{ij}-\sum_{k<j}l_{ik}l_{jk}}{l_{jj}}",
                calculations=[
                    rf"l_{{{i + 1},{j + 1}}}=\frac{{({number(w[i, j])})-({correction_terms})}}{{{number(denominator)}}}\approx {number(l[i, j])}"
                ]
                if trace.enabled
                else [],
                L=l,
            )
    if variant == "ldlt":
        factors = {"L": l, "D": np.diag(d)}
        return factors, "A=LDL^T", l @ np.diag(d) @ l.T, None
    return {"L": l}, "A=LL^T", l @ l.T, None


def _gram_schmidt(a, variant, trace, rtol):
    m, n = a.shape
    if m < n:
        raise MethodError("Gram–Schmidt reduzido exige m ≥ n e colunas linearmente independentes.")
    q, r = np.zeros((m, n)), np.zeros((n, n))
    tol = threshold(a, rtol)
    trace.add(
        "Objetivo: construir uma base ortonormal",
        "Seguindo a Aula 10, escrevemos A por suas colunas a₁,…,aₙ. Vamos obter vetores "
        "q₁,…,qₙ com norma 1 e produtos internos nulos. R guardará os coeficientes "
        "das colunas originais nessa base. Q é reduzida: m×n; se m>n, Q não possui inversa.",
        r"A=[a_1\ \cdots\ a_n]=QR,\qquad Q^TQ=I_n",
        A=a,
    )
    for j in range(n):
        v = a[:, j].copy()
        trace.add(
            f"Separar a coluna a{j + 1}",
            "Este é o vetor cuja direção queremos incorporar à base. "
            + (
                "Na primeira coluna ainda não existem projeções a retirar."
                if j == 0
                else f"Retiraremos suas componentes nas {j} direções já construídas."
            ),
            rf"w_{{{j + 1}}}\leftarrow a_{{{j + 1}}}",
            A=a,
            coluna=v[:, None],
        )
        for i in range(j):
            source = a[:, j] if variant == "qr_classical" else v.copy()
            r[i, j] = q[:, i] @ source
            if trace.enabled:
                calculation = " + ".join(
                    f"({left:.8g} × {right:.8g})" for left, right in zip(q[:, i], source)
                )
                trace.add(
                    f"Calcular o coeficiente r[{i + 1},{j + 1}]",
                    f"Produto interno: {calculation} = {r[i, j]:.10g}. "
                    "Como q tem norma 1, esse escalar mede a componente do vetor na direção de q. "
                    + (
                        "O clássico usa a coluna a original em cada produto."
                        if variant == "qr_classical"
                        else "O modificado usa o residual w atualizado."
                    ),
                    rf"r_{{{i + 1},{j + 1}}}=q_{{{i + 1}}}^T"
                    + (rf"a_{{{j + 1}}}" if variant == "qr_classical" else rf"w_{{{j + 1}}}"),
                    calculations=[
                        rf"r_{{{i + 1},{j + 1}}}={products(q[:, i], source)}\approx {number(r[i, j])}"
                    ],
                    direcao=q[:, i, None],
                    vetor_do_produto=source[:, None],
                    R=r,
                )
            before = v.copy()
            projection = r[i, j] * q[:, i]
            v -= projection
            trace.add(
                f"Subtrair a projeção na direção q{i + 1}",
                f"Multiplicamos q{i + 1} pelo coeficiente {r[i, j]:.10g} para obter o vetor "
                "projetado. Depois calculamos residual novo = residual anterior − projeção. "
                "É a construção do complemento ortogonal apresentada na Aula 10; "
                "em aritmética exata, o novo residual é perpendicular à direção retirada.",
                rf"p=r_{{{i + 1},{j + 1}}}q_{{{i + 1}}},\qquad w_{{{j + 1}}}\leftarrow w_{{{j + 1}}}-p",
                calculations=[
                    rf"p_{{{row + 1}}}=({number(r[i, j])})({number(q[row, i])})\approx {number(projection[row])},\quad w_{{{row + 1}}}^{{novo}}=({number(before[row])})-({number(projection[row])})\approx {number(v[row])}"
                    for row in range(m)
                ]
                if trace.enabled
                else [],
                antes=before[:, None],
                projecao=projection[:, None],
                depois=v[:, None],
            )
        r[j, j] = np.linalg.norm(v)
        if r[j, j] <= tol:
            raise MethodError(
                f"A coluna {j + 1} fica com norma {r[j, j]:.6g} ≤ {tol:.3g} após as projeções. "
                "Esta variante não constrói uma base completa quando as colunas são dependentes. "
                "QR por Householder/Givens e SVD continuam possíveis."
            )
        trace.add(
            f"Calcular a norma r[{j + 1},{j + 1}]",
            f"Depois de retirar todas as projeções, a norma do residual é {r[j, j]:.10g}. "
            "Ela mede o comprimento da parte independente desta coluna e fica na diagonal de R. "
            "Uma norma nula indicaria dependência linear e impediria esta normalização.",
            rf"r_{{{j + 1},{j + 1}}}=\|w_{{{j + 1}}}\|_2=\sqrt{{\sum_\ell w_\ell^2}}",
            calculations=[norm_calculation(v, rf"r_{{{j + 1},{j + 1}}}", r[j, j])]
            if trace.enabled
            else [],
            residual=v[:, None],
            R=r,
        )
        q[:, j] = v / r[j, j]
        trace.add(
            f"Normalizar q{j + 1}",
            f"Dividimos cada componente do residual por {r[j, j]:.10g}. "
            f"O vetor q{j + 1} passa a ter norma 1. As primeiras {j + 1} colunas de Q "
            "geram o mesmo subespaço das primeiras colunas de A; as demais colunas ainda são espaços reservados.",
            rf"q_{{{j + 1}}}=w_{{{j + 1}}}/r_{{{j + 1},{j + 1}}},\qquad \|q_{{{j + 1}}}\|_2=1",
            calculations=divisions(v, r[j, j], rf"(q_{{{j + 1}}})") if trace.enabled else [],
            Q=q,
            R=r,
        )
    trace.add(
        "Montar QR e verificar a ortonormalidade",
        "Cada coluna aⱼ é a combinação r₁ⱼq₁+…+rⱼⱼqⱼ. Não há coeficientes com i>j, "
        "por isso R é triangular superior. QᵀQ deve ser aproximadamente identidade. "
        "Se Q for retangular, QQᵀ é um projetor e não precisa ser identidade.",
        r"a_j=\sum_{i=1}^j r_{ij}q_i,\qquad A=QR,\qquad Q^TQ\approx I_n",
        Q=q,
        R=r,
        QtQ=q.T @ q,
    )
    return {"Q": q, "R": r}, "A=QR", q @ r, float(np.linalg.norm(q.T @ q - np.eye(n)))


def _qr_orthogonal(a, variant, trace, rtol):
    m, n = a.shape
    q, r = np.eye(m), a.copy()
    trace.add(
        "Objetivo: transformar A preservando comprimentos",
        "Q começa como identidade e R como A. Aplicaremos transformações ortogonais "
        "para zerar entradas abaixo da diagonal. A relação A=QR é preservada a cada "
        "operação. Q será quadrada m×m; R será triangular superior ou trapezoidal m×n.",
        r"A=QR,\qquad Q^TQ=I_m",
        Q=q,
        R=r,
    )

    if variant == "qr_householder":
        for k in range(min(m, n)):
            x = r[k:, k].copy()
            norm = np.linalg.norm(x)
            if norm == 0:
                trace.add(f"Coluna {k + 1} já nula", "Nenhuma reflexão é necessária.", R=r)
                continue
            alpha = -np.copysign(norm, x[0])
            trace.add(
                f"Preparar a reflexão da coluna {k + 1}",
                f"O trecho ativo x tem norma {norm:.10g}. Queremos levá-lo a αe₁, "
                f"com α={alpha:.10g}, deixando as demais componentes nulas. Escolhemos "
                "o sinal oposto ao de x₁ para reduzir cancelamento ao construir x−αe₁; "
                "na fórmula, adotamos sign(0)=1.",
                r"\alpha=-\operatorname{sign}(x_1)\|x\|_2,\quad Hx=\alpha e_1",
                calculations=[
                    norm_calculation(x, r"\|x\|_2", norm),
                    rf"\alpha=-({number(np.copysign(1.0, x[0]))})({number(norm)})={number(alpha)}",
                ]
                if trace.enabled
                else [],
                x=x[:, None],
            )
            v = x.copy()
            v[0] -= alpha
            unnormalized = v.copy()
            length = np.linalg.norm(v)
            v /= length
            h = np.eye(m)
            h[k:, k:] -= 2 * np.outer(v, v)
            trace.add(
                f"Reflexão de Householder {k + 1}",
                f"Escolhemos α={alpha:.10g} com sinal oposto a x₁ para evitar cancelamento. "
                "O vetor v é a normal unitária do plano de reflexão. H=I−2vvᵀ "
                "é simétrica e ortogonal, portanto Hᵀ=H e H²=I. "
                "H anula os elementos abaixo da diagonal; acumulamos Q←QH e R←HR, "
                "preservando QR porque QHH R=QR antes da transformação.",
                r"v=\frac{x-\alpha e_1}{\|x-\alpha e_1\|_2},\quad H=I-2vv^T",
                calculations=[
                    rf"w_1=({number(x[0])})-({number(alpha)})={number(unnormalized[0])}",
                    norm_calculation(unnormalized, r"\|w\|_2", length),
                ]
                + divisions(unnormalized, length, "v")
                + [
                    rf"H_{{{k + i + 1},{k + j + 1}}}={int(i == j)}-2({number(v[i])})({number(v[j])})\approx {number(h[k + i, k + j])}"
                    for i in range(len(v))
                    for j in range(len(v))
                ]
                if trace.enabled
                else [],
                H=h,
                normal=v[:, None],
            )
            r = record_product(trace, h, r, "R_{novo}", "HR_{antes}")
            q = record_product(trace, q, h, "Q_{novo}", "Q_{antes}H")
            r[k + 1 :, k] = 0
            trace.add(
                "Guardar os fatores após a reflexão",
                "Entradas abaixo do pivô, anuladas pela reflexão em aritmética exata, são fixadas em zero para não propagar resíduos de arredondamento.",
                Q=q,
                R=r,
            )
    else:
        for j in range(min(m, n)):
            for i in range(m - 1, j, -1):
                upper, lower = r[i - 1, j], r[i, j]
                if lower == 0:
                    continue
                norm = np.hypot(upper, lower)
                c, s = upper / norm, lower / norm
                g = np.eye(m)
                g[i - 1, i - 1] = g[i, i] = c
                g[i - 1, i], g[i, i - 1] = s, -s
                trace.add(
                    f"Givens: anular R[{i + 1},{j + 1}]",
                    f"Giramos as linhas {i} e {i + 1} com c={c:.10g}, s={s:.10g}. "
                    f"A entrada a eliminar fica −s×({upper:.8g})+c×({lower:.8g})≈0. "
                    "Como c²+s²=1, G preserva comprimentos. Acumulamos R←GR e Q←QGᵀ; "
                    "GᵀG=I garante que o produto QR continua sendo A.",
                    r"c=a/\sqrt{a^2+b^2},\ s=b/\sqrt{a^2+b^2},\quad G=\begin{bmatrix}c&s\\-s&c\end{bmatrix}",
                    calculations=[
                        norm_calculation([upper, lower], "h", norm),
                        rf"c=\frac{{{number(upper)}}}{{{number(norm)}}}={number(c)},\qquad s=\frac{{{number(lower)}}}{{{number(norm)}}}={number(s)}",
                        rf"r_{{{i + 1},{j + 1}}}^{{novo}}=-({number(s)})({number(upper)})+({number(c)})({number(lower)})\approx {number(-s * upper + c * lower)}",
                    ]
                    if trace.enabled
                    else [],
                    G=g,
                )
                r = record_product(trace, g, r, "R_{novo}", "GR_{antes}")
                q = record_product(trace, q, g.T, "Q_{novo}", "Q_{antes}G^T")
                r[i, j] = 0
                trace.add(
                    "Guardar os fatores após a rotação",
                    "A entrada anulada é fixada em zero para remover o resíduo de arredondamento da rotação.",
                    Q=q,
                    R=r,
                )
    trace.add(
        "Concluir QR e conferir a base",
        "As transformações acumuladas formam Q ortogonal. R ficou triangular superior "
        "ou trapezoidal. Produtos internos entre colunas diferentes de Q devem ser zero "
        "e as normas devem ser 1, como na definição apresentada na Aula 10.",
        r"Q^TQ\approx I_m,\qquad A=QR",
        Q=q,
        R=r,
        QtQ=q.T @ q,
    )
    return {"Q": q, "R": r}, "A=QR", q @ r, float(np.linalg.norm(q.T @ q - np.eye(m)))


def _svd(a, trace, rtol):
    """SVD espectral: gramiana escalada, autovetores e normalização de AV."""
    m, n = a.shape
    k = min(m, n)
    scale = float(np.max(np.abs(a))) or 1.0
    b = a / scale
    trace.add(
        "Objetivo da SVD: seguir a construção da Aula 18",
        "Queremos A=UΣVᵀ. Primeiro calculamos a gramiana G=AᵀA. Seus autovetores "
        "ortonormais formam V; as raízes dos autovalores fornecem os valores singulares. "
        "Depois calculamos Y=AV e normalizamos suas colunas não nulas para construir U. "
        "Na aula, a matriz de autovetores é chamada Q e depois renomeada V: V=Q, "
        "portanto Vᵀ=Qᵀ. Essa Q vem da decomposição espectral, não da QR de A.",
        r"G=A^TA=V\Lambda V^T,\qquad \Sigma_{ii}=\sqrt{\lambda_i},\qquad Y=AV=U\Sigma",
        A=a,
    )
    trace.add(
        "Escalar a matriz antes dos produtos",
        f"Usamos s={scale:.10g} e B=A/s para proteger os produtos. Os autovetores "
        "de Gₛ=BᵀB são os mesmos de AᵀA; os autovalores originais são s² vezes "
        "os escalados. A escala global não altera o condicionamento. "
        "Formar a gramiana pode perder precisão nas direções pequenas.",
        r"B=A/s,\qquad G_s=B^TB=G/s^2",
        calculations=[
            rf"B_{{{i + 1},{j + 1}}}=\frac{{{number(a[i, j])}}}{{{number(scale)}}}={number(b[i, j])}"
            for i in range(m)
            for j in range(n)
        ]
        if trace.enabled
        else [],
        B=b,
    )
    gram = record_product(
        trace, b.T, b, "G_s", "B^TB", "A gramiana usa produtos internos entre as colunas de B."
    )
    if trace.enabled:
        trace.add(
            "Relacionar as gramianas escalada e original",
            "Multiplicar Gₛ por s² recupera AᵀA. A diagonal reúne as normas ao quadrado "
            "das colunas de A; fora da diagonal aparecem os produtos internos entre colunas.",
            r"G=s^2G_s=A^TA",
            AtA=a.T @ a,
            gramiana_escalada=gram,
        )

        coefficients = [1.0]
        auxiliary = np.eye(n)
        for j in range(1, n + 1):
            multiplied = gram @ auxiliary
            coefficient = -float(np.trace(multiplied)) / j
            coefficients.append(coefficient)
            auxiliary = multiplied + coefficient * np.eye(n)
            trace.add(
                f"Polinômio característico: coeficiente c{j}",
                "Calculamos os coeficientes de det(tI−Gₛ) pela recorrência de Faddeev–LeVerrier. "
                "B₀=I, cⱼ=−tr(GₛBⱼ₋₁)/j e Bⱼ=GₛBⱼ₋₁+cⱼI. "
                "Este polinômio ilustra a equação dos autovalores; seus coeficientes arredondados "
                "não são usados para calcular raízes no programa.",
                r"p(t)=\det(tI-G_s)=t^n+c_1t^{n-1}+\cdots+c_n",
                calculations=[
                    rf"c_{{{j}}}=-\frac{{{'+'.join(f'({number(x)})' for x in np.diag(multiplied))}}}{{{j}}}\approx {number(coefficient)}"
                ],
                Gs_Banterior=multiplied,
                B_atual=auxiliary,
                coeficientes=np.array(coefficients)[None, :],
            )
        polynomial = "+".join(
            rf"({number(value)})t^{{{n - index}}}" for index, value in enumerate(coefficients)
        )
        trace.add(
            "Equação que determina os autovalores",
            "Os autovalores μ são as raízes de p(t)=0 para a matriz escalada. "
            "Para uma matriz 2×2, também podemos expandir "
            "(t−g₁₁)(t−g₂₂)−g₁₂g₂₁. A resolução numérica será feita na próxima etapa.",
            rf"p(t)={polynomial}=0",
            coeficientes=np.array(coefficients)[None, :],
        )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError as exc:
        raise MethodError(
            "A SVD existe, mas o cálculo numérico dos autovalores da gramiana não convergiu."
        ) from exc
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    trace.add(
        "Calcular e ordenar os autovalores e autovetores",
        "Usamos numpy.linalg.eigh, própria para matrizes reais simétricas, para resolver "
        "Gₛv=μv. A biblioteca retorna autovetores unitários; ordenamos os pares pelo "
        "autovalor decrescente. As iterações internas da biblioteca não são apresentadas "
        "como contas manuais. A seguir verificamos cada equação com os valores retornados. "
        "Autovalores repetidos admitem diferentes bases ortonormais igualmente válidas.",
        r"G_sv_i=\mu_i v_i,\qquad \lambda_i=s^2\mu_i",
        autovalores_escalados=eigenvalues[:, None],
        autovetores=eigenvectors,
    )
    for i in range(n):
        vector = eigenvectors[:, i].copy()
        length = np.linalg.norm(vector)
        eigenvectors[:, i] = vector / length
        gv = gram @ eigenvectors[:, i]
        residual = gv - eigenvalues[i] * eigenvectors[:, i]
        if trace.enabled:
            trace.add(
                f"Verificar e normalizar o autovetor v{i + 1}",
                f"Para μ={eigenvalues[i]:.10g}, a equação é (Gₛ−μI)v=0. "
                "O vetor veio do resolvedor simétrico; mostramos abaixo cada linha de Gₛv "
                "e sua comparação com μv. Dividimos pela norma para manter o vetor unitário. "
                f"A norma do resíduo da equação de autovetor é {np.linalg.norm(residual):.4g}.",
                r"(G_s-\mu_i I)v_i=0,\qquad \|v_i\|_2=1",
                calculations=[norm_calculation(vector, r"\|v\|_2", length)]
                + divisions(vector, length, "v")
                + [
                    rf"(G_sv)_{{{row + 1}}}={products(gram[row, :], eigenvectors[:, i])}\approx {number(gv[row])},\quad \mu v_{{{row + 1}}}=({number(eigenvalues[i])})({number(eigenvectors[row, i])})\approx {number(eigenvalues[i] * eigenvectors[row, i])}"
                    for row in range(n)
                ],
                Gs_menos_muI=gram - eigenvalues[i] * np.eye(n),
                v=eigenvectors[:, i, None],
                residuo_autovetor=residual[:, None],
            )
    transformed = record_product(
        trace, b, eigenvectors, "Y_s", "BV", "Esta é a versão escalada de Y=AV da aula."
    )
    norms = np.linalg.norm(transformed, axis=0)
    refined = norms**2
    floor = 10 * np.finfo(float).eps * max(m, n) * (float(norms.max()) if norms.size else 0.0)
    order = np.argsort(-norms, kind="stable")[:k]
    norms, refined, raw = norms[order], refined[order], eigenvalues[order]
    v, transformed = eigenvectors[:, order], transformed[:, order]
    values = scale * np.sqrt(refined)
    u = np.zeros((m, k))
    for i in range(k):
        trace.add(
            f"Calcular o valor singular σ{i + 1}",
            f"O autovalor escalado retornado foi μ={raw[i]:.10g}. Em aritmética exata, "
            "μ=vᵀGₛv=‖Bv‖². Reavaliamos μ pela soma dos quadrados de Bv, que é "
            "não negativa e evita amplificar pequenos autovalores espúrios de direções nulas. "
            "Esta reavaliação não corrige um autovetor impreciso. "
            f"O valor reavaliado é {refined[i]:.10g}; σ=s√μ={values[i]:.10g}.",
            r"\mu_i=\|Bv_i\|_2^2,\quad \lambda_i=s^2\mu_i,\quad \sigma_i=\sqrt{\lambda_i}=s\sqrt{\mu_i}",
            calculations=[
                rf"\mu_{{{i + 1}}}={'+'.join(f'({number(x)})^2' for x in transformed[:, i])}\approx {number(refined[i])}",
                rf"\lambda_{{{i + 1}}}=({number(scale)})^2({number(refined[i])})\approx {number(scale**2 * refined[i])}",
                rf"\sigma_{{{i + 1}}}=({number(scale)})\sqrt{{{number(refined[i])}}}\approx {number(values[i])}",
            ]
            if trace.enabled
            else [],
            y=(scale * transformed[:, i])[:, None],
        )
        if norms[i] > floor:
            u[:, i] = transformed[:, i] / norms[i]
            trace.add(
                f"Normalizar a direção singular {i + 1}",
                "Cada entrada de u é a entrada correspondente de y dividida por σ. "
                "Computamos a divisão equivalente na escala protegida: u=(Bv)/√μ. "
                "Isso preserva a norma unitária e evita trabalhar com números muito grandes.",
                r"u_i=Av_i/\sigma_i=(Bv_i)/\sqrt{\mu_i}",
                calculations=divisions(scale * transformed[:, i], values[i], "u")
                if trace.enabled
                else [],
                y=(scale * transformed[:, i])[:, None],
                direcao_unitaria=u[:, i, None],
            )
        else:
            values[i] = 0.0
            for index in range(m):
                candidate = np.eye(m)[:, index]
                if trace.enabled:
                    trace.add(
                        "Testar vetor para completar a base",
                        f"Testamos e{index + 1}. Retiraremos duas vezes as projeções nas direções já construídas para reduzir arredondamento.",
                        candidato=candidate[:, None],
                    )
                for repetition in range(2):
                    before = candidate.copy()
                    coordinates = u[:, :i].T @ candidate
                    candidate -= u[:, :i] @ coordinates
                    if trace.enabled and i:
                        trace.add(
                            f"Retirar projeções do candidato · passagem {repetition + 1}",
                            "Calculamos os produtos internos com as colunas anteriores de U e subtraímos a combinação desses vetores.",
                            r"w\leftarrow w-U_{anteriores}(U_{anteriores}^Tw)",
                            calculations=[
                                rf"w_{{{row + 1}}}=({number(before[row])})-({products(u[row, :i], coordinates)})\approx {number(candidate[row])}"
                                for row in range(m)
                            ],
                            antes=before[:, None],
                            depois=candidate[:, None],
                        )
                length = np.linalg.norm(candidate)
                if length > 1e-10:
                    u[:, i] = candidate / length
                    trace.add(
                        f"Completar vetor de valor singular nulo {i + 1}",
                        f"A norma de Bv está dentro do limiar de máquina {floor:.4g}. Não dividimos por σ=0. "
                        "Normalizamos o candidato ortogonal; esta coluna não contribui à reconstrução porque seu valor singular é zero.",
                        r"\sigma_i=0,\qquad u_i=w/\|w\|_2",
                        calculations=[norm_calculation(candidate, r"\|w\|_2", length)]
                        + divisions(candidate, length, "u")
                        if trace.enabled
                        else [],
                        U=u,
                    )
                    break
            else:
                raise MethodError(
                    "Não foi possível completar a base ortogonal da SVD numericamente."
                )
    sigma = np.diag(values)
    trace.add(
        "Montar os fatores da SVD reduzida",
        f"Usamos k=min({m},{n})={k} direções: U tem dimensão {m}×{k}, Σ é {k}×{k} "
        f"e Vᵀ é {k}×{n}. Os valores singulares estão em ordem decrescente. "
        "Em matrizes largas, selecionamos somente k autovetores de AᵀA e completamos U quando necessário. "
        "Sinais dos vetores e bases em autovalores repetidos podem diferir de outros programas.",
        r"A=U\Sigma V^T",
        U=u,
        Sigma=sigma,
        Vt=v.T,
    )
    if trace.enabled:
        y = a @ v
        trace.add(
            "Conferir Y=AV e os quadrados dos valores singulares",
            "Verificamos Y≈UΣ e YᵀY≈Σ². A diagonal contém os autovalores; "
            "entradas fora da diagonal avaliam a ortogonalidade das direções transformadas.",
            r"Y=AV\approx U\Sigma,\qquad Y^TY\approx\Sigma^2",
            Y=y,
            U_Sigma=u @ sigma,
            YtY=y.T @ y,
            Sigma_quadrado=sigma @ sigma,
        )
        partial = np.zeros_like(a)
        for index in range(k):
            term = values[index] * np.outer(u[:, index], v[:, index])
            before = partial.copy()
            partial += term
            trace.add(
                f"Reconstrução: somar o termo singular {index + 1}",
                "Cada entrada do termo é σᵢ vezes uma componente de uᵢ vezes uma componente de vᵢ. "
                "Depois somamos esse termo à reconstrução acumulada.",
                rf"A_{{{index + 1}}}=A_{{{index}}}+\sigma_{{{index + 1}}}u_{{{index + 1}}}v_{{{index + 1}}}^T",
                calculations=[
                    rf"T_{{{row + 1},{col + 1}}}=({number(values[index])})({number(u[row, index])})({number(v[col, index])})\approx {number(term[row, col])},\quad (A_{{{index + 1}}})_{{{row + 1},{col + 1}}}=({number(before[row, col])})+({number(term[row, col])})\approx {number(partial[row, col])}"
                    for row in range(m)
                    for col in range(n)
                ],
                termo=term,
                soma_parcial=partial,
            )
    ortho = max(np.linalg.norm(u.T @ u - np.eye(k)), np.linalg.norm(v.T @ v - np.eye(k)))
    return {"U": u, "Sigma": sigma, "Vt": v.T}, r"A=U\Sigma V^T", u @ sigma @ v.T, float(ortho)


def decompose(value, method, rtol=DEFAULT_RTOL, record_steps=True):
    a = validate_matrix(value)
    validate_rtol(rtol)
    if method not in METHODS:
        raise ValueError("Método desconhecido.")
    trace = Trace(enabled=record_steps)
    if method.startswith("lu_"):
        factors, identity, rebuilt, ortho = _lu(a, method, trace, rtol)
    elif method in {"cholesky", "ldlt"}:
        factors, identity, rebuilt, ortho = _cholesky(a, method, trace, rtol)
    elif method in {"qr_classical", "qr_modified"}:
        factors, identity, rebuilt, ortho = _gram_schmidt(a, method, trace, rtol)
    elif method.startswith("qr_"):
        factors, identity, rebuilt, ortho = _qr_orthogonal(a, method, trace, rtol)
    else:
        factors, identity, rebuilt, ortho = _svd(a, trace, rtol)
    if not all(np.isfinite(f).all() for f in factors.values()):
        raise MethodError(
            "Os fatores excederam a precisão numérica. Reescale a matriz ou use pivotamento."
        )
    error = relative_error(a, rebuilt)
    warnings = []
    if method == "svd":
        singular_values = np.diag(factors["Sigma"])
        eig_condition = (
            singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
        )
        if not np.isfinite(eig_condition) or eig_condition > 1e7:
            warnings.append(
                "SVD por autovalores: formar AᵀA pode perder precisão nas direções pequenas. "
                "A escala global protege magnitudes, mas não corrige esse efeito. "
                "Examine ortogonalidade, reconstrução e resíduo; em uso científico, prefira SVD direta."
            )
    if error > 100 * rtol:
        warnings.append(
            f"Erro relativo de reconstrução elevado: {error:.3g}. Prefira um método estável."
        )
    if ortho is not None and ortho > 100 * rtol:
        warnings.append(
            f"Perda de ortogonalidade: {ortho:.3g}. "
            + (
                "A SVD pela gramiana pode perder direções com mau condicionamento."
                if method == "svd"
                else "As transformações ou projeções acumularam erro de arredondamento."
            )
        )
    if trace.enabled:
        if method.startswith("lu_"):
            checked = record_product(trace, factors["L"], factors["U"], "T", "LU")
            if "P" in factors:
                checked = record_product(trace, factors["P"].T, checked, "S", "P^TT")
            if "C" in factors:
                record_product(trace, checked, factors["C"].T, r"\widehat A", "SC^T")
        elif method == "cholesky":
            record_product(trace, factors["L"], factors["L"].T, r"\widehat A", "LL^T")
        elif method == "ldlt":
            checked = record_product(trace, factors["L"], factors["D"], "T", "LD")
            record_product(trace, checked, factors["L"].T, r"\widehat A", "TL^T")
        elif method.startswith("qr_"):
            record_product(trace, factors["Q"], factors["R"], r"\widehat A", "QR")
        else:
            checked = record_product(trace, factors["U"], factors["Sigma"], "T", r"U\Sigma")
            record_product(trace, checked, factors["Vt"], r"\widehat A", "TV^T")
        if method.startswith("qr_"):
            record_product(
                trace,
                factors["Q"].T,
                factors["Q"],
                "K",
                "Q^TQ",
                "Na matriz final, diagonal ≈1 e demais entradas ≈0.",
            )
        elif method == "svd":
            record_product(trace, factors["U"].T, factors["U"], "K", "U^TU")
            record_product(trace, factors["Vt"], factors["Vt"].T, "K", "V^TV")
    trace.add(
        "Verificação da decomposição",
        f"Reconstruímos a matriz a partir dos fatores. Erro relativo de Frobenius: {error:.10g}. "
        "Em ponto flutuante, esperamos um erro pequeno, e não igualdade decimal exata.",
        r"\varepsilon=\frac{\|A-\widehat A\|_F}{\|A\|_F}",
        calculations=[
            rf"\Delta_{{{i + 1},{j + 1}}}=({number(a[i, j])})-({number(rebuilt[i, j])})\approx {number(a[i, j] - rebuilt[i, j])}"
            for i in range(a.shape[0])
            for j in range(a.shape[1])
        ]
        + [
            norm_calculation((a - rebuilt).ravel(), r"\|\Delta\|_F", stable_norm(a - rebuilt)),
            norm_calculation(a.ravel(), r"\|A\|_F", stable_norm(a)),
            rf"\varepsilon\approx {number(error)}",
        ]
        if trace.enabled
        else [],
        A=a,
        reconstruida=rebuilt,
    )
    return Decomposition(method, identity, factors, trace.steps, error, warnings, ortho)
