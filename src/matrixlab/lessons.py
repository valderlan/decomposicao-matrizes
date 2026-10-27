def lesson_for(method, shape):
    m, n = shape
    k = min(m, n)
    if method.startswith("qr_"):
        reduced = method in {"qr_classical", "qr_modified"}
        columns = n if reduced else m
        return rf"""
### Como interpretar a decomposição QR

**1. O que queremos construir?** Escrevemos $A=[a_1\;\cdots\;a_n]=QR$.
As colunas de Q têm norma 1 e produto interno zero entre si.
R guarda os coeficientes que permitem reconstruir cada coluna original.

**2. Como retirar uma projeção?** Para um vetor unitário $q_i$, o número
$r_{{ij}}=\langle q_i,a_j\rangle=q_i^Ta_j$ mede a componente de $a_j$ na direção de $q_i$.
A projeção é o **vetor** $r_{{ij}}q_i$. Subtraí-la deixa apenas a componente perpendicular.
Não confunda o coeficiente escalar com o vetor projetado.

$$
w_j=a_j-\sum_{{i<j}}r_{{ij}}q_i,\qquad r_{{jj}}=\|w_j\|_2,\qquad q_j=w_j/r_{{jj}}.
$$

O primeiro vetor é $q_1=a_1/\|a_1\|_2$. Nos próximos, removemos as direções já construídas.
No clássico, todos os produtos internos usam $a_j$ original; no modificado, cada produto usa
o residual atualizado. Em aritmética exata, os dois processos coincidem; em ponto flutuante,
o modificado costuma preservar melhor a ortogonalidade.

**3. Por que R é triangular superior?** A coluna $a_j$ usa apenas $q_1,\ldots,q_j$:

$$
a_j=r_{{1j}}q_1+\cdots+r_{{jj}}q_j.
$$

Logo, os coeficientes abaixo da diagonal são zero. Se $w_j=0$, aquela coluna não acrescenta
uma direção independente; a variante Gram–Schmidt reduzida deste projeto não pode dividir
por sua norma. Householder/Givens podem continuar uma QR com posto deficiente.

**4. Dimensões para esta entrada:**

| Matriz | Dimensão |
|---|---|
| A | {m} × {n} |
| Q nesta variante | {m} × {columns} |
| R nesta variante | {columns} × {n} |

$Q^TQ=I$ para as colunas completas da Q final. Se Q for retangular alta,
**ela não tem inversa**: $QQ^T$ é a projeção sobre o espaço de suas colunas,
e em geral $QQ^T\ne I$. Apenas uma Q quadrada ortogonal satisfaz $Q^{{-1}}=Q^T$.

**5. Como QR resolve mínimos quadrados?** Para $m\ge n$ e colunas independentes,
usamos a parte $Q_1$ com n colunas e o bloco triangular $R_1$ de ordem n.

$$
\|Ax-b\|_2^2=\|R_1x-Q_1^Tb\|_2^2+\|b-Q_1Q_1^Tb\|_2^2.
$$

O segundo termo não depende de x. Portanto resolvemos $R_1x=Q_1^Tb$ por
substituição regressiva, sem formar $A^TA$ nem inverter matrizes.
O vetor $Q_1Q_1^Tb$ é a aproximação de b no espaço das colunas de A;
o residual fica perpendicular a esse espaço.

**Outras construções de QR:** Householder usa uma reflexão $H=I-2vv^T$;
Givens usa uma rotação em duas linhas. Ambas preservam comprimentos e produtos internos.
Os passos mostram as transformações que anulam elementos de R e como Q é acumulada.
Essas variantes complementam o Gram–Schmidt apresentado na Aula 10.
"""
    if method == "svd":
        return rf"""
### Como interpretar a decomposição SVD

**1. Por que olhar para $A^TA$?** Essa matriz é simétrica e positiva semidefinida:
$z^TA^TAz=\|Az\|_2^2\ge0$. Seus autovalores são não negativos, e seus autovetores
podem ser escolhidos ortonormais. Chamamos esses vetores de $v_i$.

$$
A^TAv_i=\lambda_i v_i,\qquad \sigma_i=\sqrt{{\lambda_i}}.
$$

Os $\sigma_i$ são os **valores singulares de A**; os $\lambda_i=\sigma_i^2$
são autovalores de $A^TA$. Valores singulares não são, em geral, autovalores de A.

**2. Como obter U?** Mudamos as direções de entrada por V e definimos $Y=AV$.
As colunas $y_i=Av_i$ são ortogonais, pois

$$
y_i^Ty_j=v_i^TA^TAv_j=\lambda_j\,v_i^Tv_j.
$$

Assim, $\|y_i\|_2=\sigma_i$. Para $\sigma_i>0$, normalizamos
$u_i=y_i/\sigma_i$, de onde $Av_i=\sigma_i u_i$. Se $\sigma_i=0$,
não dividimos por zero: completamos uma base ortonormal no complemento das direções anteriores.

**3. Como os fatores se juntam?** Como $Y=U\Sigma$, recuperamos
$A=U\Sigma V^T$. Cada fator tem uma função:

| Fator | Interpretação |
|---|---|
| Vᵀ | Coordenadas nas direções de entrada; os vetores direitos são as colunas de V |
| Σ | Intensidade de cada direção, com $\sigma_1\ge\cdots\ge\sigma_k\ge0$ |
| U | Direções de saída, chamadas vetores singulares esquerdos |

**4. SVD reduzida e completa para esta entrada:** $k=\min(m,n)={k}$.

| Fator | Reduzida, calculada pelo painel | Completa, explicada na aula |
|---|---|---|
| U | {m} × {k} | {m} × {m} |
| Σ | {k} × {k} | {m} × {n} |
| Vᵀ | {k} × {n} | {n} × {n} |

A completa acrescenta direções ortonormais e linhas/colunas nulas de Σ.
O painel calcula a reduzida, que preserva a reconstrução e a pseudoinversa.
Nas versões retangulares, as colunas de U e V são ortonormais, mas não se deve
atribuir uma inversa a uma matriz retangular. Para $m<n$, usamos a mesma regra de k.

**5. Como o algoritmo chega a V?** O painel agora segue a construção espectral da aula.
Calculamos $B=A/s$ e $G_s=B^TB$, com s igual ao maior módulo de A (s=1 se A for nula).
Cada entrada de $G_s$ é mostrada como soma de produtos. Os autovalores e autovetores
são calculados numericamente por `numpy.linalg.eigh`, uma rotina para matrizes simétricas.
O painel mostra o polinômio característico, os pares retornados e a verificação de cada
linha de $(G_s-\mu_iI)v_i=0$, sem inventar as iterações internas da biblioteca.

Na página 12, a aula renomeia a matriz de autovetores Q para V: **V=Q e Vᵀ=Qᵀ**.
Essa Q pertence à decomposição espectral de $A^TA$; ela não é necessariamente
a Q da fatoração QR de A. Como $Y=AV=U\Sigma$ e V é ortogonal na forma completa,
obtemos $A=YV^T=U\Sigma V^T$.

Reavaliamos cada autovalor escalado pela identidade $\mu_i=\|Bv_i\|_2^2$;
assim $\lambda_i=s^2\mu_i$ e $\sigma_i=s\sqrt{{\mu_i}}$.
Em aritmética exata isso coincide com o autovalor da gramiana. A reavaliação evita
raízes de pequenos autovalores negativos por arredondamento e ajuda a identificar
direções nulas, mas não recupera autovetores imprecisos. Normas dentro de um limiar
de máquina são tratadas como zero; a tolerância escolhida pelo usuário controla
o corte adicional da pseudoinversa.

Para uma matriz larga, calculamos a mesma gramiana e selecionamos k direções para
os fatores reduzidos. Completamos U nas direções de valor singular nulo.
**Precisão:** formar $A^TA$ pode agravar o condicionamento e prejudicar as direções
pequenas. A escala global protege magnitudes; não elimina essa perda de precisão.
O painel avisa sobre condicionamento alto e perda de ortogonalidade. Para uso
científico com dados difíceis, uma SVD direta é preferível à construção pela gramiana.

**6. Como reconstruir e reduzir dimensões?**

$$
A=\sum_{{i=1}}^k\sigma_i u_i v_i^T,\qquad
A_r=\sum_{{i=1}}^r\sigma_i u_i v_i^T.
$$

Cada termo tem posto no máximo 1. Manter os primeiros r termos produz a melhor
aproximação de posto no máximo r nas normas 2 e Frobenius; a energia descartada
é $\sum_{{i>r}}\sigma_i^2$. Essa relação é explicada aqui; o painel mostra a soma dos termos
na reconstrução, sem alterar automaticamente a matriz informada.

**7. Como resolver um sistema singular?** Usamos
$A^+=V\Sigma^+U^T$ e $x=A^+b$. Invertendo apenas valores singulares acima
do corte numérico, obtemos mínimos quadrados de norma mínima sob esse critério.

**Sobre dados e PCA:** a aula parte de dados centralizados. A SVD também vale para
qualquer matriz real, sem centralização. O painel preserva A; não remove médias.
Somente com dados centralizados e a normalização de covariância apropriada
se interpretam $\sigma_i^2/(m-1)$ como variâncias amostrais, para $m>1$.
"""
    return ""
