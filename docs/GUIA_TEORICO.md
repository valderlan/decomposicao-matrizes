# Guia teórico de decomposições matriciais

Este guia acompanha a ferramenta Python/uv/Streamlit. As tabelas de dimensões usam uma entrada de referência 3×3; no painel elas são adaptadas à matriz informada. Os exemplos resolvidos têm as dimensões indicadas em cada seção.

Fontes didáticas fornecidas pelo usuário: Thelmo de Araujo, Álgebra Linear — Seção 1.3 (LU) e Fundamentos de Análise de Dados — Aulas 10 (QR) e 18 (SVD). As cópias estão em `materiais/`.
## 1 · Fundamentos e leitura de Ax=b

### Sistemas e decomposições: o que está sendo calculado?

Um sistema $Ax=b$ reúne equações lineares. A matriz A guarda os coeficientes,
x guarda as incógnitas e b guarda os resultados conhecidos. Se A tem m linhas e n colunas,
há m equações e n incógnitas. Exemplo prático: a é o preço de um caderno e c é o preço
de uma caneta. Dois cadernos e uma caneta custam 5 reais; quatro cadernos e três canetas
custam 11 reais. Assim:

$$
2a+c=5,\quad 4a+3c=11,\qquad
A=\begin{bmatrix}2&1\\4&3\end{bmatrix},\quad
x=\begin{bmatrix}a\\c\end{bmatrix},\quad b=\begin{bmatrix}5\\11\end{bmatrix}.
$$

A solução é a=2 e c=1: um caderno custa 2 reais e uma caneta custa 1 real.
Verificação: 2×2+1=5 e 4×2+3×1=11. Esse exemplo está no seletor
**LU • exemplo simples do guia (2 × 2)**; escolha Doolittle sem pivotamento.
Uma decomposição escreve A como um produto de matrizes mais simples. Isso organiza os
cálculos e permite reutilizar os fatores se apenas b mudar, por exemplo em novas medições.

### Propriedades que orientam a escolha

| Propriedade | Significado | Exemplo / efeito |
|---|---|---|
| Quadrada | Mesmo número de linhas e colunas | LU e Cholesky deste painel exigem isso |
| Triangular inferior | Zeros acima da diagonal | Resolver começando pela primeira equação |
| Triangular superior | Zeros abaixo da diagonal | Resolver começando pela última equação |
| Simétrica | $A=A^T$ | As entradas aᵢⱼ e aⱼᵢ são iguais |
| Positiva definida | $z^TAz>0$ para todo z≠0 | Com simetria, permite Cholesky |
| Colunas ortonormais | Normas 1, produtos internos zero | $Q^TQ=I$ |
| Posto | Número de direções independentes | Dependências reduzem a informação do sistema |

Uma matriz quadrada com posto completo tem solução única para qualquer b.
Se o posto for deficiente, pode haver infinitas soluções ou nenhuma solução exata,
dependendo de b. Exemplo: x₁+x₂=3 e 2x₁+2x₂=6 repetem a mesma informação;
substituir o último resultado por 7 torna as equações incompatíveis.

**Fatorar e resolver são tarefas distintas.** Uma matriz singular ainda pode admitir
LU, QR ou SVD. Um fator triangular singular, porém, não permite a resolução usual
por divisão em todos os pivôs. SVD fornece uma alternativa para mínimos quadrados/norma mínima.

### Produto interno, norma e projeção

O produto interno mede alinhamento: $(1,1)^T(1,-1)=1-1=0$, logo os vetores são ortogonais.
A norma euclidiana mede comprimento: $\|(3,4)\|_2=\sqrt{9+16}=5$.
Normalizar é dividir pelo comprimento: $(3/5,4/5)$ tem norma 1.
Para uma direção unitária q, $(q^Tv)q$ é a projeção de v nessa direção.
Essa ideia fundamenta Gram–Schmidt, mínimos quadrados e a interpretação de SVD.

## 2 · LU: matrizes elementares, eliminação e sistemas

### LU como eliminação de Gauss registrada

**Referência:** Thelmo de Araujo, *Álgebra Linear — Seção 1.3: Decomposição LU*,
pp. 2–8 (matrizes elementares), 9–13 (construção da fatoração) e 14–18 (sistemas triangulares).

Na forma Doolittle, $A=LU$: L é triangular inferior com diagonal 1,
e U é triangular superior. U resulta da eliminação de Gauss; L registra como desfazê-la.

### 1. De uma operação de linha à matriz elementar

Uma matriz elementar é obtida aplicando **uma** operação de linha à identidade.
Multiplicá-la por A à esquerda aplica a mesma operação em A:

| Operação | Matriz elementar | Como desfazer |
|---|---|---|
| Multiplicar a linha i por α≠0 | Identidade com aᵢᵢ=α | Multiplicar por 1/α |
| Trocar linhas i e j | Identidade com as linhas trocadas | Repetir a troca |
| Somar α vezes a linha k à linha i | Identidade com α na posição (i,k) | Somar −α vezes essa linha |

Na eliminação, queremos $u_{ik}-m_{ik}u_{kk}=0$; portanto
$m_{ik}=u_{ik}/u_{kk}$. Aplicamos linhaᵢ←linhaᵢ−mᵢₖlinhaₖ.
E contém **−m**, e E⁻¹ contém **+m**. Para i≠k, a matriz $e_ie_k^T$ tem um único 1 em (i,k):

$$
E=I-m_{ik}e_ie_k^T,\qquad E^{-1}=I+m_{ik}e_ie_k^T.
$$

### 2. Exemplo simples completo: A=LU

$$
A=\begin{bmatrix}2&1\\4&3\end{bmatrix}.
$$

O pivô é 2 e queremos eliminar 4. Calculamos m₂₁=4/2=2 e fazemos
linha₂←linha₂−2linha₁. Entrada por entrada: 4−2×2=0 e 3−2×1=1.

$$
E=\begin{bmatrix}1&0\\-2&1\end{bmatrix},\quad
EA=\begin{bmatrix}2&1\\0&1\end{bmatrix}=U,\quad
L=E^{-1}=\begin{bmatrix}1&0\\2&1\end{bmatrix}.
$$

Verificamos o produto: a segunda linha de LU é 2 vezes a primeira linha de U mais
a segunda linha de U, isto é, (4,3). Assim recuperamos A.

Também há um exemplo 3×3 com todos os fatores inteiros, no seletor
**LU • fatores inteiros (3 × 3)**:

$$
A=\begin{bmatrix}2&1&1\\4&5&3\\2&7&7\end{bmatrix},\quad
L=\begin{bmatrix}1&0&0\\2&1&0\\1&2&1\end{bmatrix},\quad
U=\begin{bmatrix}2&1&1\\0&3&1\\0&0&4\end{bmatrix}.
$$

Em Doolittle sem pivotamento, os multiplicadores são m₂₁=2, m₃₁=1 e m₃₂=2.
Para b=(7,23,37), Ly=b dá y=(7,9,12) e Ux=y dá x=(1,2,3).
O pivotamento pode escolher outra ordem de linhas/colunas e produzir fatores diferentes.

### 3. Por que L é um produto de inversas?

Sem trocas, depois de s eliminações, $E_s\cdots E_2E_1A=U$.
Desfazer as operações dá

$$
A=E_1^{-1}E_2^{-1}\cdots E_s^{-1}U=LU.
$$

A ordem importa: a inversa de um produto é o produto das inversas em ordem reversa.
Produtos dessas matrizes triangulares inferiores continuam triangulares inferiores.
O painel mostra o produto das inversas no caso sem trocas; com pivotamento,
as permutações também precisam ser consideradas.

### 4. Exemplo 3×3 do material

$$
A=\begin{bmatrix}1&2&3\\1&4&7\\-2&2&5\end{bmatrix}.
$$

Os multiplicadores são m₂₁=1, m₃₁=−2 e m₃₂=3. As operações são:

1. Linha₂←linha₂−linha₁, obtendo (0,2,4).
2. Linha₃←linha₃+2linha₁, obtendo (0,6,11).
3. Linha₃←linha₃−3linha₂, obtendo (0,0,−1).

$$
L=\begin{bmatrix}1&0&0\\1&1&0\\-2&3&1\end{bmatrix},\qquad
U=\begin{bmatrix}1&2&3\\0&2&4\\0&0&-1\end{bmatrix}.
$$

O multiplicador −2 fica em L; a operação que elimina a entrada usa +2.
Este exemplo está no seletor **LU • exemplo da Seção 1.3**.

### 5. Resolver dois sistemas triangulares

Para o exemplo 2×2, use b=(5,11). Primeiro $Ly=b$:
y₁=5 e 2y₁+y₂=11, logo y₂=1. Depois $Ux=y$:
x₂=1 e 2x₁+x₂=5, logo x₁=2.

Para o exemplo 3×3 da aula, b=(1,−1,−7). A substituição sucessiva fornece
y=(1,−2,1); a regressiva fornece x=(2,1,−1).
Não é preciso calcular A⁻¹, L⁻¹ ou U⁻¹ para resolver: a referência a E⁻¹ explica
a construção, enquanto a solução usa substituições.

### 6. Doolittle, Crout e pivotamento

| Variante | Identidade | Característica |
|---|---|---|
| Doolittle | A=LU | Diagonal de L igual a 1 |
| Crout | A=LU | Diagonal de U igual a 1 |
| Pivotamento parcial | PA=LU | Escolher maior módulo na coluna ativa; trocar linhas |
| Pivotamento total | PAC=LU | Escolher maior módulo no bloco ativo; trocar linhas e colunas |

Neste projeto, Crout é a forma equivalente obtida transferindo a diagonal da U de Doolittle:
$L_C=L_DD$ e $U_C=D^{-1}U_D$. Para A do exemplo simples:

$$
L_C=\begin{bmatrix}2&0\\4&1\end{bmatrix},\qquad
U_C=\begin{bmatrix}1&1/2\\0&1\end{bmatrix}.
$$

Se $A=\begin{bmatrix}0&1\\1&1\end{bmatrix}$, o primeiro pivô é zero, mas A é invertível.
Trocar as linhas permite continuar. Uma falha sem pivotamento não significa que o sistema
não possa ser resolvido. Mesmo um pivô pequeno não nulo pode gerar grandes multiplicadores;
o pivotamento melhora o comportamento da eliminação, mas não muda o condicionamento intrínseco de A.

Com P, o segundo membro vira Pb. Com C, resolvemos na ordem permutada z e recuperamos x=Cz.
Quando trocamos linhas em etapas posteriores, também trocamos em L os multiplicadores
já calculados. O painel mostra essas operações para preservar a identidade correta.

**Uso prático:** sistemas quadrados gerais e muitos vetores b para a mesma matriz A.
Fatorar custa aproximadamente 2n³/3 operações clássicas; resolver cada novo b custa O(n²).
Esses custos não incluem a gravação didática dos passos.

## 3 · Cholesky e LDLᵀ: simetria e positividade

### Aproveitar simetria e positividade

Cholesky escreve $A=LL^T$ quando A é real, simétrica e positiva definida.
Precisamos guardar apenas um fator triangular. Positiva definida significa
$z^TAz>0$ para **todo** vetor não nulo, não apenas que a diagonal tenha entradas positivas.

Exemplo: $A=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ é simétrica e
$z^TAz=(2z_1+z_2)^2+2z_2^2>0$ para z≠0.

### Exemplo completo de construção

Suponha $L=\begin{bmatrix}l_{11}&0\\l_{21}&l_{22}\end{bmatrix}$.
Igualar LLᵀ a A determina cada entrada:

1. l₁₁²=4, então l₁₁=2, escolhendo a raiz positiva.
2. l₂₁l₁₁=2, então l₂₁=1.
3. l₂₁²+l₂₂²=3, então l₂₂=√2.

$$
L=\begin{bmatrix}2&0\\1&\sqrt2\end{bmatrix},\qquad
LL^T=\begin{bmatrix}4&2\\2&3\end{bmatrix}.
$$

Em uma matriz maior, subtraímos as contribuições das colunas já construídas:

$$
l_{jj}=\sqrt{a_{jj}-\sum_{k<j}l_{jk}^2},\qquad
l_{ij}=\frac{a_{ij}-\sum_{k<j}l_{ik}l_{jk}}{l_{jj}}.
$$

O valor dentro da raiz é um pivô de Schur. Se ele não for positivo sob o critério numérico,
esta versão interrompe e identifica a etapa. Matrizes simétricas como
$\begin{bmatrix}1&2\\2&1\end{bmatrix}$ têm diagonal positiva, mas são indefinidas:
com z=(1,−1), o resultado zᵀAz é −2, e Cholesky SPD não se aplica.

### Exemplo de solução

Para b=(8,8), primeiro resolvemos Ly=b: y₁=4 e y₂=2√2.
Depois resolvemos Lᵀx=y: x₂=2 e x₁=1.
Verificação: A(1,2)ᵀ=(8,8)ᵀ.

### Formulação sem raízes: LDLᵀ

A mesma matriz pode ser escrita como

$$
A=\begin{bmatrix}1&0\\1/2&1\end{bmatrix}
\begin{bmatrix}4&0\\0&2\end{bmatrix}
\begin{bmatrix}1&1/2\\0&1\end{bmatrix}=LDL^T.
$$

Agora L tem diagonal unitária e D guarda os pivôs; não calculamos raízes.
Resolver passa por Ly=b, Dz=y e Lᵀx=z. Neste painel, **LDLᵀ é a versão SPD**
de Cholesky. Existem outras variantes LDLᵀ para matrizes indefinidas, com hipóteses
e pivotamentos diferentes; uma falha aqui não nega essas outras fatorações.

**Uso prático:** sistemas de energia, otimização quadrática, matrizes de covariância positivas
definidas e sistemas simétricos adequados. Uma gramiana com colunas dependentes é apenas
semidefinida e não satisfaz a hipótese SPD. O custo clássico é aproximadamente n³/3,
cerca de metade da LU densa, sem contar o rastreamento.

## 4 · QR: projeções, ortogonalidade e exemplo resolvido

### Como interpretar a decomposição QR

**Base didática:** Thelmo de Araujo, *Fundamentos de Análise de Dados — Aula 10*,
pp. 5–27 (ortogonalidade, projeções, Gram–Schmidt e QR) e pp. 28–32 (regressão).
A matriz do exemplo da aula está disponível no seletor de exemplos.

**1. O que queremos construir?** Escrevemos $A=[a_1\;\cdots\;a_n]=QR$.
As colunas de Q têm norma 1 e produto interno zero entre si.
R guarda os coeficientes que permitem reconstruir cada coluna original.

**2. Como retirar uma projeção?** Para um vetor unitário $q_i$, o número
$r_{ij}=\langle q_i,a_j\rangle=q_i^Ta_j$ mede a componente de $a_j$ na direção de $q_i$.
A projeção é o **vetor** $r_{ij}q_i$. Subtraí-la deixa apenas a componente perpendicular.
Não confunda o coeficiente escalar com o vetor projetado.

$$
w_j=a_j-\sum_{i<j}r_{ij}q_i,\qquad r_{jj}=\|w_j\|_2,\qquad q_j=w_j/r_{jj}.
$$

O primeiro vetor é $q_1=a_1/\|a_1\|_2$. Nos próximos, removemos as direções já construídas.
No clássico, todos os produtos internos usam $a_j$ original; no modificado, cada produto usa
o residual atualizado. Em aritmética exata, os dois processos coincidem; em ponto flutuante,
o modificado costuma preservar melhor a ortogonalidade.

**3. Por que R é triangular superior?** A coluna $a_j$ usa apenas $q_1,\ldots,q_j$:

$$
a_j=r_{1j}q_1+\cdots+r_{jj}q_j.
$$

Logo, os coeficientes abaixo da diagonal são zero. Se $w_j=0$, aquela coluna não acrescenta
uma direção independente; a variante Gram–Schmidt reduzida deste projeto não pode dividir
por sua norma. Householder/Givens podem continuar uma QR com posto deficiente.

**4. Dimensões para esta entrada:**

| Matriz | Dimensão |
|---|---|
| A | 3 × 3 |
| Q nesta variante | 3 × 3 |
| R nesta variante | 3 × 3 |

$Q^TQ=I$ para as colunas completas da Q final. Se Q for retangular alta,
**ela não tem inversa**: $QQ^T$ é a projeção sobre o espaço de suas colunas,
e em geral $QQ^T\ne I$. Apenas uma Q quadrada ortogonal satisfaz $Q^{-1}=Q^T$.

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

### Exemplo simples de QR resolvido

$$
A=\begin{bmatrix}1&1\\1&0\end{bmatrix},\quad
a_1=\begin{bmatrix}1\\1\end{bmatrix},\quad a_2=\begin{bmatrix}1\\0\end{bmatrix}.
$$

1. A primeira norma é √2, então q₁=(1,1)/√2 e r₁₁=√2.
2. O coeficiente r₁₂=q₁ᵀa₂ é 1/√2.
3. A projeção r₁₂q₁ é (1/2,1/2). O residual w₂=a₂−r₁₂q₁ é (1/2,−1/2).
4. A norma de w₂ é 1/√2; normalizar fornece q₂=(1,−1)/√2.

$$
Q=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},\qquad
R=\begin{bmatrix}\sqrt2&1/\sqrt2\\0&1/\sqrt2\end{bmatrix}.
$$

QᵀQ=I e QR=A. Para b=(3,1), temos Qᵀb=(2√2,√2).
Resolver Rx=Qᵀb de baixo para cima dá x₂=2 e x₁=1.

### Comparar os quatro algoritmos

| Algoritmo | Operação principal | Interpretação / cuidado |
|---|---|---|
| Gram–Schmidt clássico | Produtos com as colunas originais | Construção geométrica simples; pode perder ortogonalidade |
| Gram–Schmidt modificado | Produtos com o residual atualizado | Mesma ideia; costuma ser melhor numericamente |
| Householder | Refletir um trecho de coluna | Zera vários elementos de uma vez; boa estabilidade |
| Givens | Girar duas linhas | Zera uma entrada; útil em estruturas esparsas e atualizações |

Householder usa H=I−2vvᵀ, com v unitário. Givens usa
$G=\begin{bmatrix}c&s\\-s&c\end{bmatrix}$, com c²+s²=1.
Ambas preservam norma e podem produzir QR mesmo quando Gram–Schmidt reduzido
encontra uma coluna dependente. Sinais diferentes nas colunas de Q e nas linhas de R
podem representar a mesma A; não compare fatores apenas pela aparência.

**Uso prático:** ajustar uma reta, estimar parâmetros a partir de muitas medições
e resolver mínimos quadrados sem formar explicitamente as equações normais.
O exemplo de ajuste de reta está na seção de aplicações e no seletor do painel.

## 5 · SVD: direções, valores singulares e norma mínima

### Como interpretar a decomposição SVD

**Base didática:** Thelmo de Araujo, *Fundamentos de Análise de Dados — Aula 18*,
pp. 4–15 (construção e nomenclatura), pp. 16–28 (dimensões e SVD completa)
e pp. 29–31 (redução de dimensionalidade). Usamos A onde a aula escreve X.

**1. Por que olhar para $A^TA$?** Essa matriz é simétrica e positiva semidefinida:
$z^TA^TAz=\|Az\|_2^2\ge0$. Seus autovalores são não negativos, e seus autovetores
podem ser escolhidos ortonormais. Chamamos esses vetores de $v_i$.

$$
A^TAv_i=\lambda_i v_i,\qquad \sigma_i=\sqrt{\lambda_i}.
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

**4. SVD reduzida e completa para esta entrada:** $k=\min(m,n)=3$.

| Fator | Reduzida, calculada pelo painel | Completa, explicada na aula |
|---|---|---|
| U | 3 × 3 | 3 × 3 |
| Σ | 3 × 3 | 3 × 3 |
| Vᵀ | 3 × 3 | 3 × 3 |

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
assim $\lambda_i=s^2\mu_i$ e $\sigma_i=s\sqrt{\mu_i}$.
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
A=\sum_{i=1}^k\sigma_i u_i v_i^T,\qquad
A_r=\sum_{i=1}^r\sigma_i u_i v_i^T.
$$

Cada termo tem posto no máximo 1. Manter os primeiros r termos produz a melhor
aproximação de posto no máximo r nas normas 2 e Frobenius; a energia descartada
é $\sum_{i>r}\sigma_i^2$. Essa relação é explicada aqui; o painel mostra a soma dos termos
na reconstrução, sem alterar automaticamente a matriz informada.

**7. Como resolver um sistema singular?** Usamos
$A^+=V\Sigma^+U^T$ e $x=A^+b$. Invertendo apenas valores singulares acima
do corte numérico, obtemos mínimos quadrados de norma mínima sob esse critério.

**Sobre dados e PCA:** a aula parte de dados centralizados. A SVD também vale para
qualquer matriz real, sem centralização. O painel preserva A; não remove médias.
Somente com dados centralizados e a normalização de covariância apropriada
se interpretam $\sigma_i^2/(m-1)$ como variâncias amostrais, para $m>1$.

### Exemplo simples: intensidades de duas direções

$$
A=\begin{bmatrix}3&0\\0&1\end{bmatrix},\qquad
A^TA=\begin{bmatrix}9&0\\0&1\end{bmatrix}.
$$

Os autovalores da gramiana são 9 e 1. Portanto os valores singulares são 3 e 1,
e podemos escolher U=V=I e Σ=A. A primeira direção é alongada três vezes;
a segunda mantém seu comprimento. São valores singulares de A, não 9 e 1.

Para b=(6,2), a pseudoinversa é diag(1/3,1) e x=(2,2).
Se retivermos apenas a primeira direção, A₁=diag(3,0). O erro Frobenius é 1,
e a fração de energia preservada é 3²/(3²+1²)=90%.
Truncar direções é uma aproximação deliberada, diferente de reconstruir A com todos os termos.

### Exemplo com autovetores: contas completas em 2×2

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix},\quad
G=A^TA=\begin{bmatrix}2^2+1^2&2\cdot1+1\cdot2\\1\cdot2+2\cdot1&1^2+2^2\end{bmatrix}
=\begin{bmatrix}5&4\\4&5\end{bmatrix}.
$$

Os autovalores resolvem det(G−λI)=0:

$$
(5-\lambda)^2-4\cdot4=\lambda^2-10\lambda+9
=(\lambda-9)(\lambda-1)=0.
$$

Para λ₁=9, −4v₁+4v₂=0 dá v₁=v₂. Escolhemos (1,1) e dividimos por √2.
Para λ₂=1, 4v₁+4v₂=0 dá v₂=−v₁. Escolhemos (1,−1) e dividimos por √2.

$$
V=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},\quad
\Sigma=\begin{bmatrix}\sqrt9&0\\0&\sqrt1\end{bmatrix}
=\begin{bmatrix}3&0\\0&1\end{bmatrix}.
$$

Os produtos são Av₁=(3,3)/√2 e Av₂=(1,−1)/√2.
Dividir por σ₁=3 e σ₂=1 dá U=V neste exemplo específico; isso não vale para qualquer A.

$$
U\Sigma V^T=\frac12\begin{bmatrix}3&1\\3&-1\end{bmatrix}
\begin{bmatrix}1&1\\1&-1\end{bmatrix}
=\frac12\begin{bmatrix}4&2\\2&4\end{bmatrix}=A.
$$

No seletor, use **SVD • autovetores e fatores simples (2 × 2)**.
Com b=(4,5), a solução é x=(1,2). O programa pode escolher sinais diferentes
para os pares de vetores; os produtos e valores singulares continuam os mesmos.

### Valores nulos e norma mínima

Para $A=\begin{bmatrix}1&1\\2&2\end{bmatrix}$ e b=(3,6), qualquer par
com x₁+x₂=3 resolve o sistema. SVD retorna (1.5,1.5), que minimiza
x₁²+x₂² nessa família. Um valor singular nulo indica uma direção no núcleo,
na qual alterar x não altera Ax.

**Uso prático:** matrizes singulares, identificação do posto, remoção controlada de
direções pouco relevantes, compressão de dados e soluções de norma mínima.
O corte da pseudoinversa afeta quais direções são preservadas; em uma matriz
mal condicionada, inverter valores singulares muito pequenos pode amplificar erros.

## 6 · Aplicações práticas e escolha do método

### 1. Reutilizar fatores para vários resultados conhecidos

Em $A=\begin{bmatrix}2&1\\4&3\end{bmatrix}$, fatoramos uma vez.
Com b=(5,11), a solução é (2,1); com b=(4,10), é (1,2).
L e U são os mesmos. Isso ocorre em simulações que mantêm a estrutura física
e alteram cargas, forças, preços ou medições.

### 2. Ajustar uma reta a medições ruidosas

Considere t=(0,1,2,3) e y=(1,2,2,4). Procuramos $y\approx c_0+c_1t$.
Há quatro medições e dois parâmetros:

$$
A=\begin{bmatrix}1&0\\1&1\\1&2\\1&3\end{bmatrix},\qquad
b=\begin{bmatrix}1\\2\\2\\4\end{bmatrix},\qquad x=\begin{bmatrix}c_0\\c_1\end{bmatrix}.
$$

QR ou SVD produzem c₀=0.9 e c₁=0.9. As previsões são (0.9,1.8,2.7,3.6),
e Ax−b=(−0.1,−0.2,0.7,−0.4). A soma dos quadrados é 0.7,
e a norma do residual é √0.7. A reta não passa por todos os pontos,
mas minimiza o erro total quadrático. Um residual não nulo é esperado aqui.

### 3. Distinguir sistema consistente e inconsistente

Na matriz $A=\begin{bmatrix}1&1\\2&2\end{bmatrix}$, b=(3,6) é consistente,
com infinitas soluções. Se b=(3,7), não existe x que satisfaça ambas as equações.
SVD produz x=(1.7,1.7), Ax=(3.4,6.8) e residual (0.4,−0.2).
Sua norma é √0.2. Essa é uma solução de mínimos quadrados de norma mínima,
e não uma solução exata do sistema incompatível.

### 4. Mais incógnitas que equações

$$
A=\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix},\qquad b=(1,1)^T.
$$

Há infinitas soluções x₁+x₃=1 e x₂+x₃=1. SVD escolhe
(1/3,1/3,2/3), a de menor norma euclidiana. A fatoração QR também existe,
mas a rotina de resolução QR deste projeto exige m≥n e pivôs não nulos;
para este caso, selecione SVD.

### 5. Escolher um método com base no problema

| Situação | Escolha usual | O que verificar |
|---|---|---|
| Quadrada geral, muitos b | LU com pivotamento | Pivôs, singularidade e residual |
| Simétrica positiva definida | Cholesky / LDLᵀ SPD | Simetria e positividade |
| Alta com colunas independentes | QR Householder | Posto e norma do residual |
| Singular, larga ou posto incerto | SVD | Corte dos valores singulares e norma mínima |
| Apenas visualizar ortogonalização | Gram–Schmidt | Produtos internos e perda de ortogonalidade |

Essas escolhas são orientações usuais. A tabela do painel verifica cada variante
para a matriz informada e explica seus bloqueios sob a tolerância configurada.

## 7 · Precisão, condicionamento, verificação e métodos iterativos

### Ponto flutuante e tolerância

O painel usa números float64. Números como 1/3 são aproximados, e operações
como somar três vezes uma aproximação podem deixar um pequeno erro.
Por isso comparamos com uma tolerância relativa, em vez de exigir igualdade decimal exata.

O padrão é 10⁻¹². No posto e na pseudoinversa, preservamos σᵢ>rtol×σ₁.
Nos pivôs, usamos um limiar proporcional à norma infinito da matriz ou do fator ativo.
Esses testes têm propósitos e escalas diferentes; perto do limite podem dar diagnósticos distintos.
Se A=diag(1,10⁻¹⁴), ela é invertível em aritmética exata, mas o corte padrão
classifica a segunda direção como nula numericamente. Alterar a tolerância muda essa decisão.

### Condicionamento: sensibilidade do problema

Em posto completo, $\kappa_2(A)=\sigma_{max}/\sigma_{min}$ mede a sensibilidade.
Não é um erro do algoritmo. Em A=diag(1,10⁻⁸), κ₂=10⁸. Considere b=(1,10⁻⁸).
Se a segunda componente de b muda de 10⁻⁸ para 2×10⁻⁸, a segunda componente
de x muda de 1 para 2. Uma alteração pequena em relação à escala total de b
pode causar grande mudança na solução.

Para matrizes de posto deficiente, o painel mostra ∞ sob seu critério numérico,
inclusive em matrizes retangulares. Multiplicar toda A por um escalar não nulo
não melhora seu condicionamento. Pivotamento controla a eliminação; não remove
a sensibilidade intrínseca do problema.

### Reconstrução, ortogonalidade e residual

São verificações diferentes:

| Verificação | Pergunta respondida |
|---|---|
| $\|A-\widehat A\|_F/\|A\|_F$ | Os fatores reconstruíram A? |
| $\|Q^TQ-I\|_F$ | As colunas de Q ficaram ortonormais? |
| $\|Ax-b\|_2$ | A solução satisfaz as equações? |
| $\|Ax-b\|_2/(\|A\|_F\|x\|_2+\|b\|_2)$ | Qual o residual em relação à escala dos dados? |

Em mínimos quadrados, o residual pode ser não nulo e ainda assim a solução ser correta.
Um residual pequeno não garante erro pequeno em x se A for mal condicionada.
Em A nula, as razões que teriam denominador zero usam uma proteção numérica no programa.

### Estabilidade: comportamento do algoritmo

Um algoritmo estável limita a propagação de arredondamento. Usamos pivotamento na LU
para evitar divisões desnecessárias por pivôs pequenos; Householder/Givens preservam
normas; SVD permite examinar as direções pequenas. Gram–Schmidt clássico pode perder
ortogonalidade em colunas quase dependentes. Calcular inversas explicitamente para
resolver Ax=b costuma adicionar trabalho e deve ser evitado aqui: usamos fatores e substituições.

### Métodos diretos e iterativos

LU, Cholesky e QR são usados como métodos diretos de solução: fatoração seguida de
substituições ou projeções. Métodos iterativos atualizam aproximações de x e usam critérios
de parada; são úteis em grandes problemas, especialmente esparsos, quando convergem.

Exemplo de Jacobi para $4x_1+x_2=5$, $x_1+3x_2=4$:
$x_1^{(k+1)}=(5-x_2^{(k)})/4$ e $x_2^{(k+1)}=(4-x_1^{(k)})/3$.
Partindo de (0,0), a primeira atualização é (1.25,1.3333); a segunda,
aproximadamente (0.9167,0.9167). Neste exemplo com dominância diagonal estrita,
as aproximações convergem para (1,1). Isso não é uma garantia para qualquer matriz.

O **Jacobi para Ax=b** explicado acima é apenas um exemplo teórico de método
iterativo; ele não é executado pelo painel. A SVD atual usa os autovalores e
autovetores da gramiana, conforme a Aula 18.
