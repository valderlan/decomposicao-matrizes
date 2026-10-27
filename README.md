# Laboratório de decomposições matriciais

Projeto de Álgebra Linear Numérica baseado na proposta **Métodos de Decomposição para Sistemas de Equações Lineares**. Python + uv + Streamlit; interface e explicações em português. Uso local no navegador, em Linux, Windows ou macOS.

## Iniciar

Instale Python 3.11 ou mais recente e [uv](https://docs.astral.sh/uv/getting-started/installation/). Na pasta que contém este README:

```bash
uv sync --locked
uv run streamlit run app.py
```

Abra **http://localhost:8501**. No Windows, execute os mesmos comandos pelo PowerShell; não precisa ativar `.venv`. Para encerrar, pressione Ctrl+C no terminal.


## O que está implementado

1. Entrada por texto (espaços, vírgulas, frações ou JSON) ou grade editável; matrizes reais de 1 × 1 até 10 × 10.
2. Diagnóstico de dimensões, simetria, posto numérico, positividade, normas e condicionamento.
3. Tabela com todas as variantes, disponibilidade e motivo de falha. O diagnóstico usa a mesma implementação que executa a fatoração.
4. LU Doolittle, Crout por reescalonamento equivalente, pivotamento parcial e total.
5. Cholesky LLᵀ e versão sem raízes LDLᵀ, ambas restritas a matrizes simétricas positivas definidas.
6. QR por Gram–Schmidt clássico/modificado, Householder e Givens.
7. SVD reduzida pelos autovalores/autovetores de AᵀA, incluindo matrizes largas, singulares e nulas. Os pares espectrais são calculados com `numpy.linalg.eigh`; os demais passos de construção e verificação são explícitos.
8. Passos reais do algoritmo com foco em evolução de estado: objetivo simples, operação aplicada, valores intermediários, fórmulas, fatores finais e erro de reconstrução. Cada operação mostra matrizes **Antes** e **Depois** com cópias independentes, além de comparação lado a lado quando disponível. Navegação por botões **Primeiro**, **Anterior**, **Próximo** e **Último**, com indicação do passo atual, tanto na decomposição quanto na solução.
9. Vetor b opcional: substituições sucessivas/regressivas, mínimos quadrados por QR e pseudoinversa por SVD com rastreamento.
10. Exemplos, comparação de complexidade, exportação JSON e testes matemáticos e de interface. O exemplo QR da Aula 10 é distribuído junto dos demais casos.
11. Guias e explicações ampliados com base nas Aulas 10 e 18 de *Fundamentos de Análise de Dados*, de Thelmo de Araujo: projeções, Gram–Schmidt, formação de R, mínimos quadrados, Y=AV, valores/vetores singulares, completamento de base, dimensões da SVD reduzida/completa e reconstrução por termos de posto 1.
12. LU ampliada com base em *Álgebra Linear — Seção 1.3*: matrizes elementares E/E⁻¹, cálculo dos multiplicadores, atualização de cada entrada da linha, registro em L, ordem do produto das inversas, Crout e permutações.
13. Guia teórico em **sete seções expansíveis**, com fundamentos, LU, Cholesky/LDLᵀ, QR, SVD, aplicações e precisão. Exemplos 2×2 resolvidos, caso LU 3×3 do material, ajuste de reta, norma mínima e comparação com métodos iterativos. Os exemplos principais também podem ser carregados no painel. Uma cópia do guia está em `docs/GUIA_TEORICO.md`.
14. Identidades, fatores e solução adequados à variante: sem pivotamento, A=LU e apenas L/U; parcial, PA=LU e L/U/P; total, PAC=LU e L/U/P/C. A exportação JSON segue a mesma seleção de fatores. Blocos matemáticos do guia são renderizados com `st.latex`.
15. Seção **Contas deste passo**: substituição dos valores em multiplicadores, correções de Cholesky/LDLᵀ, projeções e normas de QR, refletores/rotações, gramiana, polinômio característico, autovetores, valores singulares e normalização. Produtos dos fatores, sistemas triangulares e resíduos também mostram contas por componente. O JSON inclui essas expressões no campo `calculations` de cada passo, e também exporta os destaques de células, a operação registrada e a ordem de exibição dos estados.
16. Legenda visual por passo: pivô, entrada-alvo, trecho ativo e entradas atualizadas são destacados com cores consistentes na matriz. Isso facilita acompanhar por que cada operação é necessária e qual parte do fator foi alterada.

Para começar com resultados inteiros, selecione **LU • fatores inteiros (3 × 3)** e
**LU • Doolittle sem pivotamento**. A matriz é `[[2,1,1],[4,5,3],[2,7,7]]` e
b=(7,23,37). Os multiplicadores são 2, 1 e 2; a solução é x=(1,2,3).

Digite, por exemplo:

```text
4 12 -16
12 37 -43
-16 -43 98
```

Use **ponto decimal**: `0.5`; a vírgula separa elementos. `1/3` é convertido para ponto flutuante, não aritmética racional exata. `;` também separa linhas. Para resolver Ax=b, habilite a opção e informe uma linha ou coluna com m valores.

## Uma distinção essencial

Toda matriz real finita admite SVD. QR por transformações ortogonais também pode fatorar matrizes com posto deficiente. Portanto, uma matriz singular **não significa que nenhuma decomposição existe**. A interface explica as restrições de cada **algoritmo implementado** e não apresenta uma falha numérica como prova de inexistência matemática.

LU pode produzir fatores de uma matriz singular, embora a substituição triangular não forneça solução única. QR Householder/Givens é válido para matrizes largas; a rotina de resolução por QR aqui exige m ≥ n e pivôs não nulos. Para casos gerais, a SVD oferece pseudoinversa; a variante espectral deste painel exige examinar seus avisos de precisão.

LDLᵀ também existe em versões para matrizes indefinidas, com outras hipóteses e pivotamento. A variante deste projeto é a **formulação sem raízes de Cholesky para positivas definidas**, seguindo a proposta. Não se afirma que indefinidas jamais admitem LDLᵀ.

## Organização

```text
decomposicao-matrizes/
├── app.py                    # dashboard e fluxo de interação
├── pyproject.toml            # dependências e configuração
├── uv.lock                   # versões reproduzíveis
├── .streamlit/config.toml    # tema
├── src/matrixlab/
│   ├── validation.py         # parser, limites e tolerâncias
│   ├── analysis.py           # propriedades e métodos aplicáveis
│   ├── decompositions.py     # algoritmos e rastreamento
│   ├── solvers.py            # Ax=b, mínimos quadrados e pseudoinversa
│   ├── models.py             # estruturas de passos e resultados
│   ├── arithmetic.py         # contas numéricas exibidas nas fórmulas
│   ├── examples.py           # casos para apresentação
│   ├── lessons.py            # construção conceitual de QR e SVD baseada nas aulas
│   └── theory.py             # guia completo em seções com exemplos resolvidos
├── tests/                    # verificação independente + interface
├── docs/PROJETO.md           # arquitetura, critérios e roteiro acadêmico
└── examples/                 # matrizes de exemplo em JSON
```

## Verificar

```bash
uv sync --locked --group dev
uv run pytest -q
uv run ruff check .
```

Os testes conferem reconstrução, estrutura triangular, ortogonalidade, permutações, soluções e pseudoinversas contra NumPy. Há casos retangulares, singulares, pivôs zero, mau condicionamento, escalas distintas e entrada inválida. Os testes de Streamlit verificam o fluxo do painel sem depender de um navegador instalado.

## API Python

```python
import numpy as np
from matrixlab import decompose, inspect_matrix, solve

A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
b = np.array([-20.0, -43.0, 192.0])
diagnostico = inspect_matrix(A)
fatoracao = decompose(A, "cholesky")
solucao = solve(A, b, fatoracao)
print(solucao.x)  # [1., 2., 3.]
for etapa in fatoracao.steps:
    print(etapa.title, etapa.explanation)
```

## Precisão e escopo

- Aritmética `float64`. Tolerância relativa padrão: 1e-12, ajustável entre 1e-15 e 1e-4. Os critérios locais de pivô usam a norma infinito; o posto usa o maior valor singular. Eles não são testes simbólicos exatos.
- Valores não nulos entre 1e-100 e 1e100 em módulo. Para magnitudes maiores/menores, reescale os dados. Limites mantêm a ferramenta didática responsiva e reduzem overflow.
- SVD segue a Aula 18: gramiana escalada BᵀB, pares espectrais por `numpy.linalg.eigh`, reavaliação μᵢ=‖Bvᵢ‖², σᵢ=s√μᵢ e uᵢ=Avᵢ/σᵢ. O polinômio característico é calculado para explicação; o resolvedor simétrico não usa seus coeficientes arredondados. As iterações internas da biblioteca não são inventadas no registro.
- A construção pela gramiana pode perder precisão nas direções pequenas. Escala global não corrige condicionamento. Direções com norma até 10·eps·max(m,n)·maxᵢ‖Bvᵢ‖ são tratadas como nulas; o corte da pseudoinversa pelo usuário é uma etapa adicional. Reavaliar μ não recupera autovetores imprecisos. O painel mostra avisos, erro de reconstrução e ortogonalidade; para uso científico difícil, prefira uma SVD direta. NumPy SVD é usado apenas no diagnóstico e nos testes independentes.
- Para SVD com valores tratados como nulos, completamos a base ortogonal. A pseudoinversa aplica corte relativo: com valores pequenos descartados, trata-se de uma pseudoinversa numérica truncada.
- Cholesky pode simetrizar uma assimetria dentro da tolerância; o passo informa isso e o erro é calculado contra a matriz original.
- Resíduo pequeno não garante pequeno erro em x para matrizes mal condicionadas. Os algoritmos didáticos não substituem LAPACK em uso científico de grande escala.
- O projeto é local e não exige banco de dados, conta, API ou servidor externo. Exportações são feitas pelo navegador.
- Métodos iterativos para Ax=b, cálculo iterativo de autovalores, números complexos e implantação pública estão fora desta versão. O guia teórico contempla comparações com métodos iterativos e a relação entre SVD/autovalores.

## Referências de implementação

- [uv: projetos](https://docs.astral.sh/uv/guides/projects/) e [lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/).
- [Streamlit: execução e interface](https://docs.streamlit.io/) e [testes AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
- [LAPACK Users' Guide](https://www.netlib.org/lapack/lug/) e [problemas de mínimos quadrados](https://www.netlib.org/lapack/lug/node27.html).
- [NumPy: autovalores e autovetores simétricos](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html).
- Thelmo de Araujo, *Fundamentos de Análise de Dados — Aula 10*: QR, projeções e Gram–Schmidt. Material fornecido pelo usuário;
- Thelmo de Araujo, *Fundamentos de Análise de Dados — Aula 18*: SVD reduzida/completa e redução de dimensionalidade. Material fornecido pelo usuário; 
- Golub e Van Loan, *Matrix Computations*, 4ª edição; Trefethen e Bau, *Numerical Linear Algebra*.

## Licença

Este projeto é distribuído sob a **Licença MIT**, uma licença open source permissiva de livre uso.
Você pode usar, copiar, modificar, mesclar, publicar, distribuir e sublicenciar o software,
inclusive para fins comerciais, desde que mantenha o aviso de copyright e a licença.

