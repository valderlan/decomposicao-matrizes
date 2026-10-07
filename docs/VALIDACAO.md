# Validação da entrega

Data: 07/10/2026. Ambiente de verificação: Linux, Python 3.12.14, dependências fixadas em `uv.lock`.

## Resultado

- `uv sync --locked --group dev`: ambiente instalado a partir do lockfile.
- `uv run pytest -q`: **124 testes aprovados** após a substituição da SVD e a ampliação das contas.
- `uv run ruff check .`: nenhuma ocorrência.
- `uv run ruff format .`: código formatado.
- Inicialização real de `uv run streamlit run app.py`: endpoint de saúde HTTP 200 (`ok`) e página principal HTTP 200.

## Cobertura

Testes matemáticos verificam reconstrução e estrutura dos fatores, diagonal das variantes LU, permutações de linhas/colunas, ortogonalidade, valores singulares, substituições triangulares, mínimos quadrados e as quatro identidades de Moore–Penrose para um caso de posto deficiente. Os resultados são comparados com rotinas independentes do NumPy.

Os casos incluem matrizes de até 10 × 10, altas, largas, nulas, singulares, indefinidas, com pivô inicial zero, mal condicionadas e em escalas diferentes. Também se verificam parsing, rejeição de valores inválidos e independência das cópias dos estados intermediários.

Testes `AppTest` verificam diagnóstico, escolha, fatoração, resolução, erro de resolvedor preservando fatores, modo grade, alternância texto/grade, preservação de b e invalidação de resultados após mudar a entrada. Também conferem os botões Primeiro/Anterior/Próximo/Último, desativação nos extremos, independência entre navegação da decomposição e da solução, reinício na nova execução e navegação em SVD de matriz larga.

Os dezesseis exemplos distribuídos foram executados com SVD. O exemplo SPD retornou x=(1,2,3); o ajuste de reta retornou (0.9,0.9); o sistema subdeterminado retornou (1/3,1/3,2/3). A matriz singular inconsistente produziu resíduo não nulo, como esperado.

O exemplo da Aula 10, pp. 20–27, é conferido contra os fatores Q e R do desenvolvimento da aula, tanto no Gram–Schmidt clássico quanto no modificado. Os estados das projeções verificam antes−projeção=depois. As novas verificações didáticas de SVD conferem Y=AV=UΣ, YᵀY≈Σ² e a soma dos termos σᵢuᵢvᵢᵀ para matrizes altas, largas e quadradas. A SVD completa é explicada no guia; o painel calcula fatores reduzidos.

## Limites da verificação

Os testes da interface são simulações com Streamlit AppTest; a inicialização HTTP foi verificada separadamente. Não foi realizada inspeção visual em navegador real nem execução nativa em Windows/macOS. O projeto utiliza comandos e caminhos portáveis; nesses sistemas, a equipe deve repetir os comandos do README.

Estes resultados verificam a implementação nos casos cobertos. Aritmética de ponto flutuante, tolerâncias e matrizes mal condicionadas continuam exigindo interpretação numérica.


## Atualização de LU e do guia teórico

O exemplo da Seção 1.3 é comparado com L/U de Doolittle e Crout, incluindo b=(1,-1,-7) e x=(2,1,-1). Para cada eliminação registrada, os testes conferem E·U_antes≈U e E·E_inversa≈I. Sem trocas, o produto ordenado das inversas coincide com a L de Doolittle. Casos com pivotamento parcial/total verificam as operações elementares e a reconstrução após permutações.

Os exemplos simples 2×2 de LU, Cholesky, QR e SVD são executados e comparados com as soluções escritas no guia. AppTest confirma as sete seções do guia, a presença da referência de LU, o carregamento do exemplo do material e a navegação após a ampliação.

## Identidades por método e renderização matemática

Os testes verificam A=LU sem pivotamento, PA=LU no parcial e PAC=LU no total, tanto na identidade inicial quanto nos sistemas triangulares e nos fatores retornados. P aparece apenas nas variantes com pivotamento de linhas; C aparece apenas no total. A solução x=(1,2,3) do novo exemplo com fatores inteiros é verificada nas quatro variantes LU.

AppTest confirma que a fórmula de fundamentos, antes apresentada como “undefined”, é enviada como elemento LaTeX separado, que os parágrafos e tabelas permanecem em Markdown, e que o exemplo pode ser selecionado, resolvido e navegado. Foram renderizadas sem erro 334 fórmulas do guia e dos passos de todas as famílias usando `renderToString` com `throwOnError=true` e `strict=error` no próprio KaTeX 0.16.47 distribuído com o Streamlit instalado. Esta verificação não substitui uma inspeção visual em navegador real.

## SVD espectral e contas detalhadas

Os novos testes cobrem a construção pela gramiana em matrizes quadradas, altas, largas, singulares e nulas, verificando GₛV≈VΛ, VᵀV≈I e AV≈UΣ. Uma restrição impede chamadas a numpy.linalg.svd durante essa construção. No exemplo A=[[2,1],[1,2]], verificam-se os coeficientes [1,-2.5,0.5625] do polinômio da gramiana escalada, os valores singulares (3,1) e x=(1,2). Há um caso com condicionamento alto que verifica o aviso de precisão.

Householder/Givens são verificados após cada transformação registrada: QR≈A e QᵀQ≈I. AppTest confere o novo nome da SVD, o seletor do exemplo não diagonal, as etapas espectrais, a seção Contas deste passo e a presença de cálculos no JSON.

A validação do KaTeX foi ampliada para **10.862 expressões**, abrangendo o guia e todos os métodos aplicáveis aos dezesseis exemplos, incluindo fórmulas gerais e contas numéricas dos fatores e das soluções. Nenhuma expressão produziu erro. Esse teste verifica sintaxe/renderização das expressões; os testes numéricos verificam as identidades matemáticas. Não houve inspeção visual em navegador real.
