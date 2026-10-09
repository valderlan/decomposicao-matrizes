# Organização do trabalho

## Objetivo

Desenvolver uma ferramenta educacional que receba A, analise suas propriedades, indique os algoritmos de decomposição aplicáveis, explique os bloqueios e apresente a fatoração escolhida com operações intermediárias. Com b opcional, resolver Ax=b ou produzir mínimos quadrados/norma mínima quando apropriado.

## Arquitetura

O painel `app.py` captura e valida a entrada; `analysis.py` mede propriedades e verifica os métodos; `decompositions.py` executa o método; `solvers.py` usa os fatores para resolver o sistema. `models.py` guarda os estados como cópias, evitando que alterações posteriores modifiquem os passos anteriores. A interface renderiza o resultado sem recalcular fatores ao navegar pelos passos.

Nenhum serviço externo ou banco de dados é necessário. Sessões do Streamlit mantêm entrada, diagnóstico e resultado em memória. Uma assinatura da matriz, tolerância, método e b impede exibir um resultado antigo após mudar os dados. O relatório JSON permite salvar entrada, critérios, fatores, etapas e resíduos.

## Critérios por método

| Método implementado | Condição verificada | Fatoração mostrada | Restrição da resolução |
|---|---|---|---|
| Doolittle sem pivotamento | Quadrada; pivôs usados em divisão não nulos. Um pivô nulo com coluna já eliminada pode permanecer | A=LU; diag(L)=1 | Pivôs triangulares não nulos |
| Crout equivalente | Quadrada; pivôs não nulos para reescalonar | A=LU; diag(U)=1 | Pivôs triangulares não nulos |
| LU parcial | Quadrada; troca de linhas escolhe maior módulo na coluna ativa | PA=LU | Singularidade pode impedir solução única |
| LU total | Quadrada; troca de linhas e colunas escolhe maior módulo no bloco ativo | PAC=LU | Recuperar x=Cz; singularidade pode impedir solução única |
| Cholesky | Simétrica dentro da tolerância; pivôs de Schur positivos | A=LLᵀ | SPD numérica |
| LDLᵀ sem raízes SPD | Mesma hipótese de Cholesky | A=LDLᵀ; diag(L)=1 | Não implementa LDLᵀ indefinida com pivotamento |
| Gram–Schmidt clássico/modificado | m≥n; vetor residual de cada coluna não nulo | A=QR reduzido | Colunas numericamente independentes |
| Householder/Givens | Matriz real válida, inclusive larga ou singular | A=QR completo | Rotina de solução implementada: m≥n e R sem pivôs nulos |
| SVD espectral | Matriz real válida; resolvedor simétrico precisa convergir | A=UΣVᵀ reduzida | Pseudoinversa numérica; corte relativo nos σ |

O teste positivo definido pelo diagnóstico usa autovalores; Cholesky verifica pivôs de Schur. Perto do limiar esses critérios numéricos podem diferir: a tabela de aplicabilidade executa o algoritmo correspondente para decidir sua disponibilidade.

## Fluxo de uso e critérios de aceitação

1. Informar matriz válida, de ordem limitada; entrada irregular, complexa, infinita ou NaN deve produzir erro compreensível.
2. Analisar: mostrar dimensões, posto, simetria, positividade, normas e condicionamento.
3. Mostrar uma linha para cada variante, com disponibilidade e justificativa. Os métodos bloqueados continuam visíveis na tabela.
4. Selecionar um método disponível e executar. Se a execução falhar numericamente, apresentar a razão.
5. Mostrar fatores finais e identidade correspondente à variante: L/U e A=LU sem pivotamento; L/U/P e PA=LU no parcial; L/U/P/C e PAC=LU no total. Uma permutação pode ser identidade quando o método não precisar efetuar trocas.
6. Navegar pelos passos reais com números, fórmulas e matrizes intermediárias usando botões Primeiro, Anterior, Próximo e Último. Cada operação mostra objetivo, operação aplicada, estados Antes/Depois (com cópias independentes) e legenda de destaques para pivô, entrada-alvo, trecho ativo e atualizações. Os controles da decomposição e da solução são independentes; uma nova execução reinicia ambos no primeiro passo.
7. Verificar a reconstrução. Para Q/U/V, verificar ortogonalidade. Advertir sobre perda de precisão.
8. Com b, apresentar as etapas da solução e o resíduo; diferenciar fatoração de resolução. Se o resolvedor falhar, conservar os fatores e explicar a restrição.
9. Exportar um JSON válido contendo todos os passos, sem NaN ou Infinity não padronizados.

## Escolhas de implementação

- Python e NumPy simplificam operações vetoriais. Os algoritmos didáticos são explícitos, com valores calculados e registrados no mesmo lugar. SVD de NumPy é uma referência independente de diagnóstico/teste.
- Streamlit evita uma segunda aplicação JavaScript e já fornece grade, LaTeX, tabelas, navegação e download. O núcleo não importa Streamlit e pode ser reutilizado em CLI/API futura.
- Crout é obtido por eliminação Doolittle seguida de transferência da diagonal de U para L. O painel informa esta construção equivalente; não a apresenta como um laço direto clássico de Crout.
- Householder/Givens acumulam matrizes ortogonais explícitas para mostrar cada transformação. Isso custa mais que implementações otimizadas que armazenam refletores/rotações compactamente.
- SVD segue a construção espectral da Aula 18: B=A/s, Gₛ=BᵀB, pares espectrais por numpy.linalg.eigh, reavaliação μᵢ=‖Bvᵢ‖², valores singulares e normalização de AV. A etapa da biblioteca é identificada explicitamente; as demais contas são mostradas com os valores usados.
- Um limiar de máquina identifica colunas transformadas numericamente nulas; a tolerância do usuário define o corte adicional da pseudoinversa. A construção pela gramiana pode perder precisão nas direções pequenas, e o painel informa esse limite.

## Roteiro de desenvolvimento e apresentação

Esta entrega contém uma primeira versão funcional completa. Para organizar apresentações ao longo do semestre, a equipe pode trabalhar em quatro blocos:

| Bloco | Trabalho acadêmico | Demonstração sugerida |
|---|---|---|
| 1. Fundamentos e LU | Triangularidade, operações elementares, permutações e estabilidade | Matriz com primeiro pivô zero; comparar LU sem/parcial/total |
| 2. Cholesky e QR | Positividade, ortogonalidade, projeções, reflexões e rotações | Exemplo SPD 3×3 e ajuste de reta 4×2 |
| 3. SVD e sistemas gerais | Valores singulares, posto, pseudoinversa e condicionamento | Singular consistente/inconsistente e subdeterminado |
| 4. Comparação e seminário | Complexidade, validação, limitações numéricas e discussão | Mesmo sistema 3×3 por LU, Cholesky, QR e SVD; conferir x e resíduos |

No relatório escrito, seguir os tópicos da proposta: introdução; fundamentos; LU; Cholesky; QR; SVD; exercícios/comparação; implementação/aplicações; referências. O relatório teórico completo e os slides ainda devem ser redigidos pela equipe; este arquivo organiza o desenvolvimento computacional e a apresentação.

## Casos essenciais de validação

- SPD 3×3: todos os métodos relevantes reconstruem A e retornam a solução de referência.
- Pivô inicial zero: Doolittle/Crout bloqueados; LU parcial/total disponíveis.
- Singular: Householder/Givens/SVD disponíveis; Gram–Schmidt sinaliza dependência.
- Indefinida simétrica: Cholesky e LDLᵀ SPD bloqueados com o pivô identificado.
- Retangular alta: QR/SVD; mínimos quadrados comparados com `numpy.linalg.lstsq`.
- Retangular larga: QR válido; resolvedor QR explica restrição; SVD norma mínima.
- Nula: SVD/QR reconstruem; pseudoinversa nula.
- Escalas distintas e mau condicionamento: tolerância, reconstrução, ortogonalidade e resíduos explícitos.
- Interface: entrada alterada invalida diagnóstico; b alterado invalida solução; método bloqueado não pode ser escolhido.

## Extensões possíveis

Não necessárias para executar esta versão: cálculo iterativo de autovalores com QR; comparação cronometrada sem rastreamento; análise de sensibilidade ao perturbar b; LDLᵀ indefinida com pivotamento; Crout direto clássico; exportação PDF; aritmética exata; números complexos; matrizes esparsas e implantação em servidor.

## Integração das aulas fornecidas

A Aula 10 orienta a explicação geométrica de QR: retirar projeções, normalizar o residual e escrever cada aⱼ como combinação das colunas qᵢ. Cada projeção mostra o produto interno componente a componente, o vetor projetado e os residuais antes/depois. O painel apresenta o exemplo 3×3 das pp. 20–27 e explica por que R é triangular e por que uma Q retangular não tem inversa. A solução por QR inclui a projeção de b e a decomposição da norma do residual.

A Aula 18 orienta a interpretação da SVD por AᵀA: λᵢ=σᵢ², Y=AV, normalização yᵢ/σᵢ, completamento para σ nulo e distinção entre dimensões reduzidas/completas. A gramiana agora é a entrada do resolvedor de autovalores simétrico. O polinômio característico é calculado para explicar a equação espectral; os coeficientes arredondados não são usados pelo resolvedor. Os autovetores são verificados linha por linha antes de construir os fatores. Os fatores obtidos verificam Y=AV=UΣ e YᵀY≈Σ². A reconstrução acumula σᵢuᵢvᵢᵀ, explicando como cada direção contribui para A.

`lessons.py` reúne os guias, com atribuição ao autor e às páginas. As cópias dos materiais fornecidos estão em `docs/materiais/`. A matriz do usuário é preservada, sem centralização automática: a interpretação estatística de variâncias exige dados centralizados e a normalização de covariância apropriada. A SVD completa é explicada, enquanto o cálculo fornecido pelo painel permanece reduzido.

### LU e guia completo por seções

A Seção 1.3 orienta a ligação entre operações de linha e matrizes elementares. Os passos de LU mostram seleção do pivô, cálculo do multiplicador, E e E⁻¹, atualização de cada entrada, armazenamento em L, trocas e interpretação das inversas. No caso sem trocas, o produto das inversas é mostrado na ordem que recupera A. Crout é explicado como transferência dos pivôs de U para L, mantendo o produto.

`theory.py` organiza sete seções: fundamentos, LU, Cholesky/LDLᵀ, QR, SVD, aplicações e precisão. Cada seção inclui definições, fórmulas, exemplos resolvidos e restrições dos métodos. Fundamentos e a família escolhida começam abertos no painel; as demais seções continuam disponíveis para comparação. O guia também está em `docs/GUIA_TEORICO.md`, com dimensões de referência 3×3. Exemplos simples 2×2 e o caso LU 3×3 do material podem ser carregados pelo seletor.

Não publicar o painel na internet sem configurar limites de recursos e acesso. Esta entrega usa execução local; não inclui autenticação ou persistência de sessões.


## Contas explícitas nos passos

Step.calculations guarda expressões LaTeX com os números substituídos nas fórmulas; Trace copia a lista e as matrizes. A interface abre a seção Contas deste passo e a exportação JSON preserva as expressões. Além das contas, cada passo pode registrar objetivo, operação, ordem de estados e destaques por célula para renderizar Antes/Depois com legenda. arithmetic.py formata números, somas de produtos, normas, divisões e entradas de produtos matriciais. Todos os métodos mostram a reconstrução e os resíduos com contas por componente. Householder/Givens mostram também os produtos usados para atualizar Q e R.
