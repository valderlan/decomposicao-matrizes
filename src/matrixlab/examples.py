EXAMPLES = {
    "Simétrica positiva definida (3 × 3)": {
        "matrix": "4 12 -16\n12 37 -43\n-16 -43 98",
        "b": "-20 -43 192",
        "description": "Permite comparar LU, Cholesky, QR e SVD. Solução de referência: x=(1,2,3).",
    },
    "QR • exemplo da Aula 10 (3 × 3)": {
        "matrix": "0 1 1\n1 0 1\n1 1 0",
        "b": "5 4 3",
        "description": "Matriz das páginas 20–27 da Aula 10. Escolha Gram–Schmidt para acompanhar projeções, normas e formação de R. Com este b, x=(1,2,3).",
    },
    "LU • exemplo da Seção 1.3 (3 × 3)": {
        "matrix": "1 2 3\n1 4 7\n-2 2 5",
        "b": "1 -1 -7",
        "description": "Exemplo do material de LU: m21=1, m31=-2 e m32=3. Sem pivotamento, y=(1,-2,1) e x=(2,1,-1).",
    },
    "LU • exemplo simples do guia (2 × 2)": {
        "matrix": "2 1\n4 3",
        "b": "5 11",
        "description": "Sem pivotamento, m21=2, L tem diagonal unitária e U tem segunda linha (0,1). Solução x=(2,1).",
    },
    "LU • fatores inteiros (3 × 3)": {
        "matrix": "2 1 1\n4 5 3\n2 7 7",
        "b": "7 23 37",
        "description": "Escolha Doolittle sem pivotamento: multiplicadores 2, 1 e 2; "
        "diagonal de U=(2,3,4). Todos os fatores são inteiros, y=(7,9,12) e x=(1,2,3).",
    },
    "Cholesky • exemplo simples do guia (2 × 2)": {
        "matrix": "4 2\n2 3",
        "b": "8 8",
        "description": "L tem entradas 2, 1 e raiz de 2. LDLᵀ tem D=diag(4,2). Solução x=(1,2).",
    },
    "QR • exemplo simples do guia (2 × 2)": {
        "matrix": "1 1\n1 0",
        "b": "3 1",
        "description": "Gram–Schmidt: q1=(1,1)/raiz(2), projeção de a2=(1/2,1/2), q2=(1,-1)/raiz(2). Solução x=(1,2).",
    },
    "SVD • exemplo simples do guia (2 × 2)": {
        "matrix": "3 0\n0 1",
        "b": "6 2",
        "description": "Valores singulares 3 e 1, autovalores da gramiana 9 e 1. Solução x=(2,2); a primeira direção concentra 90% da energia Frobenius ao quadrado.",
    },
    "SVD • autovetores e fatores simples (2 × 2)": {
        "matrix": "2 1\n1 2",
        "b": "4 5",
        "description": "AᵀA=[[5,4],[4,5]], autovalores 9 e 1, σ=(3,1). "
        "As direções são (1,1)/raiz(2) e (1,-1)/raiz(2), com possíveis mudanças de sinal. Solução x=(1,2).",
    },
    "Exige pivotamento (3 × 3)": {
        "matrix": "0 2 1\n1 1 0\n2 0 1",
        "b": "7 3 5",
        "description": "O primeiro pivô é zero: LU sem pivotamento falha. Com pivotamento, x=(1,2,3).",
    },
    "Singular consistente (3 × 3)": {
        "matrix": "1 2 3\n2 4 6\n0 1 1",
        "b": "6 12 2",
        "description": "As linhas são dependentes. Há infinitas soluções; SVD retorna uma de norma mínima.",
    },
    "Singular inconsistente (2 × 2)": {
        "matrix": "1 2\n2 4",
        "b": "1 3",
        "description": "Ax=b não tem solução exata. SVD retorna mínimos quadrados; examine Ax−b.",
    },
    "Mínimos quadrados / ajuste de reta (4 × 2)": {
        "matrix": "1 0\n1 1\n1 2\n1 3",
        "b": "1 2 2 4",
        "description": "Ajustamos y≈c₀+c₁t. QR e SVD fornecem intercepto 0.9 e inclinação 0.9.",
    },
    "Subdeterminado (2 × 3)": {
        "matrix": "1 0 1\n0 1 1",
        "b": "1 1",
        "description": "Há mais variáveis que equações. A solução de norma mínima é (1/3,1/3,2/3).",
    },
    "Simétrica indefinida (2 × 2)": {
        "matrix": "1 2\n2 1",
        "b": "3 3",
        "description": "Ser simétrica não basta para Cholesky. Esta matriz tem um autovalor negativo.",
    },
    "Matriz nula (2 × 2)": {
        "matrix": "0 0\n0 0",
        "b": "0 0",
        "description": "SVD e QR continuam possíveis; x=0 é a solução de norma mínima para b=0.",
    },
}
