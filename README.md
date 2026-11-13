# Estudos em C++

Algoritmos de machine learning implementados do zero em C++17, sem bibliotecas externas. Cada pasta é um projeto independente.

## 1. K-NN do zero (`01_knn`)

Classificador K-NN com distância euclidiana e Manhattan sobre três nuvens gaussianas em 2D, com sobreposição proposital entre as classes:

- divisão em treino, validação e teste, com o `k` escolhido na validação e o teste usado uma única vez, para a escolha não contaminar a avaliação;
- padronização (z-score) ajustada só no treino, para não vazar informação de validação e teste;
- `std::partial_sort` para ordenar só os `k` vizinhos mais próximos, com custo O(n log k) por previsão em vez de O(n log n);
- desempate determinístico, tanto na distância quanto nos votos.

| Resultado | Valor |
|---|---|
| Amostras (treino, validação, teste) | 540, 180, 180 |
| Acurácia na validação, `k` entre 3 e 51 | cerca de 0,90 a 0,92 |
| Acurácia na validação, `k = 1` | 0,867 (decora o ruído do treino) |
| Melhor `k` (euclidiana, pela validação) | 5 |
| Acurácia no teste | 0,878 |
| Tempo para prever 180 amostras | cerca de 0,6 ms |

Com 180 amostras de validação, um acerto a mais vale 0,0056, então as diferenças entre os valores de `k` de 3 a 51 estão dentro do ruído. O único `k` claramente pior é o 1. A acurácia de teste ficou abaixo da de validação, como esperado, porque o melhor `k` foi escolhido na validação.

## Como compilar e executar

Requer um compilador com suporte a C++17 (testado com g++ 16.2 do MSYS2).

```bash
cd 01_knn
g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o knn
./knn
```

No PowerShell, rode com `.\knn`.
