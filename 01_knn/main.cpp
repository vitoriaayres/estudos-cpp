#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dados.hpp"
#include "knn.hpp"

int main() {
    // 900 amostras: 80% treino+validação e 20% teste; depois 75% treino e 25% validação
    const Conjunto todos = gerar_gaussianas(300, 42);

    Conjunto treino_validacao, teste, treino, validacao;
    dividir(todos, 0.8, 1, treino_validacao, teste);
    dividir(treino_validacao, 0.75, 2, treino, validacao);

    Padronizador padronizador;
    padronizador.ajustar(treino);
    padronizador.aplicar(treino);
    padronizador.aplicar(validacao);
    padronizador.aplicar(teste);

    std::cout << "Treino: " << treino.size() << "  Validacao: " << validacao.size()
              << "  Teste: " << teste.size() << "\n\n";

    const std::vector<std::size_t> valores_k = {1, 3, 5, 7, 9, 11, 15, 21, 31, 51};

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Acuracia na validacao\n";
    std::cout << "  k   Euclidiana  Manhattan\n";

    std::size_t melhor_k = valores_k.front();
    double melhor_acuracia = -1.0;

    for (const std::size_t k : valores_k) {
        KNN euclidiana(k, Metrica::Euclidiana);
        euclidiana.ajustar(treino);
        KNN manhattan(k, Metrica::Manhattan);
        manhattan.ajustar(treino);

        const double acc_euclidiana = euclidiana.acuracia(validacao);
        const double acc_manhattan = manhattan.acuracia(validacao);

        std::cout << std::setw(3) << k << "   " << std::setw(10) << acc_euclidiana
                  << "  " << std::setw(9) << acc_manhattan << "\n";

        if (acc_euclidiana > melhor_acuracia) {
            melhor_acuracia = acc_euclidiana;
            melhor_k = k;
        }
    }

    // O teste só é usado uma vez, com o k escolhido na validação
    KNN final_(melhor_k, Metrica::Euclidiana);
    final_.ajustar(treino);

    const auto inicio = std::chrono::steady_clock::now();
    const double acuracia_teste = final_.acuracia(teste);
    const auto fim = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(fim - inicio).count();

    std::cout << "\nMelhor k (euclidiana, pela validacao): " << melhor_k << "\n";
    std::cout << "Acuracia no teste: " << acuracia_teste << "\n";
    std::cout << "Tempo para prever " << teste.size() << " amostras: " << ms << " ms\n";

    return 0;
}