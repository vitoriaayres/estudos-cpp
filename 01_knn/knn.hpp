#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dados.hpp"

enum class Metrica { Euclidiana, Manhattan };

inline double distancia(const std::vector<double>& a, const std::vector<double>& b,
                        Metrica metrica) {
    double soma = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        soma += (metrica == Metrica::Euclidiana) ? d * d : std::abs(d);
    }
    return (metrica == Metrica::Euclidiana) ? std::sqrt(soma) : soma;
}

class KNN {
public:
    KNN(std::size_t k, Metrica metrica) : k_(k), metrica_(metrica) {
        if (k == 0) throw std::invalid_argument("k deve ser maior que zero");
    }

    // O K-NN não "aprende": só guarda o treino. O custo vem na hora de prever.
    void ajustar(const Conjunto& treino) {
        if (treino.empty()) throw std::invalid_argument("treino vazio");
        if (k_ > treino.size()) throw std::invalid_argument("k maior que o tamanho do treino");

        treino_ = treino;
        n_classes_ = 0;
        for (const auto& a : treino_)
            n_classes_ = std::max(n_classes_, static_cast<std::size_t>(a.y) + 1);
    }

    int prever(const std::vector<double>& x) const {
        std::vector<std::pair<double, int>> vizinhos;
        vizinhos.reserve(treino_.size());
        for (const auto& a : treino_)
            vizinhos.emplace_back(distancia(x, a.x, metrica_), a.y);

    
        const auto fim = vizinhos.begin() + static_cast<std::ptrdiff_t>(k_);
        std::partial_sort(vizinhos.begin(), fim, vizinhos.end());

        std::vector<int> votos(n_classes_, 0);
        for (std::size_t i = 0; i < k_; ++i) ++votos[static_cast<std::size_t>(vizinhos[i].second)];

        // Em empate de votos, vence a classe de menor índice
        return static_cast<int>(std::max_element(votos.begin(), votos.end()) - votos.begin());
    }

    double acuracia(const Conjunto& conjunto) const {
        std::size_t acertos = 0;
        for (const auto& a : conjunto)
            if (prever(a.x) == a.y) ++acertos;
        return static_cast<double>(acertos) / static_cast<double>(conjunto.size());
    }

private:
    std::size_t k_;
    Metrica metrica_;
    Conjunto treino_;
    std::size_t n_classes_ = 0;
};