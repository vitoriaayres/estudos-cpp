#pragma once 

#include <algorithm>
#include <cmath> 
#include <cstddef> 
#include <random> 
#include <vector> 

struct Amostra { 
    std::vector<double> x; 
    int y;
}; 
using Conjunto = std::vector<Amostra>;

inline Conjunto gerar_gaussianas(std::size_t por_classe, unsigned semente) { 
    const std::vector<std::vector<double>> centros = {{0.0, 0.0}, {3.0, 3.0}, {0.0, 4.0}}; 
    const double desvio = 1.2; 

    std::mt19937 gerador(semente);
    std::normal_distribution<double> ruido(0.0, desvio);
    
    Conjunto dados;
    dados.reserve(por_classe * centros.size());
    for (std::size_t c = 0; c < centros.size(); ++c) {
        for (std::size_t i = 0; i < por_classe; ++i) {
            const double x0 = centros[c][0] + ruido(gerador);
            const double x1 = centros[c][1] + ruido(gerador);
            dados.push_back({{x0, x1}, static_cast<int>(c)});
        }
    }
    return dados;
}
inline void dividir(Conjunto dados, double proporcao_a, unsigned semente,
                    Conjunto& parte_a, Conjunto& parte_b) {
    std::mt19937 gerador(semente);
    std::shuffle(dados.begin(), dados.end(), gerador);

    const auto corte = static_cast<std::ptrdiff_t>(
        static_cast<double>(dados.size()) * proporcao_a);
    parte_a.assign(dados.begin(), dados.begin() + corte);
    parte_b.assign(dados.begin() + corte, dados.end());
}
struct Padronizador {
    std::vector<double> media;
    std::vector<double> desvio;

    void ajustar(const Conjunto& conjunto) {
        const std::size_t dimensoes = conjunto.front().x.size();
        const auto n = static_cast<double>(conjunto.size());
        media.assign(dimensoes, 0.0);
        desvio.assign(dimensoes, 0.0);

        for (const auto& a : conjunto)
            for (std::size_t j = 0; j < dimensoes; ++j) media[j] += a.x[j] / n;

        for (const auto& a : conjunto)
            for (std::size_t j = 0; j < dimensoes; ++j) {
                const double d = a.x[j] - media[j];
                desvio[j] += d * d / n;
            }

        for (auto& d : desvio) {
            d = std::sqrt(d);
            if (d == 0.0) d = 1.0;   // variável constante: evita dividir por zero
        }
    }

    void aplicar(Conjunto& conjunto) const {
        for (auto& a : conjunto)
            for (std::size_t j = 0; j < a.x.size(); ++j)
                a.x[j] = (a.x[j] - media[j]) / desvio[j];
    }
};