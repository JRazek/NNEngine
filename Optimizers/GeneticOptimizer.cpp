//
// Created by user on 08.11.2021.
//

#include "GeneticOptimizer.h"
#include <map>
#include "../Network/Network.h"
#include <random>

cn::GeneticOptimizer::GeneticOptimizer(int _populationSize, float _mutationFactor, int seed)
        : populationSize(_populationSize), mutationFactor(_mutationFactor), seed(seed)
{}

void cn::GeneticOptimizer::setScore(int netID, double score) {
    population[netID].second = score;
}

void cn::GeneticOptimizer::reproducePopulation() {
    double sum = 0;
    std::map<double, Network *> populationSet;

    for(auto &p : population){
        sum += p.second;
        populationSet[sum] = &p.first;
    }

    std::vector<std::pair<Network, double>> newPopulation;
    newPopulation.reserve(populationSize);

    std::uniform_real_distribution<double> dist(0, sum);
    std::default_random_engine engine(seed);
    for(auto i = 0; i < populationSize; i ++){
        const Network& n1 = *populationSet.lower_bound(dist(engine))->second;
        const Network& n2 = *populationSet.lower_bound(dist(engine))->second;
        for(auto j = 0; j < n1.layersCount(); j++){
            newPopulation.push_back({reproduceIndividuals(n1, n2, seed), 0});
        }
    }
    population = std::move(newPopulation);
}

cn::Network cn::GeneticOptimizer::reproduceIndividuals(const Network &n1, const Network &n2, int seed) {
    for(u_int j = 0; j < n1.layersCount(); j++){
        n1.layers[j]->reproduce(n2.layers[j].get(), seed);
    }
}

const std::vector<std::pair<cn::Network, double>> &cn::GeneticOptimizer::getPopulation(u_int id) {
    return population;
}
