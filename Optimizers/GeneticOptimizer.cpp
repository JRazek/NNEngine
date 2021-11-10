//
// Created by user on 08.11.2021.
//

#include "GeneticOptimizer.h"
#include <algorithm>
#include <map>
#include "../Network/Network.h"
#include <random>

cn::GeneticOptimizer::GeneticOptimizer(int _populationSize, int _seed) : populationSize(_populationSize), seed(_seed) {

}

void cn::GeneticOptimizer::setScore(int netID, double score) {
    population[netID].second = score;
}

void cn::GeneticOptimizer::reproducePopulation() {
    double sum = 0;
    std::map<double, Network *> populationSet;

    for(auto &p : population){
        sum += p.second;
        populationSet[sum] = p.first;
    }

    std::uniform_real_distribution<double> dist(0, sum);
    std::default_random_engine engine(seed);
    for(auto i = 0; i < populationSize; i ++){
        const Network &n1 = *populationSet.lower_bound(dist(engine))->second;
        const Network &n2 = *populationSet.lower_bound(dist(engine))->second;
        for(auto j = 0; j < n1.layersCount(); j++){

        }
    }
}

cn::Network cn::GeneticOptimizer::reproduceIndividuals(const Network &n1, const Network &n2, int seed) {
//    return cn::Network();
}
