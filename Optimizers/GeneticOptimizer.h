//
// Created by user on 08.11.2021.
//

#ifndef NEURALFLOWS_GENETICOPTIMIZER_H
#define NEURALFLOWS_GENETICOPTIMIZER_H
#include <vector>

namespace cn {
    class Network;
    class GeneticOptimizer {
        int populationSize;
        std::vector<Network *> population;
    public:
        GeneticOptimizer(int _populationSize);
    };
}

#endif //NEURALFLOWS_GENETICOPTIMIZER_H
