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
        std::vector<std::pair<Network *, double>> population; //net pointer, score
        int seed;
    public:
        GeneticOptimizer(int _populationSize, int seed = 1);
        void setScore(int netID, double score);
        void reproducePopulation();

        /**
         *
         * @param n1 parent 1
         * @param n2 parent 2
         * @return child
         * [TODO]
         */
        static Network reproduceIndividuals(const Network &n1, const Network &n2, int seed = 1);
    };
}

#endif //NEURALFLOWS_GENETICOPTIMIZER_H
