//
// Created by user on 08.11.2021.
//

#ifndef NEURALFLOWS_GENETICOPTIMIZER_H
#define NEURALFLOWS_GENETICOPTIMIZER_H
#include <vector>
#include <zconf.h>

namespace cn {
    class Network;
    class GeneticOptimizer {
        int populationSize;
        std::vector<std::pair<Network, double>> population; //net pointer, score
        float mutationFactor;
        int seed;
    public:
        GeneticOptimizer(int _populationSize, float mutationFactor, int seed = 1);

        /**
         *
         * @param netID
         * to use with custom fitness function
         */
        void setScore(int netID, double score);
        void reproducePopulation();


        const std::vector<std::pair<Network, double>>& getPopulation(u_int id);

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
