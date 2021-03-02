#pragma once
#include <vector>
#include "Net.h"
class GeneticLearningUnit{
    std::vector<Net *> currIndividuals;
public:
    const std::vector< std::vector < int > > structure ;
    int generationNum = 0;

    int goalGenerations;
    int individualsPerGeneration;
    GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration);
    void start(int seed);
};