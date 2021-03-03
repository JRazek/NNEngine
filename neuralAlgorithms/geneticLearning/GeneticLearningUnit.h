#pragma once
#include <vector>
#include "Net.h"
#include "csnake.hpp"
class GeneticLearningUnit{
    std::vector<Net *> currIndividuals;
public:
    const std::vector< std::vector < int > > structure ;
    int generationNum = 0;
    csnake::SnakeGame game = csnake::SnakeGame(20, 20, false);

    int goalGenerations;
    int individualsPerGeneration;
    GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration);
    void start(int seed);
};