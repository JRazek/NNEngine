#pragma once
#include <vector>
#include "Net.h"
#include "csnake.hpp"
class GeneticLearningUnit{
    std::vector< std::pair<Net *, csnake::SnakeGame *> > currIndividuals;
public:
    const std::vector< std::vector < int > > structure ;
    int generationNum = 0;

    const int goalGenerations;
    const int individualsPerGeneration;

    /*
        map config for each individual
    */
    const int mapSizeX = 10;
    const int mapSizeY = 10;
    const bool impenetrableWalls = true;
    GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration);
    ~GeneticLearningUnit();
    void start(int seed);
};