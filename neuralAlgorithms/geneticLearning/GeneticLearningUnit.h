#pragma once
#include <vector>
#include "Net.h"
#include "csnake.hpp"
class GeneticLearningUnit{
    std::vector< std::pair<Net *, csnake::Renderer *> > currIndividuals;
    int generationNum = 0;

public:
    const std::vector< std::vector < int > > structure ;


    /**
        map config for each individual
    */
    const int mapSizeX = 10;
    const int mapSizeY = 20;
    const int goalGenerations;
    WINDOW * win;

    const int individualsPerGeneration;

    const float mutationRate = 0.08f;

    const bool impenetrableWalls = true;
    GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration);
    ~GeneticLearningUnit();



    /**
     * inits random population with seed given
    */
    void initPopulation(bool cross, int seed);

    /**
     * runs current population until it dies.
    */
    void runCurrentGeneration();

    int getGenerationNum() const;

private:
    /**
     * inits population with crossing individials
    */
    void initPopulation(std::vector< std::pair<Net *, csnake::Renderer *> > &currIndividuals, int seed);
};