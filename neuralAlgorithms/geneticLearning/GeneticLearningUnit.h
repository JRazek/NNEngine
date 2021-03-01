#pragma once
#include <vector>
#include <neuralNetwork/Net.h>
class GeneticLearningUnit{
    std::vector<Net> currIndividuals;
public:
    const std::vector< std::vector < float> > structure ;
    int generationNum = 0;

    int goalGenerations;
    int individualsPerGeneration;
    GeneticLearningUnit(const std::vector< std::vector < float> > structure, int goalGenerations, int individualsPerGeneration);
    void start();
};