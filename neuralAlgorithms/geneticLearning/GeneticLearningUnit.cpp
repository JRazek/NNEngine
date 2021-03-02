#include "GeneticLearningUnit.h"
#include <stdlib.h>
#include <iostream>
GeneticLearningUnit::GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration):structure(structure){
    this->goalGenerations = goalGenerations;
    this->individualsPerGeneration = individualsPerGeneration;
}
void GeneticLearningUnit::start(int seed = 0){
    for(int i = 0; i < this->individualsPerGeneration; i ++){
        srand(seed);
        this->currIndividuals.push_back(new Net(this->structure, (rand() % 1000)));
        std::cout<<"";
    }
}