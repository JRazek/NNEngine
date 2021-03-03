#include "GeneticLearningUnit.h"
#include <stdlib.h>
#include <iostream>
GeneticLearningUnit::GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration):
    structure(structure), goalGenerations(goalGenerations), individualsPerGeneration(individualsPerGeneration){}
    
void GeneticLearningUnit::start(int seed = 0){
    for(int i = 0; i < this->individualsPerGeneration; i ++){
        srand(seed);
        csnake::SnakeGame * game = new csnake::SnakeGame(this->mapSizeX, this->mapSizeY, this->impenetrableWalls);
        Net * net = new Net(this->structure, rand() % (int)1e9+7);
        this->currIndividuals.push_back({net, game});
    }
}
GeneticLearningUnit::~GeneticLearningUnit(){
    for(auto k : currIndividuals){
        delete k.first;
        delete k.second;
    }
}