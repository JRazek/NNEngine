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
    while(1){
        int deadCount = 0;
        for(auto k : this->currIndividuals){
            Tensor input = Tensor(k.second->getWidth(), k.second->getWidth(), 1);
            for(int y = 0; y < input.getY(); y++){
                for(int x = 0; x < input.getX(); x++){
                    //input.edit(x, y, 0, k.second.get)
                }
            }
        }
        if(deadCount == this->currIndividuals.size()){
            break;
        }
    }
}

GeneticLearningUnit::~GeneticLearningUnit(){
    for(auto k : currIndividuals){
        delete k.first;
        delete k.second;
    }
}