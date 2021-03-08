#include <iostream>
#include <vector>
#include "geneticLearning/GeneticLearningUnit.h"

int main(){
    //{ layerNum:{type, tensorsCount, matrixSizeX, matrixSizeY, tensorDepth} }

    
    std::vector< std::vector < int > > structure = {{1, 2, 3, 3, 1}, {1, 1, 3, 3, 2}, {2, 2, 2}, {0, 16, 64}, {0, 4, 16}, {0, 4, 4}};
    GeneticLearningUnit learningUnit = GeneticLearningUnit(structure, 10000, 1000);
    learningUnit.initPopulation(false, 4534);

    while(learningUnit.getGenerationNum() != learningUnit.goalGenerations - 1){
        learningUnit.runCurrentGeneration();
        learningUnit.initPopulation(true, 2);
    }
}