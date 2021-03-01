#include "GeneticLearningUnit.h"
GeneticLearningUnit::GeneticLearningUnit(const std::vector< std::vector < float> > structure, int goalGenerations, int individualsPerGeneration):structure(structure){
    this->goalGenerations = goalGenerations;
    this->individualsPerGeneration = individualsPerGeneration;
}
void GeneticLearningUnit::start(){
    for(int i = 0; i < this->individualsPerGeneration; i ++){
        
    }
}