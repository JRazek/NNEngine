add_library(
    GeneticLearningUnit
    GeneticLearningUnit.h
    GeneticLearningUnit.cpp
)
target_link_libraries(GeneticLearningUnit PRIVATE Net)
#include_directories(.)
target_include_directories(GeneticLearningUnit PUBLIC "../neuralNetwork/")