add_library(
    GeneticLearningUnit
    GeneticLearningUnit.h
    GeneticLearningUnit.cpp
)
target_link_libraries(GeneticLearningUnit PRIVATE Net)
target_include_directories(GeneticLearningUnit PUBLIC "../neuralNetwork/")
target_include_directories(GeneticLearningUnit PUBLIC "../../snakeEngine/")
target_link_libraries(GeneticLearningUnit PUBLIC csnake)