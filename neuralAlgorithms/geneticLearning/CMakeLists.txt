add_library(
    GeneticLearningUnit
    GeneticLearningUnit.h
    GeneticLearningUnit.cpp
)
include_directories(../)
link_libraries(GeneticLearningUnit Net)