
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Net)
target_link_libraries(${PROJECT_NAME} PUBLIC GeneticLearningUnit)
target_include_directories(${PROJECT_NAME} PRIVATE "../neuralAlgorithms")