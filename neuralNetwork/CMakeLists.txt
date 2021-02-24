add_subdirectory(netStructure)
add_library(
    Net
    Net.h
    Net.cpp
)
target_link_libraries(Net PRIVATE FFLayer)