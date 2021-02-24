add_library(Layer INTERFACE)
target_include_directories(Layer INTERFACE .)
link_libraries(Layer PUBLIC Net)