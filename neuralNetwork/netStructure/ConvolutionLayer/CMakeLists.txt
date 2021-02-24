add_library(
    ConvolutionLayer
    CLayer.h
    CLayer.cpp
)
link_libraries(ConvolutionLayer PUBLIC Net)