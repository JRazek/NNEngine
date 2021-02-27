#pragma once
#include <netStructure/Layer/Layer.h>

struct CLayer : Layer{
    struct Kernel{
        struct Matrix{
            Matrix(int sizeX, int sizeY);
            std::vector< std::vector< float > > weights;
        };
        float bias;
        std::vector<Matrix> matrices;
        Kernel(int x, int y, int z);
    };
    CLayer(int id, Net * net, int tensorCount, int tensorDepth, int matrixSizeX, int matrixSizeY);
    void run(const std::vector<float> &input);
    void initWeights();
    std::vector<Kernel> tensors;

    ~CLayer();
};