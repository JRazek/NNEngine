#pragma once
#include <netStructure/Layer/Layer.h>

struct CLayer : Layer{
    CLayer(int id, Net * net, int kernelSizeX, int kernelSizeY);
    void run(const std::vector<float> &input);
    ~CLayer();
    struct Tensor{
        struct Matrix{
            Matrix(int sizeX, int sizeY);
            std::vector< std::vector< float > > weights;
        };
        std::vector<Matrix> matrices;
        Tensor(int x, int y, int z);
    };
};