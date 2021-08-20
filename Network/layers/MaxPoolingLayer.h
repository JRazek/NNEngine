//
// Created by user on 20.08.2021.
//

#ifndef NEURALNETLIBRARY_MAXPOOLINGLAYER_H
#define NEURALNETLIBRARY_MAXPOOLINGLAYER_H
#include "interfaces/Layer.h"

namespace cn {
    class MaxPoolingLayer : public Layer{
    public:
        const int kernelSizeX, kernelSizeY;
        MaxPoolingLayer(int _id, int _kernelSizeX, int _kernelSizeY, cn::Network *_network);
    };
}

#endif //NEURALNETLIBRARY_MAXPOOLINGLAYER_H
