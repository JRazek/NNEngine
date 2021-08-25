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
        MaxPoolingLayer(int _id, Network &_network, int _kernelSizeX, int _kernelSizeY);
        void run(const Bitmap<float> &bitmap) override;
        float getChain(const Vector3<float> &input) override;
    };
}

#endif //NEURALNETLIBRARY_MAXPOOLINGLAYER_H
